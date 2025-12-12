#!/usr/bin/env python
# coding: utf-8
"""
Direct Preference Optimization (DPO) training script.

- Expects a preprocessed Anthropic HH-RLHF-style dataset saved by your data
  preprocessing pipeline at ./processed_data, with 'train' split containing
  'chosen' and 'rejected' conversation strings.
- Trains a GPT-2 policy via DPO using a frozen GPT-2 reference model.
- Only trains and saves the policy; no evaluation or GPT-4 judging here.

Usage (typical):
    python DPO_training.py
"""

import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import sys
from tqdm import tqdm
tqdm_disable = not sys.stdout.isatty()
from datasets import load_dataset  # add if missing

MIN_LENGTH = 10
MAX_LENGTH = 1024
CHAR_LIMIT = MAX_LENGTH * 6

def filter_edge_cases(example):
    chosen = example["chosen"]
    rejected = example["rejected"]
    if chosen == rejected:
        return False
    if len(chosen) < MIN_LENGTH or len(rejected) < MIN_LENGTH:
        return False
    if len(rejected) == 0:
        return False
    ratio = len(chosen) / len(rejected)
    if ratio < 0.5 or ratio > 2.0:
        return False
    if len(chosen) >= CHAR_LIMIT or len(rejected) >= CHAR_LIMIT:
        return False
    return True

def load_dpo_source(config):
    """Return an HF dataset split that has 'chosen' and 'rejected' columns."""
    # Try processed_data if it already contains raw text (some pipelines do)
    if os.path.exists(config.data_dir):
        ds = load_from_disk(config.data_dir)
        split = ds.get(config.train_split, None)
        if split is not None:
            feat_names = set(split.features.keys())
            if "chosen" in feat_names and "rejected" in feat_names:
                return split

    # Fallback: load raw anthropic/hh-rlhf and apply the same cleaning as in the notebook
    raw = load_dataset("anthropic/hh-rlhf")
    train_ds = raw["train"]
    cleaned = train_ds.filter(filter_edge_cases, num_proc=4)
    return cleaned

# -----------------------------
# Device setup
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class DPOConfig:
    # Data & model paths
    data_dir: str = "./processed_data"        # must contain a 'train' split with chosen/rejected
    train_split: str = "train"
    base_model_name: str = "gpt2"
    output_dir: str = "./dpo_rlhf_model"

    # Training hyperparameters
    num_epochs: int = 1                       # keep modest by default
    batch_size: int = 4
    grad_accum_steps: int = 8                 # effective batch = batch_size * grad_accum_steps
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    beta: float = 0.1                         # DPO temperature
    max_seq_len: int = 512

    # Logging / checkpointing
    log_every: int = 20                       # log every N optimizer steps
    save_every: int = 500                     # save checkpoint every N optimizer steps
    max_train_examples: Optional[int] = None  # if not None, truncate train set

    # Precision
    fp16: bool = torch.cuda.is_available()    # enable mixed precision on GPU


# -----------------------------
# Conversation utilities
# -----------------------------
def split_conversation(full_text: str):
    """
    Split Anthropic HH-RLHF-style text into (prompt, last_assistant_reply).

    The format is typically:
        "Human: ...\\n\\nAssistant: ..."

    We use the last Human: / Assistant: pair to define the prompt region and
    the final assistant reply.
    """
    text = full_text.strip()
    last_human_idx = text.rfind("Human:")
    if last_human_idx == -1:
        # Fallback: no 'Human:' marker, treat whole thing as prompt
        return text, ""

    assistant_idx = text.find("Assistant:", last_human_idx)
    if assistant_idx == -1:
        # No assistant after last human; treat everything up to last human as prompt
        prompt = text[:last_human_idx]
        return prompt, text[last_human_idx:]

    prompt_end = assistant_idx + len("Assistant:")
    prompt = text[:prompt_end]
    last_assistant = text[prompt_end:]
    return prompt.strip(), last_assistant.strip()


# -----------------------------
# Dataset & collator
# -----------------------------
class DPOPairDataset(Dataset):
    """
    Wrap a HuggingFace Dataset split and produce (prompt, chosen, rejected) triples
    for DPO training.
    """

    def __init__(self, hf_dataset):
        items: List[Dict[str, str]] = []
        for ex in hf_dataset:
            pc, chosen_tail = split_conversation(ex["chosen"])
            pr, rejected_tail = split_conversation(ex["rejected"])

            # Use whichever prompt parse is non-empty, as in your notebook DPO logic.
            prompt = pc if pc.strip() else pr

            if not prompt.strip():
                continue
            if not chosen_tail.strip() or not rejected_tail.strip():
                continue

            items.append(
                {
                    "prompt": prompt,
                    "chosen": chosen_tail,
                    "rejected": rejected_tail,
                }
            )

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


class DPOCollator:
    """
    Collate list of dicts with (prompt, chosen, rejected) into batched tensors.

    We simply concatenate `prompt + chosen` and `prompt + rejected` and let
    DPO's log-ratio difference cancel out the prompt portion, as in your
    assignment3 DPO implementation.
    """

    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_texts: List[str] = []
        rejected_texts: List[str] = []

        for f in features:
            prompt = f["prompt"]
            chosen_texts.append(prompt + f["chosen"])
            rejected_texts.append(prompt + f["rejected"])

        batch_chosen = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_rejected = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
        }


# -----------------------------
# Model utilities
# -----------------------------
def prepare_tokenizer_and_models(config: DPOConfig):
    """Load tokenizer, trainable policy model, and frozen reference model."""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    policy_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    policy_model.to(DEVICE)
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    policy_model.config.use_cache = False
    # Gradient checkpointing helps memory without extra math
    policy_model.gradient_checkpointing_enable()

    ref_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    ref_model.to(DEVICE)
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    return tokenizer, policy_model, ref_model


def seq_logprob(model: AutoModelForCausalLM,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute log p(x_1, ..., x_T) for each sequence, summing over all non-pad tokens.

    This matches the DPO objective in your notebook where we feed the full
    prompt + response and rely on chosen/rejected sharing the same prompt.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, L, V]

    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_shifted = log_probs[:, :-1, :]          # [B, L-1, V]
    target_tokens = input_ids[:, 1:]                  # [B, L-1]
    token_logprobs = log_probs_shifted.gather(
        2, target_tokens.unsqueeze(-1)
    ).squeeze(-1)                                     # [B, L-1]

    mask = attention_mask[:, 1:]                      # ignore pads
    seq_lp = (token_logprobs * mask).sum(dim=-1)      # [B]
    return seq_lp


def dpo_step(batch: Dict[str, torch.Tensor],
             policy_model: AutoModelForCausalLM,
             ref_model: AutoModelForCausalLM,
             config: DPOConfig,
             scaler: torch.amp.GradScaler):
    """
    Perform a single DPO update step over one batch (with gradient accumulation).

    Returns (loss_value, approx_kl_value) for logging.
    """
    input_ids_ch = batch["input_ids_chosen"].to(DEVICE)
    attn_ch = batch["attention_mask_chosen"].to(DEVICE)
    input_ids_re = batch["input_ids_rejected"].to(DEVICE)
    attn_re = batch["attention_mask_rejected"].to(DEVICE)

    # Reference log-probs are computed without gradients
    with torch.no_grad():
        ref_ch = seq_logprob(ref_model, input_ids_ch, attn_ch)
        ref_re = seq_logprob(ref_model, input_ids_re, attn_re)

    # Policy log-probs with optional mixed precision
    with torch.amp.autocast("cuda", enabled=config.fp16):
        pi_ch = seq_logprob(policy_model, input_ids_ch, attn_ch)
        pi_re = seq_logprob(policy_model, input_ids_re, attn_re)

        # log πθ / π_ref for chosen and rejected
        lograt_ch = pi_ch - ref_ch
        lograt_re = pi_re - ref_re

        # Standard DPO objective
        dpo_logits = config.beta * (lograt_ch - lograt_re)
        loss = -F.logsigmoid(dpo_logits).mean()

    # Backward with gradient accumulation
    if config.fp16:
        scaler.scale(loss / config.grad_accum_steps).backward()
    else:
        (loss / config.grad_accum_steps).backward()

    # Simple approximate KL diagnostic (not used in loss)
    with torch.no_grad():
        approx_kl = (lograt_ch - lograt_re).mean()

    return loss.detach().cpu().item(), approx_kl.detach().cpu().item()


# -----------------------------
# Main training routine
# -----------------------------
def train_dpo(config: DPOConfig):
    os.makedirs(config.output_dir, exist_ok=True)

    # 1) Load data from disk
    print(f"Loading dataset for DPO...")
    raw_train = load_dpo_source(config)

    if config.max_train_examples is not None:
        raw_train = raw_train.select(range(config.max_train_examples))
        print(f"Truncated train set to {config.max_train_examples} examples.")

    train_dataset = DPOPairDataset(raw_train)
    print(f"DPO train triples: {len(train_dataset)}")

    # 2) Prepare tokenizer and models
    tokenizer, policy_model, ref_model = prepare_tokenizer_and_models(config)

    collator = DPOCollator(tokenizer, max_length=config.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # 3) Optimizer, scheduler, scaler
    params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
    total_training_steps = config.num_epochs * steps_per_epoch
    warmup_steps = int(config.warmup_ratio * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    scaler = torch.amp.GradScaler(enabled=config.fp16)

    print(f"Total training steps (optimizer updates): {total_training_steps}")
    print(f"Using device: {DEVICE}")

    # 4) Training loop
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    for epoch in range(config.num_epochs):
        policy_model.train()
        pbar = tqdm(train_loader, desc=f"DPO epoch {epoch + 1}/{config.num_epochs}", disable = tqdm_disable)
        for step_idx, batch in enumerate(pbar):
            loss_val, approx_kl = dpo_step(batch, policy_model, ref_model, config, scaler)
            running_loss += loss_val

            # Perform optimizer step after grad_accum_steps batches
            if (step_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # Logging
                if global_step % config.log_every == 0:
                    avg_loss = running_loss / config.log_every
                    pbar.set_postfix(
                        step=global_step,
                        loss=f"{avg_loss:.4f}",
                        kl=f"{approx_kl:.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    running_loss = 0.0

                # Periodic checkpoint
                if config.save_every > 0 and global_step % config.save_every == 0:
                    ckpt_dir = os.path.join(config.output_dir, f"checkpoint_step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    policy_model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"Saved checkpoint to {ckpt_dir}")

    # 5) Final save
    print(f"Training complete. Saving final DPO policy to {config.output_dir}...")
    policy_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Done.")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    cfg = DPOConfig()
    train_dpo(cfg)

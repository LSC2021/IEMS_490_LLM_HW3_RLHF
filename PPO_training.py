#!/usr/bin/env python
# coding: utf-8

"""
PPO training script that reproduces the PPO behavior in assignment3.py,
using the reward model trained by reward_model_training.py and the
same HH-RLHF dataset filtering logic as data_preprocessing.py.

- Uses Anthropic HH-RLHF as the source of prompts.
- Re-applies the same filter_edge_cases() as in assignment3/data_preprocessing.
- Builds prompts by splitting the "chosen" conversation into (prompt, last assistant).
- Runs PPO training only (no validation or test-time evaluation).
"""

import os
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import sys
from tqdm import tqdm
tqdm_disable = not sys.stdout.isatty()
# ============================================================
# 0. Global config and device
# ============================================================

MODEL_NAME = "gpt2"
DATASET_NAME = "anthropic/hh-rlhf"
REWARD_MODEL_DIR = "./final_reward_model"   # produced by reward_model_training.py
PPO_OUTPUT_DIR = "./ppo_rlhf_model"         # final PPO policy output
PPO_CHECKPOINT_DIR = "./ppo_checkpoint"     # intermediate checkpoints

SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. Dataset filtering & prompt building
#    (mirrors assignment3 + data_preprocessing)
# ============================================================

MIN_LENGTH = 10
MAX_LENGTH = 1024
CHAR_LIMIT = MAX_LENGTH * 6  # same as in assignment3/data_preprocessing


def filter_edge_cases(example):
    """
    Filter out ties, short responses, length-mismatch, and too-long sequences.
    Same logic as assignment3/data_preprocessing.
    """
    chosen = example["chosen"]
    rejected = example["rejected"]

    # 1) Filter out identical responses (ties)
    if chosen == rejected:
        return False

    # 2) Filter short responses
    if len(chosen) < MIN_LENGTH or len(rejected) < MIN_LENGTH:
        return False

    # 3) Filter length outliers (length mismatch)
    if len(rejected) == 0:
        return False
    ratio = len(chosen) / len(rejected)
    if ratio < 0.5 or ratio > 2.0:
        return False

    # 4) Filter too-long sequences
    if len(chosen) >= CHAR_LIMIT or len(rejected) >= CHAR_LIMIT:
        return False

    return True


def split_conversation(full_text: str):
    """
    Same as assignment3:
    Split Anthropic HH-RLHF-style conversation into:
    - prompt: up to and including the last 'Assistant:' after the last 'Human:'.
    - last_assistant: the trailing assistant text after that.
    """
    text = full_text.strip()

    # Find the last "Human:"
    last_human_idx = text.rfind("Human:")
    if last_human_idx == -1:
        # No human marker; treat whole thing as prompt
        return text, ""

    # Find "Assistant:" after that
    assistant_idx = text.find("Assistant:", last_human_idx)
    if assistant_idx == -1:
        # No assistant after last human
        prompt = text[:last_human_idx]
        return prompt.strip(), text[last_human_idx:].strip()

    # Prompt is everything up to "Assistant:"
    prompt_end = assistant_idx + len("Assistant:")
    prompt = text[:prompt_end]
    last_assistant = text[prompt_end:]
    return prompt.strip(), last_assistant.strip()


def build_ppo_dataset(example):
    """
    For each cleaned HH-RLHF example, extract the prompt from the chosen side.
    """
    prompt, last_assistant = split_conversation(example["chosen"])
    return {
        "prompt": prompt,
        "chosen_tail": last_assistant,
    }


def load_ppo_prompts(train_subset_size: int = 50000):
    """
    Load HH-RLHF, apply the same filtering as assignment3/data_preprocessing,
    and build a list of prompts for PPO rollouts.
    """
    print(f"Loading dataset: {DATASET_NAME} ...")
    dataset = load_dataset(DATASET_NAME)

    train_ds = dataset["train"].select(range(train_subset_size))
    test_ds = dataset["test"]
    print(f"Original train size: {len(train_ds)}")
    print(f"Original test size:  {len(test_ds)}")

    print("Filtering edge cases (same as assignment3/data_preprocessing)...")
    cleaned_train_ds = train_ds.filter(filter_edge_cases, num_proc=4)
    print(f"Filtered train size: {len(cleaned_train_ds)}")

    print("Building PPO prompts from 'chosen' conversations...")
    ppo_base = cleaned_train_ds.map(build_ppo_dataset)
    prompts = ppo_base["prompt"]
    print(f"Number of PPO prompts: {len(prompts)}")

    return prompts


# ============================================================
# 2. PPO configuration (same as assignment3 PPOConfig)
# ============================================================

@dataclass
class PPOConfig:
    total_steps: int = 1500
    rollout_batch_size: int = 8
    ppo_epochs: int = 2
    gradient_accumulation_steps: int = 1

    clip_range: float = 0.1
    kl_coef: float = 0.05
    target_kl: float = 0.3
    adapt_kl: bool = True
    entropy_coef: float = 0.1
    target_entropy: float = 1.8
    value_coef: float = 0.3

    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    gamma: float = 1.0
    lam: float = 0.95

    max_prompt_length: int = 256
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0

    save_every: int = 100
    log_every: int = 1
    fp16: bool = torch.cuda.is_available()


# ============================================================
# 3. PPO helper utilities (mirroring assignment3)
# ============================================================

def sample_prompts(all_prompts, n: int):
    """Randomly sample prompts from the preprocessed preference dataset."""
    idxs = random.sample(range(len(all_prompts)), n)
    return [all_prompts[i] for i in idxs]


def generate_rollout(batch_prompts,
                     policy_model,
                     value_head,
                     ref_policy,
                     tokenizer,
                     config: PPOConfig,
                     device):
    """
    Generate responses, compute token-level logprobs and values,
    and mark the response region tokens for PPO (same logic as assignment3).
    """
    policy_model.eval()

    enc = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_prompt_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        gen_outputs = policy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else None,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
        )

    sequences = gen_outputs.sequences.to(device)
    seq_attention = (sequences != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = policy_model(
            input_ids=sequences,
            attention_mask=seq_attention,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]
        values = value_head(hidden).squeeze(-1)  # [B, L]

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_logprobs = log_probs.gather(
            2, sequences[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # [B, L-1]

        ref_outputs = ref_policy(input_ids=sequences, attention_mask=seq_attention)
        ref_log_probs = F.log_softmax(ref_outputs.logits[:, :-1, :], dim=-1)
        ref_token_logprobs = ref_log_probs.gather(
            2, sequences[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # [B, L-1]

    B, L = sequences.shape
    response_mask = torch.zeros_like(token_logprobs, dtype=torch.bool)
    responses = []

    # Identify response region: from last prompt token to last non-pad token
    for i in range(B):
        # Non-pad positions in the original prompt (left padded)
        nonpad_prompt = (input_ids[i] != tokenizer.pad_token_id).nonzero(as_tuple=False).squeeze(-1)
        if nonpad_prompt.numel() == 0:
            responses.append("")
            continue

        last_prompt_idx = nonpad_prompt[-1].item()

        # Non-pad positions in the full sequence (prompt + response)
        nonpad_seq = (sequences[i] != tokenizer.pad_token_id).nonzero(as_tuple=False).squeeze(-1)
        if nonpad_seq.numel() == 0:
            responses.append("")
            continue

        last_token_idx = nonpad_seq[-1].item()

        # token_logprobs[t] = log p(sequences[t+1] | sequences[:t+1])
        # So first response token at sequences[last_prompt_idx + 1]
        # corresponds to token_logprobs index last_prompt_idx.
        start = last_prompt_idx
        end = last_token_idx  # exclusive in token_logprobs space

        if end > start:
            response_mask[i, start:end] = True

        # Slice and decode response tokens
        resp_start = last_prompt_idx + 1
        resp_end = last_token_idx + 1  # exclusive
        resp_tokens = sequences[i, resp_start:resp_end]
        responses.append(tokenizer.decode(resp_tokens, skip_special_tokens=True))

    policy_model.train()

    return {
        "prompts": batch_prompts,
        "sequences": sequences,
        "attention_mask": seq_attention,
        "old_logprobs": token_logprobs,
        "ref_logprobs": ref_token_logprobs,
        "values": values,
        "response_mask": response_mask,
        "responses": responses,
    }


def score_with_reward_model(prompts, responses, tokenizer, reward_model, device):
    """
    Concatenate prompt + response and pass through the scalar reward model.
    Same idea as in assignment3.
    """
    texts = [p + r for p, r in zip(prompts, responses)]

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    with torch.no_grad():
        outputs = reward_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        scores = outputs.logits.squeeze(-1)

    return scores


def compute_advantages(rewards, values, response_mask, gamma, lam):
    """
    Simple baseline advantage:
    - Compute sequence-level baseline as mean value over response tokens.
    - Advantage = reward - baseline
    - Return is baseline + advantage.
    This matches the simplified approach used in assignment3.
    """
    value_tokens = values[:, :-1]  # align with token_logprobs shape
    lengths = response_mask.sum(1).clamp(min=1)
    value_means = (value_tokens * response_mask).sum(1) / lengths  # [B]

    advantages = rewards - value_means
    returns = advantages + value_means

    # Normalize advantages across the batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages.detach(), returns.detach(), value_means.detach(), lengths


def ppo_update(policy_model,
               value_head,
               rollout,
               advantages,
               returns,
               config: PPOConfig,
               scaler,
               device):
    """
    PPO update step (same structure as assignment3).
    """
    sequences = rollout["sequences"].to(device)
    attn = rollout["attention_mask"].to(device)
    response_mask = rollout["response_mask"].to(device)
    old_logprobs = rollout["old_logprobs"].to(device)
    ref_logprobs = rollout["ref_logprobs"].to(device)

    lengths = response_mask.sum(1).clamp(min=1)

    with torch.amp.autocast("cuda", enabled=config.fp16):
        outputs = policy_model(
            input_ids=sequences,
            attention_mask=attn,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]
        values = value_head(hidden).squeeze(-1)

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_logprobs = log_probs.gather(
            2, sequences[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # [B, L-1]

        # Sequence-level average logprob on response tokens
        new_logprob_seq = (token_logprobs * response_mask).sum(1) / lengths
        old_logprob_seq = (old_logprobs * response_mask).sum(1) / lengths
        ref_logprob_seq = (ref_logprobs * response_mask).sum(1) / lengths

        ratios = (new_logprob_seq - old_logprob_seq).exp()
        adv = advantages.detach()

        policy_loss_1 = -ratios * adv
        policy_loss_2 = -torch.clamp(
            ratios,
            1.0 - config.clip_range,
            1.0 + config.clip_range,
        ) * adv
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

        # Value prediction baseline for response region
        value_pred = (values[:, :-1] * response_mask).sum(1) / lengths
        value_loss = F.mse_loss(value_pred, returns)

        # Entropy bonus on response tokens
        entropy_tokens = -(torch.exp(log_probs) * log_probs).sum(-1)
        entropy = (entropy_tokens * response_mask.float()).sum(1) / lengths
        entropy_loss = (entropy - config.target_entropy).abs().mean()

        # KL vs reference policy (sequence-level)
        kl = (new_logprob_seq - ref_logprob_seq)
        kl_mean = kl.mean()
        kl_abs = kl.abs().mean()

        total_loss = (
            policy_loss
            + config.value_coef * value_loss
            + config.entropy_coef * entropy_loss
            + config.kl_coef * kl_abs
        )

    scaler.scale(total_loss / config.gradient_accumulation_steps).backward()

    stats = {
        "loss": total_loss.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy_loss.detach(),
        "kl": kl_mean.detach(),
        "kl_abs": kl_abs.detach(),
        "ratios": ratios.detach(),
    }
    return stats


# ============================================================
# 4. Main PPO training loop
# ============================================================

def main():
    # --------------------------------------------------------
    # Load tokenizer (consistent with reward_model_training)
    # --------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # --------------------------------------------------------
    # Load PPO prompts (from HH-RLHF with same filtering)
    # --------------------------------------------------------
    ppo_prompts = load_ppo_prompts(train_subset_size=50000)

    # --------------------------------------------------------
    # Load policy model, reference model, value head
    # --------------------------------------------------------
    print("Loading policy and reference models...")
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    policy_model.train()

    ref_policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    ref_policy.config.pad_token_id = tokenizer.pad_token_id
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    value_head = nn.Linear(policy_model.config.n_embd, 1).to(device)

    # --------------------------------------------------------
    # Load trained reward model
    # --------------------------------------------------------
    print(f"Loading reward model from {REWARD_MODEL_DIR} ...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_DIR,
        num_labels=1,
    ).to(device)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    # --------------------------------------------------------
    # PPO optimizer, scheduler, and scaler
    # --------------------------------------------------------
    config = PPOConfig()
    print("PPO config:", config)

    params = list(policy_model.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_opt_steps = math.ceil(
        config.total_steps * config.ppo_epochs / config.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_opt_steps),
        num_training_steps=total_opt_steps,
    )
    scaler = torch.amp.GradScaler(enabled=config.fp16)

    # Training metrics (for logging only)
    reward_history = []
    kl_history = []
    loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []
    kl_coef_history = []

    os.makedirs(PPO_CHECKPOINT_DIR, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(range(config.total_steps), desc="PPO Training", disable=tqdm_disable)
    accumulated_steps = 0

    for step in progress:
        # 1) Sample prompts and generate rollout
        batch_prompts = sample_prompts(ppo_prompts, config.rollout_batch_size)
        rollout = generate_rollout(
            batch_prompts,
            policy_model,
            value_head,
            ref_policy,
            tokenizer,
            config,
            device,
        )

        # 2) Get rewards from reward model
        rewards = score_with_reward_model(
            rollout["prompts"],
            rollout["responses"],
            tokenizer,
            reward_model,
            device,
        )

        # 3) Compute advantages and returns
        advantages, returns, value_means, lengths = compute_advantages(
            rewards=rewards,
            values=rollout["values"],
            response_mask=rollout["response_mask"],
            gamma=config.gamma,
            lam=config.lam,
        )

        # 4) PPO policy/value updates for a few epochs
        last_stats = None
        for _ in range(config.ppo_epochs):
            stats = ppo_update(
                policy_model,
                value_head,
                rollout,
                advantages,
                returns,
                config,
                scaler,
                device,
            )
            accumulated_steps += 1
            last_stats = stats

            if accumulated_steps % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        # 5) Record metrics for logging
        reward_history.append(rewards.mean().item())
        kl_history.append(last_stats["kl_abs"].item())
        loss_history.append(last_stats["loss"].item())
        policy_loss_history.append(last_stats["policy_loss"].item())
        value_loss_history.append(last_stats["value_loss"].item())
        entropy_history.append(last_stats["entropy"].item())
        kl_coef_history.append(config.kl_coef)

        # 6) Adapt KL coefficient
        current_kl = last_stats["kl_abs"].item()
        if config.adapt_kl:
            min_kl_coef, max_kl_coef = 1e-4, 5.0
            if current_kl > config.target_kl * 1.2:
                config.kl_coef = min(config.kl_coef * 1.2, max_kl_coef)
            elif current_kl < config.target_kl / 1.2:
                config.kl_coef = max(config.kl_coef / 1.2, min_kl_coef)

        # 7) Update tqdm bar text
        if step % config.log_every == 0 or step == config.total_steps - 1:
            progress.set_postfix({
                "rew": f"{reward_history[-1]:.3f}",
                "loss": f"{last_stats['loss'].item():.3f}",
                "kl": f"{last_stats['kl_abs'].item():.3f}",
                "ent": f"{last_stats['entropy'].item():.3f}",
                "klc": f"{config.kl_coef:.3f}",
            })

        # 8) Periodic checkpoint save (training only, no eval)
        if (step + 1) % config.save_every == 0 or step == config.total_steps - 1:
            ckpt_dir = os.path.join(PPO_CHECKPOINT_DIR, f"ppo_checkpoint_step_{step + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(value_head.state_dict(), os.path.join(ckpt_dir, "value_head.pt"))

    # --------------------------------------------------------
    # Save final PPO model
    # --------------------------------------------------------
    os.makedirs(PPO_OUTPUT_DIR, exist_ok=True)
    policy_model.save_pretrained(PPO_OUTPUT_DIR)
    tokenizer.save_pretrained(PPO_OUTPUT_DIR)
    torch.save(value_head.state_dict(), os.path.join(PPO_OUTPUT_DIR, "value_head.pt"))

    print(f"PPO training complete. Final model saved to {PPO_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

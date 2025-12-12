#!/usr/bin/env python
# coding: utf-8

"""
GRPO training script for HH-RLHF with GPT-2 + scalar reward model.

- Uses the same dataset filtering as data_preprocessing.py
- Uses the same reward model as reward_model_training.py
- Follows the GRPO-style group-relative advantage training procedure
  implemented in assignment3.py, but as a standalone script.

Important:
- This script only trains GRPO (no evaluation / win-rate / KL plots).
- It assumes you have already run:
    - data_preprocessing.py   (to create ./processed_data, for reward training)
    - reward_model_training.py (to create ./final_reward_model)
"""

import os
import math
import random
from dataclasses import dataclass

import torch
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
GRPO_OUTPUT_DIR = "./grpo_rlhf_model"       # final GRPO policy output
GRPO_CHECKPOINT_DIR = "./grpo_checkpoint"   # intermediate checkpoints
GRPO_BEST_DIR = "./grpo_best"               # best model by mean reward

TRAIN_SUBSET_SIZE = 50000    # same as in assignment3 / data_preprocessing
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. Dataset filtering & prompt building
#    (mirrors assignment3 + data_preprocessing.py)
# ============================================================

# Same constants as data_preprocessing.py
MIN_LENGTH = 10
MAX_LENGTH = 1024
CHAR_LIMIT = MAX_LENGTH * 6


def filter_edge_cases(example):
    """
    Filter out ties, short responses, length-mismatch, and too-long sequences.
    This is the same logic as in data_preprocessing.py / assignment3.
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
    Same as in assignment3:
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


def build_grpo_dataset(example):
    """
    For each cleaned HH-RLHF example, extract the prompt from the chosen side.
    """
    prompt, last_assistant = split_conversation(example["chosen"])
    return {
        "prompt": prompt,
        "chosen_tail": last_assistant,
    }


def load_grpo_prompts(train_subset_size: int = TRAIN_SUBSET_SIZE):
    """
    Load HH-RLHF, apply the same filtering as assignment3/data_preprocessing,
    and build a list of prompts for GRPO rollouts.
    """
    print(f"Loading dataset: {DATASET_NAME} ...")
    dataset = load_dataset(DATASET_NAME)

    train_ds = dataset["train"].select(range(train_subset_size))
    test_ds = dataset["test"]
    print(f"Original train size: {len(train_ds)}")
    print(f"Original test size:  {len(test_ds)}")

    print("Filtering edge cases (same as data_preprocessing.py)...")
    cleaned_train_ds = train_ds.filter(filter_edge_cases, num_proc=4)
    print(f"Filtered train size: {len(cleaned_train_ds)}")

    print("Building GRPO prompts from 'chosen' conversations...")
    grpo_base = cleaned_train_ds.map(build_grpo_dataset)
    prompts = grpo_base["prompt"]
    print(f"Number of GRPO prompts: {len(prompts)}")

    return prompts


# ============================================================
# 2. GRPO configuration (mirrors GRPOConfig in assignment3)
# ============================================================

@dataclass
class GRPOConfig:
    # Overall training
    total_steps: int = 1500
    rollout_batch_size: int = 2      # number of different prompts per step
    group_size: int = 3              # number of responses per prompt
    ppo_epochs: int = 1              # 1 update per rollout
    gradient_accumulation_steps: int = 1

    # Regularization
    kl_coef: float = 0.02
    target_kl: float = 0.3
    adapt_kl: bool = False           # can be turned on if desired

    # Entropy regularization (target-entropy style, as in revised PPO script)
    entropy_coef: float = 0.1
    target_entropy: float = 1.8

    # Optimizer
    learning_rate: float = 2e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Advantage / return
    gamma: float = 1.0
    lam: float = 0.95

    # Generation hyperparams
    max_prompt_length: int = 192
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0

    # Logging / checkpointing
    save_every: int = 100
    log_every: int = 1
    fp16: bool = torch.cuda.is_available()


# ============================================================
# 3. Helper utilities
# ============================================================

def sample_prompts(all_prompts, n: int):
    """Randomly sample n prompts from the preprocessed preference dataset."""
    idxs = random.sample(range(len(all_prompts)), n)
    return [all_prompts[i] for i in idxs]


def score_with_reward_model(prompts, responses, tokenizer, reward_model, device):
    """
    Concatenate prompt + response and pass through the scalar reward model.
    Same idea as in assignment3 PPO/GRPO.
    """
    # Simple concatenation: "[prompt][response]"
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
        scores = outputs.logits.squeeze(-1)  # [B]

    return scores


def generate_grpo_rollout(
    batch_prompts,
    policy_model,
    ref_policy,
    tokenizer,
    reward_model,
    config: GRPOConfig,
    device,
):
    """
    GRPO rollout:
    - For each prompt, generate `group_size` independent responses
      with the current policy.
    - Compute sequence-level log-probabilities under current and
      reference policies.
    - Score each (prompt, response) with the frozen reward model.
    - Compute group-relative advantages: A_i = r_i - mean_group.
    """
    group_size = config.group_size

    # 1) Expand prompts so each appears group_size times
    expanded_prompts = []
    group_ids = []
    for g, p in enumerate(batch_prompts):
        for _ in range(group_size):
            expanded_prompts.append(p)
            group_ids.append(g)
    group_ids = torch.tensor(group_ids, device=device)

    # 2) Tokenize expanded prompts
    enc = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_prompt_length,
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # 3) Generate responses with the current policy
    policy_model.eval()
    with torch.no_grad():
        gen = policy_model.generate(
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
            repetition_penalty=1.2,
        )
    sequences = gen.sequences                      # [B*K, L]
    seq_attention = (sequences != tokenizer.pad_token_id).long()
    policy_model.train()

    # 4) Token-level logprobs under current policy and reference
    with torch.no_grad():
        out_pi = policy_model(input_ids=sequences, attention_mask=seq_attention)
        logits_pi = out_pi.logits                               # [B*K, L, V]
        log_probs = F.log_softmax(logits_pi[:, :-1, :], dim=-1) # [B*K, L-1, V]
        token_logprobs = log_probs.gather(
            2, sequences[:, 1:].unsqueeze(-1)
        ).squeeze(-1)                                           # [B*K, L-1]

        out_ref = ref_policy(input_ids=sequences, attention_mask=seq_attention)
        logits_ref = out_ref.logits
        ref_log_probs = F.log_softmax(logits_ref[:, :-1, :], dim=-1)
        ref_token_logprobs = ref_log_probs.gather(
            2, sequences[:, 1:].unsqueeze(-1)
        ).squeeze(-1)                                           # [B*K, L-1]

    # 5) Identify response region (same scheme as in PPO)
    B, Lm1 = token_logprobs.shape
    response_mask = torch.zeros_like(token_logprobs, dtype=torch.bool)
    responses = []

    for i in range(B):
        # Prompt tokens in the original encoding (left-padded)
        nonpad_prompt = (input_ids[i] != tokenizer.pad_token_id).nonzero(as_tuple=False).squeeze(-1)
        if nonpad_prompt.numel() == 0:
            responses.append("")
            continue
        last_prompt_idx = nonpad_prompt[-1].item()

        # All non-pad tokens in the full sequence (prompt + response)
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

        # Decode response tokens for reward model
        resp_tokens = sequences[i, last_prompt_idx + 1 : last_token_idx + 1]
        responses.append(tokenizer.decode(resp_tokens, skip_special_tokens=True))

    lengths = response_mask.sum(1).clamp(min=1)

    # 6) Sequence-level logprobs on response tokens
    logprob_seq = (token_logprobs * response_mask).sum(1) / lengths
    ref_logprob_seq = (ref_token_logprobs * response_mask).sum(1) / lengths

    # 7) Rewards from reward model
    rewards = score_with_reward_model(expanded_prompts, responses, tokenizer, reward_model, device)
    rewards = rewards.to(device)

    # 8) Group-relative advantages: A_i = r_i - mean_group
    advantages = torch.zeros_like(rewards)
    num_groups = len(batch_prompts)
    for g in range(num_groups):
        mask = (group_ids == g)
        if mask.sum() == 0:
            continue
        group_rew = rewards[mask]
        baseline = group_rew.mean()
        advantages[mask] = group_rew - baseline

    rollout = {
        "sequences": sequences,
        "attention_mask": seq_attention,
        "response_mask": response_mask,
        "logprob_seq": logprob_seq,
        "ref_logprob_seq": ref_logprob_seq,
        "advantages": advantages,
        "rewards": rewards,
        "responses": responses,
    }
    return rollout


def grpo_update(policy_model, rollout, config: GRPOConfig, scaler, device):
    """
    Single GRPO update step:
    - Recompute log-probabilities under the current policy.
    - Policy loss: - E[adv * log π(a|s)]
    - KL regularization vs reference
    - Target-entropy regularization (to avoid entropy explosion).
    """
    sequences = rollout["sequences"].to(device)
    attn = rollout["attention_mask"].to(device)
    response_mask = rollout["response_mask"].to(device)
    adv = rollout["advantages"].to(device)
    ref_logprob_seq = rollout["ref_logprob_seq"].to(device)

    lengths = response_mask.sum(1).clamp(min=1)

    # Normalize advantages across the batch (helps stability)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    with torch.amp.autocast("cuda", enabled=config.fp16):
        outputs = policy_model(
            input_ids=sequences,
            attention_mask=attn,
        )
        logits = outputs.logits                             # [B*K, L, V]

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_logprobs = log_probs.gather(
            2, sequences[:, 1:].unsqueeze(-1)
        ).squeeze(-1)                                       # [B*K, L-1]

        logprob_seq = (token_logprobs * response_mask).sum(1) / lengths

        # GRPO policy gradient term
        policy_loss = -(adv.detach() * logprob_seq).mean()

        # Entropy on response tokens (sequence-level)
        entropy_tokens = -(torch.exp(log_probs) * log_probs).sum(-1)  # [B*K, L-1]
        entropy = (entropy_tokens * response_mask.float()).sum(1) / lengths  # [B*K]

        # Target-entropy regularization (matches revised PPO training)
        entropy_loss = (entropy - config.target_entropy).abs().mean()

        # KL vs reference (sequence-level)
        kl = (logprob_seq - ref_logprob_seq)
        kl_abs = kl.abs().mean()

        loss = (
            policy_loss
            + config.kl_coef * kl_abs
            + config.entropy_coef * entropy_loss
        )

    # Backward (with gradient accumulation support)
    if config.fp16:
        scaler.scale(loss / config.gradient_accumulation_steps).backward()
    else:
        (loss / config.gradient_accumulation_steps).backward()

    stats = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "entropy": entropy.mean().detach(),
        "kl_abs": kl_abs.detach(),
        # for a unified plotting API (even though GRPO has no value head)
        "value_loss": torch.tensor(0.0, device=loss.device),
    }
    return stats


# ============================================================
# 4. Main GRPO training loop
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
    # Load GRPO prompts (from HH-RLHF with same filtering)
    # --------------------------------------------------------
    grpo_prompts = load_grpo_prompts(train_subset_size=TRAIN_SUBSET_SIZE)

    # --------------------------------------------------------
    # Load GRPO policy and frozen reference model
    # --------------------------------------------------------
    print("Loading GRPO policy and reference models...")
    grpo_policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    grpo_policy.config.pad_token_id = tokenizer.pad_token_id
    grpo_policy.train()

    grpo_ref_policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    grpo_ref_policy.config.pad_token_id = tokenizer.pad_token_id
    grpo_ref_policy.eval()
    for p in grpo_ref_policy.parameters():
        p.requires_grad = False

    # --------------------------------------------------------
    # Load trained reward model (frozen)
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
    # Optimizer, scheduler, scaler
    # --------------------------------------------------------
    config = GRPOConfig()
    print("GRPO config:", config)

    params = [p for p in grpo_policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_opt_steps = math.ceil(
        config.total_steps / config.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_opt_steps),
        num_training_steps=total_opt_steps,
    )
    scaler = torch.amp.GradScaler(enabled=config.fp16)

    # --------------------------------------------------------
    # Training bookkeeping
    # --------------------------------------------------------
    reward_history = []
    kl_history = []
    loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []
    kl_coef_history = []

    os.makedirs(GRPO_CHECKPOINT_DIR, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    best_mean_reward = -float("inf")

    progress = tqdm(range(config.total_steps), desc="GRPO Training", disable=tqdm_disable)
    accumulated_steps = 0

    for step in progress:
        # 1) Sample prompts
        batch_prompts = sample_prompts(grpo_prompts, config.rollout_batch_size)

        # 2) Collect group rollout
        rollout = generate_grpo_rollout(
            batch_prompts,
            grpo_policy,
            grpo_ref_policy,
            tokenizer,
            reward_model,
            config,
            device,
        )

        # 3) Single GRPO update
        stats = grpo_update(grpo_policy, rollout, config, scaler, device)
        accumulated_steps += 1

        # 4) Optimizer / scheduler step
        if accumulated_steps % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # 5) Record metrics
        step_reward = rollout["rewards"].mean().item()
        reward_history.append(step_reward)
        kl_history.append(stats["kl_abs"].item())
        loss_history.append(stats["loss"].item())
        policy_loss_history.append(stats["policy_loss"].item())
        value_loss_history.append(stats["value_loss"].item())
        entropy_history.append(stats["entropy"].item())
        kl_coef_history.append(config.kl_coef)

        # 6) Track best checkpoint by reward
        if step_reward > best_mean_reward:
            best_mean_reward = step_reward
            if os.path.exists(GRPO_BEST_DIR):
                # Remove previous best to save space
                import shutil
                shutil.rmtree(GRPO_BEST_DIR)
            os.makedirs(GRPO_BEST_DIR, exist_ok=True)
            grpo_policy.save_pretrained(GRPO_BEST_DIR)
            tokenizer.save_pretrained(GRPO_BEST_DIR)
            # print(f"[GRPO] New best mean reward {best_mean_reward:.4f} at step {step + 1}")

        # 7) (Optional) KL adaptation — off by default
        if config.adapt_kl:
            min_kl_coef, max_kl_coef = 1e-4, 5.0
            current_kl = stats["kl_abs"].item()
            if current_kl > config.target_kl * 1.2:
                config.kl_coef = min(config.kl_coef * 1.2, max_kl_coef)
            elif current_kl < config.target_kl / 1.2:
                config.kl_coef = max(config.kl_coef / 1.2, min_kl_coef)

        # 8) Progress bar info
        if step % config.log_every == 0 or step == config.total_steps - 1:
            progress.set_postfix({
                "rew": f"{reward_history[-1]:.3f}",
                "loss": f"{loss_history[-1]:.3f}",
                "kl": f"{kl_history[-1]:.3f}",
                "ent": f"{entropy_history[-1]:.3f}",
                "klc": f"{config.kl_coef:.3f}",
            })

        # 9) Save periodic checkpoints
        if (step + 1) % config.save_every == 0 or step == config.total_steps - 1:
            ckpt_dir = os.path.join(GRPO_CHECKPOINT_DIR, f"grpo_checkpoint_step_{step + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            grpo_policy.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    # --------------------------------------------------------
    # Save final GRPO model
    # --------------------------------------------------------
    os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)
    grpo_policy.save_pretrained(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)
    print(f"GRPO training complete. Final model saved to {GRPO_OUTPUT_DIR}")
    print(f"Best-by-reward checkpoint saved to {GRPO_BEST_DIR}")


if __name__ == "__main__":
    main()

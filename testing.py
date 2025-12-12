#!/usr/bin/env python
# coding: utf-8

"""
Evaluate base / PPO / GRPO / DPO policies on Anthropic HH test prompts,
score with the reward model, and save results to an Excel file.

- Uses same filtering & prompt splitting as training.
- Shares the same 150 prompts across all models.
- Base model is the original GPT-2 (no RL training).
"""

import os
import random
from dataclasses import dataclass
from typing import List, Iterable, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm


# ============================================================
# 0. Which models to evaluate
# ============================================================

EVAL_PPO = True
EVAL_GRPO = True
EVAL_DPO = True

NUM_EVAL_PROMPTS = 150
EVAL_BATCH_SIZE = 4
OUTPUT_EXCEL_PATH = "rlhf_eval_results_with_base.xlsx"

# Paths — update if your directories differ
BASE_MODEL_NAME = "gpt2"                 # base LM + tokenizer
PPO_MODEL_PATH = "./ppo_rlhf_model"      # PPO policy checkpoint dir
GRPO_MODEL_PATH = "./grpo_best"          # GRPO best checkpoint dir
DPO_MODEL_PATH = "./dpo_rlhf_model"      # DPO policy checkpoint dir
REWARD_MODEL_PATH = "./final_reward_model"


# ============================================================
# 1. Utility helpers: batching, filtering, prompt splitting
# ============================================================

MIN_LENGTH = 10
MAX_LENGTH = 1024
CHAR_LIMIT = MAX_LENGTH * 6


def batched(iterable: Iterable, n: int) -> Iterable[List]:
    """Yield successive batches of size n from an iterable."""
    iterable = list(iterable)
    for i in range(0, len(iterable), n):
        yield iterable[i: i + n]


def filter_edge_cases(example: Dict) -> bool:
    """
    Same logic as training scripts:
    - remove ties
    - remove very short responses
    - remove extreme length mismatches
    - remove overly long sequences
    """
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


def split_conversation(full_text: str):
    """
    Split Anthropic HH-RLHF-style conversation into:
    - prompt: up to and including the last 'Assistant:' after the last 'Human:'.
    - last_assistant: trailing assistant text after that.
    Same convention as in your PPO / GRPO scripts.
    """
    text = full_text.strip()
    last_human_idx = text.rfind("Human:")
    if last_human_idx == -1:
        return text, ""

    assistant_idx = text.find("Assistant:", last_human_idx)
    if assistant_idx == -1:
        prompt = text[:last_human_idx]
        return prompt.strip(), text[last_human_idx:].strip()

    prompt_end = assistant_idx + len("Assistant:")
    prompt = text[:prompt_end]
    last_assistant = text[prompt_end:]
    return prompt.strip(), last_assistant.strip()


def build_eval_prompts(num_prompts: int, seed: int = 42) -> List[str]:
    """
    Build evaluation prompts from Anthropic HH test split,
    using the same filtering + prompt splitting as training.
    """
    print("Loading Anthropic HH-RLHF dataset (test split)...")
    dataset = load_dataset("anthropic/hh-rlhf")
    test_ds = dataset["test"]

    print("Filtering edge cases on test split...")
    cleaned_test_ds = test_ds.filter(filter_edge_cases, num_proc=4)
    print(f"Filtered test size: {len(cleaned_test_ds)}")

    # Build prompts from 'chosen' side
    print("Building evaluation prompts from 'chosen' conversations...")
    prompts: List[str] = []
    max_eval_samples = min(500, len(cleaned_test_ds))

    for ex in cleaned_test_ds.select(range(max_eval_samples)):
        prompt, _ = split_conversation(ex["chosen"])
        prompts.append(prompt)

    print(f"Total candidate eval prompts: {len(prompts)}")

    random.seed(seed)
    if len(prompts) <= num_prompts:
        selected = prompts
    else:
        selected = random.sample(prompts, num_prompts)

    print(f"Using {len(selected)} prompts for evaluation.")
    return selected


# ============================================================
# 2. Generation & reward scoring helpers
# ============================================================

@dataclass
class EvalConfig:
    max_prompt_length: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int = 0


# Per-model generation configs (roughly matching training settings)
BASE_EVAL_CONFIG = EvalConfig(
    max_prompt_length=256,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    top_k=0,
)

PPO_EVAL_CONFIG = EvalConfig(
    max_prompt_length=256,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    top_k=0,
)

GRPO_EVAL_CONFIG = EvalConfig(
    max_prompt_length=192,
    max_new_tokens=32,
    temperature=1.0,
    top_p=0.9,
    top_k=0,
)

DPO_EVAL_CONFIG = EvalConfig(
    max_prompt_length=256,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    top_k=0,
)


def generate_responses(
    model: AutoModelForCausalLM,
    prompts: List[str],
    tokenizer: AutoTokenizer,
    config: EvalConfig,
    device: torch.device,
) -> List[str]:
    """
    Generate responses from a policy model, then strip off the prompt and keep only
    the newly generated suffix.
    """
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_prompt_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else None,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        sequences = outputs.sequences  # [B, T_total]

    B = sequences.size(0)
    responses: List[str] = []

    for i in range(B):
        # Prompt tokens in original encoding (left-padded)
        nonpad_prompt = (encoded["input_ids"][i] != tokenizer.pad_token_id).nonzero(
            as_tuple=False
        ).squeeze(-1)
        if nonpad_prompt.numel() == 0:
            responses.append("")
            continue
        last_prompt_idx = nonpad_prompt[-1].item()

        # All non-pad tokens in the full sequence (prompt + response)
        nonpad_seq = (sequences[i] != tokenizer.pad_token_id).nonzero(
            as_tuple=False
        ).squeeze(-1)
        if nonpad_seq.numel() == 0:
            responses.append("")
            continue
        last_token_idx = nonpad_seq[-1].item()

        # Response is everything after prompt tokens
        start = last_prompt_idx + 1
        end = last_token_idx + 1
        if end <= start:
            responses.append("")
            continue

        resp_ids = sequences[i, start:end]
        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        responses.append(resp_text)

    return responses


def score_with_reward_model(
    prompts: List[str],
    responses: List[str],
    tokenizer: AutoTokenizer,
    reward_model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Compute reward scores for (prompt + response) using the trained reward model.
    Returns a 1D tensor of rewards.
    """
    reward_model.eval()
    all_scores = []

    for batch_prompts, batch_resps in zip(
        batched(prompts, batch_size), batched(responses, batch_size)
    ):
        full_texts = [p + r for p, r in zip(batch_prompts, batch_resps)]
        enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = reward_model(**enc)
            logits = outputs.logits.squeeze(-1)  # [B]
            scores = logits.detach().cpu()

        all_scores.append(scores)

    return torch.cat(all_scores, dim=0)  # [N]


# ============================================================
# 3. Main: load models, run evaluation, save Excel
# ============================================================

def main():
    if not (EVAL_PPO or EVAL_GRPO or EVAL_DPO):
        print("Note: PPO/GRPO/DPO toggles are all False, but base model will still be evaluated.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Shared tokenizer for policies & reward model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    # Reward model
    print("Loading reward model...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        num_labels=1,
    ).to(device)
    reward_model.eval()

    # Evaluation prompts
    eval_prompts = build_eval_prompts(NUM_EVAL_PROMPTS)
    N = len(eval_prompts)

    # Base model
    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.eval()

    # RL policies as requested
    models: Dict[str, AutoModelForCausalLM] = {}
    configs: Dict[str, EvalConfig] = {}

    # Base model is always evaluated
    models["base"] = base_model
    configs["base"] = BASE_EVAL_CONFIG

    if EVAL_PPO:
        print(f"Loading PPO model from {PPO_MODEL_PATH}...")
        ppo_model = AutoModelForCausalLM.from_pretrained(PPO_MODEL_PATH).to(device)
        ppo_model.config.pad_token_id = tokenizer.pad_token_id
        ppo_model.eval()
        models["ppo"] = ppo_model
        configs["ppo"] = PPO_EVAL_CONFIG

    if EVAL_GRPO:
        print(f"Loading GRPO model from {GRPO_MODEL_PATH}...")
        grpo_model = AutoModelForCausalLM.from_pretrained(GRPO_MODEL_PATH).to(device)
        grpo_model.config.pad_token_id = tokenizer.pad_token_id
        grpo_model.eval()
        models["grpo"] = grpo_model
        configs["grpo"] = GRPO_EVAL_CONFIG

    if EVAL_DPO:
        print(f"Loading DPO model from {DPO_MODEL_PATH}...")
        dpo_model = AutoModelForCausalLM.from_pretrained(DPO_MODEL_PATH).to(device)
        dpo_model.config.pad_token_id = tokenizer.pad_token_id
        dpo_model.eval()
        models["dpo"] = dpo_model
        configs["dpo"] = DPO_EVAL_CONFIG

    # Collect results
    results: Dict[str, List] = {}
    results["prompt"] = eval_prompts

    # Evaluate each model
    for name, model in models.items():
        cfg = configs[name]
        print(f"\nGenerating responses for model: {name}")

        all_responses: List[str] = []

        for batch_prompts in tqdm(
            list(batched(eval_prompts, EVAL_BATCH_SIZE)),
            desc=f"Generating ({name})",
        ):
            batch_resps = generate_responses(
                model=model,
                prompts=batch_prompts,
                tokenizer=tokenizer,
                config=cfg,
                device=device,
            )
            all_responses.extend(batch_resps)

        assert len(all_responses) == N, f"{name}: response length mismatch"

        print(f"Scoring rewards for model: {name}")
        rewards_tensor = score_with_reward_model(
            eval_prompts,
            all_responses,
            tokenizer=tokenizer,
            reward_model=reward_model,
            device=device,
            batch_size=EVAL_BATCH_SIZE,
        )
        all_rewards = rewards_tensor.tolist()

        results[f"{name}_response"] = all_responses
        results[f"{name}_reward"] = all_rewards

    # Save to Excel
    print(f"\nSaving results to {OUTPUT_EXCEL_PATH}...")
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()

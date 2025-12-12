# IEMS 490 HW3 – RLHF with GPT-2 (Reward Model, PPO, GRPO, DPO)

This repository implements a complete small-scale RLHF pipeline on top of `gpt2`:

1. **Process** the Anthropic `hh-rlhf` dataset into a local pairwise preference dataset.
2. **Train a reward model** on chosen vs rejected responses.
3. **Train three RLHF policies**:
   - PPO (`PPO_training.py`)
   - GRPO (`GRPO_training.py`)
   - DPO (`DPO_training.py`)
4. **Evaluate** base vs PPO/GRPO/DPO with:
   - The learned reward model
   - An external GPT-4 judge (win rates)
   - Reward vs KL analysis.

The code is designed to be reproducible on a single-GPU machine or GPU node in a cluster environment.

---

## 1. Repository Structure

Key files:

- `data_process.py` – loads Anthropic `hh-rlhf` from Hugging Face and writes processed preference data into `./processed_data/`.
- `reward_model_training.py` – trains a scalar reward model on (chosen, rejected) pairs and saves it as `./final_reward_model/`.
- `PPO_training.py` – trains an RLHF policy with PPO and saves it as `./ppo_rlhf_model/`.
- `GRPO_training.py` – trains an RLHF policy with GRPO and saves it as `./grpo_best/`.
- `DPO_training.py` – trains an RLHF policy with DPO and saves it as `./dpo_rlhf_model/`.
- `evaluate_models.py` – generates responses from base / PPO / GRPO / DPO, scores them with the reward model, and saves an Excel file `rlhf_eval_results_with_base.xlsx`.
- `analysis_notebook.ipynb` (or similar) – notebook that:
  - calls GPT-4 as a judge to compute win rates vs base,
  - plots reward distributions,
  - computes approximate KL divergence vs the reference policy.
- `Dockerfile` – container recipe for the full pipeline.
- `.dockerignore` – keeps large model/data folders out of the Docker build context.

Output directories (created during training):

- `./processed_data/` – tokenized and filtered dataset.
- `./final_reward_model/` – trained reward model.
- `./ppo_rlhf_model/` – PPO policy checkpoint.
- `./grpo_best/` – best GRPO policy checkpoint.
- `./dpo_rlhf_model/` – DPO policy checkpoint.
- `./rlhf_eval_results_with_base.xlsx` – evaluation results (prompts, responses, rewards).

Trained Models (HuggingFace)

- Reward model: https://huggingface.co/LSC2021/gpt2-hh-reward-hw3
- PPO policy: https://huggingface.co/LSC2021/gpt2-hh-ppo-rlhf-hw3
- DPO policy: https://huggingface.co/LSC2021/gpt2-hh-dpo-rlhf-hw3
- GRPO policy: https://huggingface.co/LSC2021/gpt2-hh-grpo-rlhf-hw3

---

## 2. Software Setup

You can run the project either:

- in a **conda/venv environment**, or  
- in the provided **Docker container**.

### 2.1. Python environment (conda/venv)

Requirements:

- Linux (Ubuntu 20.04/22.04 tested)  
- Python ≥ 3.9  
- CUDA-capable GPU with recent NVIDIA driver (for GPU training)

Create an environment (example with conda):

```bash
conda create -n llm python=3.10 -y
conda activate llm
````

Install PyTorch (CUDA version adjusted to your system; this example uses CUDA 12.1 wheels):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision 
```

Install project dependencies:

```bash
pip install \
    transformers==4.46.0 \
    datasets \
    accelerate \
    pandas \
    numpy \
    tqdm \
    matplotlib \
    scikit-learn \
    openpyxl \
    sentencepiece \
    einops \
    openai \
    jupyter
```

> Note: `openai` is only required for GPT-4 judgement in the analysis notebook. The core training (reward model + PPO/GRPO/DPO) works entirely offline once `hh-rlhf` is downloaded.

---

### 2.2. Docker setup

The `Dockerfile` defines a full GPU-ready environment.

Build the image (from the repo root):

```bash
docker build -t rlhf-hw3 .
```

If you are on a GPU machine with `nvidia-docker`:

```bash
docker run -it --rm -v $PWD:/workspace rlhf-hw3
```

Inside the container:

```bash
cd /workspace
python data_process.py
python reward_model_training.py
python PPO_training.py
python GRPO_training.py
python DPO_training.py
python testing.py
```

The `.dockerignore` file should exclude large model checkpoints and data from the build context so the image builds quickly.

---

## 3. Compute Requirements

This is a small-scale RLHF exercise, but still GPU-heavy. Below are the practical requirements used in development.

### 3.1. GPU

* **Recommended:** 1 × GPU with **16 GB** VRAM

  * e.g., NVIDIA Quadro RTX 5000 / RTX 3080 / A4000 class
* **Minimum (with hyperparameter tweaks):** 1 × GPU with **12 GB** VRAM

  * You may need to reduce:

    * `rollout_batch_size`
    * `max_new_tokens`
    * `batch_size` and `grad_accum_steps` in DPO/reward model training.

All three RLHF methods (PPO, GRPO, DPO) are implemented for **single-GPU** usage. Multi-GPU is not required.

### 3.2. CPU & RAM

* **CPU:** 4–8 cores are sufficient
* **RAM:** 16 GB recommended

  * The heaviest use is tokenization and data loading from `datasets`.

### 3.3. Disk space

Approximate disk usage:

* Hugging Face caches (models + `hh-rlhf`): ~5–10 GB
* Reward model + RLHF policy checkpoints: ~5–10 GB
* Logs, Excel outputs, and intermediates: a few GB

It is safe to assume **≥ 30 GB free disk** is comfortable for running the full pipeline, especially if Docker images are also built on the same machine.

---

## 4. End-to-End Pipeline

### 4.1. Step 1 – Data preprocessing

This step downloads Anthropic `hh-rlhf` via `datasets`, filters it, and writes a processed dataset to `./processed_data/`.

```bash
python data_process.py
```

The script:

* removes low-quality or extremely long examples,
* keeps both `chosen` and `rejected` responses,
* prepares them for the reward model and RLHF policies.

---

### 4.2. Step 2 – Train the reward model

Train a scalar reward model that scores full conversations (`prompt + response`):

```bash
python reward_model_training.py
```

Key aspects:

* Base: `gpt2` with a single linear head for a scalar reward.
* Dataset: uses the processed `./processed_data/` pairs.
* Objective: maximize `log σ(r_chosen − r_rejected)`.
* Output: final model saved to `./final_reward_model/`.

This reward model is used by both PPO and GRPO during training and by the evaluation script for scoring responses.

---

### 4.3. Step 3 – Train PPO policy

```bash
python PPO_training.py
```

Configuration (simplified):

* `total_steps = 1500`
* `rollout_batch_size = 8`
* `ppo_epochs = 2`
* `kl_coef = 0.05`, `target_kl = 0.3`, `adapt_kl = True`
* `entropy_coef = 0.1`, `target_entropy = 1.8`
* `max_prompt_length = 256`, `max_new_tokens = 64`

The script:

* samples rollouts from the current policy,
* uses the reward model to score responses,
* applies PPO updates with KL and entropy regularization,
* saves the final policy to `./ppo_rlhf_model/`.

---

### 4.4. Step 4 – Train GRPO policy

```bash
python GRPO_training.py
```

Configuration (simplified):

* `total_steps = 1500`
* `rollout_batch_size = 2`
* `group_size = 3` (multiple responses per prompt)
* `kl_coef = 0.02`, `target_kl = 0.3` (no adaptive KL by default)
* `entropy_coef = 0.1`, `target_entropy = 1.8`
* `max_prompt_length = 192`, `max_new_tokens = 32`

The script:

* samples groups of responses per prompt,
* computes group-relative rewards,
* pushes the policy towards the highest-reward responses,
* saves its best checkpoint to `./grpo_best/`.

---

### 4.5. Step 5 – Train DPO policy

```bash
python DPO_training.py
```

Configuration (simplified):

```python
num_epochs = 1
batch_size = 4
grad_accum_steps = 8
learning_rate = 5e-6
beta = 0.1       # DPO temperature
max_seq_len = 512
```

The script:

* uses the processed preference pairs `(prompt + chosen, prompt + rejected)`,
* optimizes the DPO objective against a frozen reference `gpt2`,
* does not require rollouts or a value function,
* saves the resulting model to `./dpo_rlhf_model/`.

---

### 4.6. Step 6 – Evaluation: responses + rewards

```bash
python evaluate_models.py
```

This script:

* Samples 150 test prompts from the Anthropic `hh-rlhf` test split using the same filtering logic as training.
* Generates responses from:

  * base `gpt2`,
  * PPO policy (`./ppo_rlhf_model/`),
  * GRPO policy (`./grpo_best/`),
  * DPO policy (`./dpo_rlhf_model/`).
* Scores each `(prompt, response)` with the reward model (`./final_reward_model/`).
* Saves everything to `rlhf_eval_results_with_base.xlsx` with columns like:

  * `prompt`
  * `base_response`, `base_reward`
  * `ppo_response`, `ppo_reward`
  * `grpo_response`, `grpo_reward`
  * `dpo_response`, `dpo_reward`

This file is the basis for the quantitative and qualitative analyses.

---

## 5. Analysis & GPT-4 Judging (Optional)

The notebook (`analysis_notebook.ipynb`) performs three types of analysis:

1. **GPT-4 win rates vs base**

   * Compares base vs PPO, base vs GRPO, base vs DPO
   * Uses a strict GPT-4 system prompt and only accepts “A” or “B” as a decision.
   * Caches judgements in `gpt4_judgements.json` to avoid re-using tokens.

   To use GPT-4 via the OpenAI API, set:

   ```bash
   export OPENAI_API_KEY=YOUR_KEY_HERE
   ```

2. **Reward distributions**

   * Plots boxplots of `*_reward` columns to compare alignment according to the learned reward model.

3. **Approximate KL divergence**

   * For each policy (PPO, GRPO, DPO), computes the average log-prob difference on response tokens relative to the base `gpt2` policy.
   * Produces a scatter plot of reward vs KL to visualize trade-offs between alignment and proximity to the reference policy.

These results are then used in the written analysis for Task B (alignment quality vs computational efficiency).

---

## 6. Reproducibility Notes

* Random seeds can be set inside each script (e.g., `torch.manual_seed`, `random.seed`, `numpy.random.seed`) to make runs more deterministic.
* Checkpoints are saved deterministically based on the provided configs; re-running with the same seed and hardware should produce very similar statistics.
* Large intermediate checkpoints (`*_checkpoint*` folders) can be deleted after training once the final directories

  * `final_reward_model/`,
  * `ppo_rlhf_model/`,
  * `grpo_best/`,
  * `dpo_rlhf_model/`
    are confirmed.


## Appendix: Data Processing and Training Pipelines

This section documents how the HH-RLHF data is preprocessed and how each model
(reward model, PPO, GRPO, DPO) is trained in this repository.

---

### 1. Data preprocessing (`data_process.py`)

We start from the **Anthropic HH-RLHF** dataset (`anthropic/hh-rlhf`) and build a
clean, tokenized subset that is reused across scripts. :contentReference[oaicite:0]{index=0}  

**Steps:**

1. **Load dataset & tokenizer**
   - Base model: `gpt2`.
   - Load the full HH-RLHF train/test splits with `datasets.load_dataset`.
   - Configure the tokenizer with `eos_token` as `pad_token`, left padding and left
     truncation.

2. **Subsample & clean**
   - Take a subset of **50,000** training examples.
   - Apply `filter_edge_cases` to remove:
     - ties (`chosen == rejected`);
     - very short responses (`len < 10`);
     - strong length mismatches (`len(chosen) / len(rejected) ∉ [0.5, 2.0]`);
     - extremely long conversations (character length beyond a fixed limit).

3. **Tokenization**
   - For each pair, tokenize `chosen` and `rejected` separately with a max length
     of 512 tokens.
   - Store:
     - `input_ids_chosen`, `attention_mask_chosen`
     - `input_ids_rejected`, `attention_mask_rejected`.

4. **Train / validation / test splits**
   - Shuffle the processed training set.
   - Reserve **10%** of the training data as a validation split.
   - Keep the filtered test split from HH-RLHF as the final test set.

5. **Save to disk**
   - Save a `DatasetDict` with `train / validation / test` splits to
     `./processed_data`, which is then consumed by the reward-model training
     script.

---

### 2. Reward model training (`reward_model_training.py`)

The reward model is a scalar classifier on top of GPT-2 that scores a full
conversation (prompt + response). :contentReference[oaicite:1]{index=1}  

**Architecture and objective**

- Base model: `AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)`.
- Input: tokenized `chosen` and `rejected` conversations from `./processed_data`.
- Output: a single scalar reward for each sequence.
- Loss: for each pair, we compute  
  \[
  L = -\log\sigma(r_\text{chosen} - r_\text{rejected}),
  \]
  encouraging the model to assign higher reward to the human-preferred
  `chosen` response.

**Training setup**

- Optimized with Hugging Face `Trainer` using a custom `RewardTrainer` that
  implements the pairwise loss and evaluation logic.
- `RewardDataCollatorWithPadding` handles padding `chosen` and `rejected`
  sequences separately and assembling them into the batch.
- Key hyperparameters:
  - 3 epochs, learning rate `2e-5`,
  - batch size 8 with gradient accumulation of 2,
  - 10% warmup, evaluation and checkpointing every 500 steps.
- Metric: **accuracy** = fraction of pairs where `r_chosen > r_rejected` on the
  validation split.
- Final model and tokenizer are saved to `./final_reward_model`, which is reused
  by PPO and GRPO.

---

### 3. PPO policy training (`PPO_training.py`)

The PPO script trains a **GPT-2 policy** to maximize the learned reward while
staying close to a frozen reference policy. :contentReference[oaicite:2]{index=2}  

**Data and prompts**

- Reload the raw **HH-RLHF** dataset and apply the same `filter_edge_cases`
  function as in preprocessing.
- For each cleaned example, we use the **`chosen` conversation** and split it
  into:
  - `prompt`: everything up to and including the last `"Assistant:"` following
    the last `"Human:"`;
  - `chosen_tail`: the final assistant reply.
- Only the `prompt` is used as the start state for rollouts; new responses are
  generated by the policy.

**Models**

- Trainable **policy model**: `gpt2` causal LM.
- Frozen **reference model**: another copy of GPT-2.
- Separate **value head**: a linear layer on the last hidden state (one scalar
  per token).
- Frozen **reward model**: `./final_reward_model` from the previous step.

**Rollout + reward**

For each PPO step:

1. Sample a batch of prompts from the cleaned dataset.
2. Generate responses with the current policy using sampling
   (`max_new_tokens`, `temperature`, `top_p`).
3. Identify the **response region** in the full token sequence (prompt + response)
   and compute:
   - token-level log-probs under the current policy and reference policy;
   - token-level value predictions from the value head.
4. Concatenate prompt + response, feed through the frozen reward model, and
   obtain a scalar reward for each sequence.

**Advantages and PPO update**

- Compute a simple baseline: mean value prediction over the response tokens;
  **advantage = reward − baseline**; returns = advantage + baseline.
- Normalize advantages across the batch.
- PPO loss:
  - clipped policy objective on **sequence-level** log-prob for response tokens;
  - value loss (MSE between value baseline and returns);
  - entropy regularization toward a target entropy;
  - KL penalty vs reference policy, with an adaptive coefficient that increases
    if KL drifts too high and decreases if KL is too small.
- Training runs for 1500 steps with small batch sizes, periodic checkpoint
  saving, and the final PPO policy is written to `./ppo_rlhf_model`.

---

### 4. GRPO policy training (`GRPO_training.py`)

The GRPO script trains another GPT-2 policy using **group-relative preference
optimization**, again with the same HH-RLHF filtering and the same reward
model. :contentReference[oaicite:3]{index=3}  

**Data and prompts**

- Load HH-RLHF, filter with `filter_edge_cases`, and keep the first 50,000
  training examples.
- For each cleaned example, extract a `prompt` from the **`chosen` side** by
  splitting at the last `"Human:"` / `"Assistant:"` markers (same logic as PPO).
- Store the list of prompts; these are sampled during GRPO rollouts.

**Models**

- Trainable **GRPO policy**: GPT-2 causal LM.
- Frozen **reference policy**: another GPT-2 copy.
- Frozen **reward model**: `./final_reward_model`.

**Group-based rollout**

For each training step:

1. Sample a small batch of prompts.
2. For each prompt, generate **`group_size` independent responses** with the
   current policy (e.g., 3 responses per prompt).
3. Compute response-region log-probs under both the current policy and the
   reference model.
4. Concatenate prompt + response, run through the reward model, and get scalar
   rewards for each (prompt, response) pair.

**Group-relative advantages and update**

- For each prompt group, compute
  - baseline = mean reward across that group,
  - advantage for each response = `r_i − baseline`.
- Normalize advantages across the batch.
- GRPO loss:
  - policy gradient term `−E[adv * log π(a|s)]` over the response region;
  - KL penalty vs reference (sequence-level);
  - target-entropy regularization, encouraging the entropy of the response
    distribution to stay near a desired level.
- Keep track of the **best-by-reward checkpoint** and save it to `./grpo_best`,
  while the final policy is saved to `./grpo_rlhf_model`.

---

### 5. DPO policy training (`DPO_training.py`)

The DPO script implements **Direct Preference Optimization** on HH-RLHF pairs,
using a frozen reference model and no rollout generation. :contentReference[oaicite:4]{index=4}  

**Data**

- Attempt to load a `train` split with `"chosen"`/`"rejected"` columns from
  `./processed_data`.  
  If not present, fall back to the raw HH-RLHF dataset and apply the same
  `filter_edge_cases` as in preprocessing.
- Wrap the dataset with `DPOPairDataset`, which:
  - splits each `chosen` and `rejected` conversation into (`prompt`,
    last_assistant_reply) using the same `split_conversation` logic as PPO/GRPO;
  - stores triples `(prompt, chosen_tail, rejected_tail)`.

**Models**

- Trainable **policy model**: GPT-2 causal LM.
- Frozen **reference model**: another GPT-2.
- Both share the same tokenizer configuration as the other scripts.

**Objective**

- For each batch, construct full sequences:
  - `prompt + chosen_tail`
  - `prompt + rejected_tail`.
- Compute **sequence log-probabilities** under both the policy and reference
  models by summing token log-probs over non-pad tokens.
- Let `Δ_logπ = log πθ(x_chosen) − log πθ(x_rejected)` and
  `Δ_logπ_ref = log π_ref(x_chosen) − log π_ref(x_rejected)`.  
  The DPO logit is  
  \[
  β\left[(\log πθ - \log π_\text{ref})_\text{chosen}
        - (\log πθ - \log π_\text{ref})_\text{rejected}\right],
  \]
  and the loss is `−log σ(logit)` averaged over the batch.
- This directly encourages the policy to increase relative probability mass on
  `chosen` responses compared to `rejected` ones, while implicitly keeping it
  close to the reference.

**Training loop**

- Build a dataloader over `DPOPairDataset` with a custom collator that tokenizes
  the concatenated sequences.
- Use AdamW with a small learning rate and linear warmup/decay schedule.
- Track the DPO loss and an approximate KL diagnostic between policy and
  reference.
- Save the final DPO policy and tokenizer to `./dpo_rlhf_model`.

---

Together, these components form a complete RLHF pipeline:

1. **Preprocess HH-RLHF** into a clean, tokenized preference dataset.
2. **Train a scalar reward model** that scores conversations.
3. **Train three RLHF policies (PPO, GRPO, DPO)** that optimize this reward
   under different algorithmic assumptions and KL–reward trade-offs.

# data_preprocessing.py

import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "gpt2"
DATASET_NAME = "anthropic/hh-rlhf"
OUTPUT_DIR = "./processed_data"  # Where to save the processed dataset
SEED = 42
TRAIN_SUBSET_SIZE = 50000  # As per your notebook
TEST_SPLIT_SIZE = 0.1      # 10% for validation

# Filter constants
MIN_LENGTH = 10
MAX_LENGTH = 1024
CHAR_LIMIT = MAX_LENGTH * 6

def get_tokenizer(model_name):
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set truncation/padding sides as per notebook
    tokenizer.truncation_side = 'left'
    tokenizer.padding_side = 'left'
    return tokenizer

def filter_edge_cases(example):
    """
    Filter out ties, short responses, and length outliers.
    """
    chosen = example["chosen"]
    rejected = example["rejected"]
    
    # 1. Filter out identical responses (Ties)
    if chosen == rejected:
        return False
        
    # 2. Filter short responses (Garbage / Refusal)
    if len(chosen) < MIN_LENGTH or len(rejected) < MIN_LENGTH:
        return False
        
    # 3. Filter length outliers (Length Mismatch)
    if len(rejected) == 0: 
        return False
    ratio = len(chosen) / len(rejected)
    if ratio < 0.5 or ratio > 2.0:
        return False
        
    # 4. Filter long sequences
    if len(chosen) >= CHAR_LIMIT or len(rejected) >= CHAR_LIMIT:
        return False
        
    return True

def preprocess_function(examples, tokenizer):
    """
    Tokenize the data for GPT-2.
    """
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, truncation=True, max_length=512, add_special_tokens=True)
        tokenized_rejected = tokenizer(rejected, truncation=True, max_length=512, add_special_tokens=True)
        
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        
    return new_examples

def main():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = get_tokenizer(MODEL_NAME)
    
    print(f"Loading dataset: {DATASET_NAME}...")
    # Load raw dataset
    dataset = load_dataset(DATASET_NAME)
    
    # Select subset for training as done in the notebook
    train_ds = dataset["train"].select(range(TRAIN_SUBSET_SIZE))
    test_ds = dataset["test"]
    
    print(f"Original Train size: {len(train_ds)}")
    print(f"Original Test size: {len(test_ds)}")

    # 1. Filtering
    print("Filtering edge cases...")
    cleaned_train_ds = train_ds.filter(filter_edge_cases, num_proc=4)
    cleaned_test_ds = test_ds.filter(filter_edge_cases, num_proc=4)
    print(f"Filtered Train size: {len(cleaned_train_ds)}")

    # 2. Tokenization
    print("Tokenizing dataset...")
    # Using a lambda to pass the tokenizer into the map function
    processed_train = cleaned_train_ds.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True, 
        num_proc=4,
        remove_columns=cleaned_train_ds.column_names
    )
    
    processed_test = cleaned_test_ds.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True, 
        num_proc=4,
        remove_columns=cleaned_test_ds.column_names
    )

    # 3. Create Splits (Train / Validation / Test)
    print("Creating splits...")
    processed_train = processed_train.shuffle(seed=SEED)
    
    # Split 10% of training data for validation
    split_dataset = processed_train.train_test_split(test_size=TEST_SPLIT_SIZE)
    train_set = split_dataset["train"]
    eval_set = split_dataset["test"]
    
    final_dataset = DatasetDict({
        "train": train_set,
        "validation": eval_set,
        "test": processed_test
    })

    print("Final Dataset Structure:")
    print(final_dataset)

    # 4. Save to disk
    # This is crucial so your next scripts can simply load this pre-processed data
    print(f"Saving processed dataset to {OUTPUT_DIR}...")
    final_dataset.save_to_disk(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
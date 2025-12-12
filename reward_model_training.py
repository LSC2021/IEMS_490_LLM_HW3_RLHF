# reward_model_training.py

import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase
)
from transformers.utils import PaddingStrategy
from datasets import load_from_disk

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "gpt2"
DATA_DIR = "./processed_data"          # Must match the output of step 1
OUTPUT_DIR = "./reward_model_gpt2"     # Checkpoints go here
FINAL_MODEL_DIR = "./final_reward_model" # Final saved model
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
EVAL_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 100

# ==========================================
# Custom Trainer Class
# ==========================================
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Forward Pass for "Chosen"
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"]
        ).logits
        
        # 2. Forward Pass for "Rejected"
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"]
        ).logits
        
        # 3. Calculate Loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected
            }
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Override to enable evaluation during training without errors
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            rewards_chosen = model(
                input_ids=inputs['input_ids_chosen'],
                attention_mask=inputs['attention_mask_chosen']
            ).logits
            rewards_rejected = model(
                input_ids=inputs['input_ids_rejected'],
                attention_mask=inputs['attention_mask_rejected']
            ).logits
            loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        
        if prediction_loss_only:
            return (loss, None, None)
            
        # prediction_step must return (loss, logits, labels)
        # We concatenate rewards to pass them to compute_metrics
        logits = torch.cat((rewards_chosen, rewards_rejected), dim=1)
        labels = torch.zeros(len(inputs), device=logits.device) # Dummy labels
        return (loss, logits, labels)

# ==========================================
# Metrics & Collator
# ==========================================
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    rewards_chosen = predictions[:, 0]
    rewards_rejected = predictions[:, 1]
    # Accuracy: How often is the chosen reward higher than rejected?
    accuracy = (rewards_chosen > rewards_rejected).mean()
    return {"accuracy": accuracy}

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Split inputs into chosen and rejected lists
        chosen_features = []
        rejected_features = []
        
        for feature in features:
            chosen_features.append({
                "input_ids": feature["input_ids_chosen"],
                "attention_mask": feature["attention_mask_chosen"],
            })
            rejected_features.append({
                "input_ids": feature["input_ids_rejected"],
                "attention_mask": feature["attention_mask_rejected"],
            })

        # Pad them separately
        batch_chosen = self.tokenizer.pad(
            chosen_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        batch_rejected = self.tokenizer.pad(
            rejected_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Re-assemble into the batch dictionary expected by RewardTrainer
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        
        return batch

# ==========================================
# Main Training Function
# ==========================================
def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the Pre-processed Data from Step 1
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Processed data not found at {DATA_DIR}. Run data_preprocessing.py first.")
    
    print(f"Loading dataset from {DATA_DIR}...")
    dataset = load_from_disk(DATA_DIR)
    
    # Initialize Model
    # num_labels=1 produces a single scalar (reward) per sequence
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_total_limit=2,
        metric_for_best_model="accuracy",
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        remove_unused_columns=False,  # CRITICAL for custom column names
        report_to="none",
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        fp16=torch.cuda.is_available(),
    )

    # Initialize Collator and Trainer
    data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting Reward Model Training...")
    trainer.train()

    print(f"Saving final model to {FINAL_MODEL_DIR}...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print("Training Complete!")

if __name__ == "__main__":
    main()
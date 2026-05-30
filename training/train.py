"""
Fine-tunes microsoft/Phi-3.5-mini-instruct on the generated dataset using LoRA.
Optimised for Apple Silicon (M3) using MPS device.

Steps:
  1. Load dataset from training/dataset.json
  2. Load Phi-3.5-mini base model
  3. Apply LoRA adapters (trains only ~1% of parameters)
  4. Fine-tune for a few epochs
  5. Save the merged model to training/finetuned_model/

Run:
  python training/train.py
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

DATASET_FILE    = os.path.join(os.path.dirname(__file__), "dataset.json")
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "finetuned_model")
BASE_MODEL      = "microsoft/Phi-3.5-mini-instruct"

# LoRA settings — small ranks keep memory usage low on 16GB
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

# Training settings
MAX_LENGTH      = 1024
EPOCHS          = 3
BATCH_SIZE      = 1     # Keep at 1 for M3 16GB
LEARNING_RATE   = 2e-4


def load_dataset() -> Dataset:
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_FILE}. "
            "Run python training/create_dataset.py first."
        )
    with open(DATASET_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training examples.")
    return Dataset.from_list(data)


def format_prompt(example: dict) -> str:
    """Format each example into the Phi-3.5 chat template."""
    return (
        f"<|user|>\n{example['instruction']}\n\n"
        f"Context:\n{example['context']}\n<|end|>\n"
        f"<|assistant|>\n{example['response']}<|end|>"
    )


def tokenize(example: dict, tokenizer) -> dict:
    prompt = format_prompt(example)
    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    tokens = {k: v.squeeze(0) for k, v in tokens.items()}
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens


def train():
    # Detect device via cross-platform router
    from training.model import get_device as _get_device
    device = str(_get_device())
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"\nLoading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device,
    )

    # Apply LoRA — only trains a small fraction of parameters
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize dataset
    raw_dataset = load_dataset()
    tokenized = raw_dataset.map(
        lambda ex: tokenize(ex, tokenizer),
        remove_columns=raw_dataset.column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        fp16=False,         # MPS doesn't support fp16 training
        bf16=False,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",   # No wandb/tensorboard
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    # Save the final model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! Model saved.")


if __name__ == "__main__":
    train()

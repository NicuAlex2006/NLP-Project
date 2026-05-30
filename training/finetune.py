"""
Fine-tune a SMALL PRETRAINED seq2seq model on the same summarization data the
from-scratch model used. Unlike train_scratch.py (which trains a transformer from
random init), this starts from a model that already knows English, so it produces
genuinely readable summaries with far less compute.

This one script fine-tunes EITHER of the two fine-tuned brains, selected by --model:
    distilbart : sshleifer/distilbart-cnn-12-6   (encoder-decoder, ~306M)
    flan-t5    : google/flan-t5-small            (instruction-tuned T5, ~80M)

Examples
--------
# Fine-tune distilbart (run on the RunPod GPU; ~20-40 min on a subset)
python training/finetune.py --model distilbart --max-per-source 15000 --epochs 1

# Fine-tune flan-t5 (smaller, faster)
python training/finetune.py --model flan-t5 --max-per-source 15000 --epochs 1

Output goes to training/finetuned_<name>/ as a standard HuggingFace folder
(config + weights + tokenizer), which agents/finetuned_agent.py loads directly.
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from training.data_utils import (
    load_cnn_dailymail,
    load_xsum,
    load_billsum,
    load_samsum,
    load_custom_dataset,
)
from training.model import get_device

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Registry of the two fine-tunable models.
# `prefix` is prepended to every input (T5 family expects a task prefix; BART does not).
MODEL_REGISTRY = {
    "distilbart": {"hf_name": "sshleifer/distilbart-cnn-12-6", "prefix": ""},
    "flan-t5":    {"hf_name": "google/flan-t5-small",          "prefix": "summarize: "},
}


def build_combined(split_examples_per_source: int, val_examples_per_source: int):
    """Reuse data_utils loaders to assemble combined (article, summary) pairs."""
    train_articles, train_summaries = [], []
    val_articles, val_summaries = [], []

    # CNN/DailyMail — news highlights
    ta, ts = load_cnn_dailymail("train", split_examples_per_source)
    va, vs = load_cnn_dailymail("validation", val_examples_per_source)
    train_articles += ta; train_summaries += ts
    val_articles += va; val_summaries += vs

    # XSum — single-sentence BBC summaries
    ta, ts = load_xsum("train", split_examples_per_source)
    va, vs = load_xsum("validation", val_examples_per_source)
    train_articles += ta; train_summaries += ts
    val_articles += va; val_summaries += vs

    # BillSum — formal/legal (small dataset, take a slice)
    ta, ts = load_billsum("train", min(split_examples_per_source, 18000))
    va, vs = load_billsum("ca_test", val_examples_per_source)
    train_articles += ta; train_summaries += ts
    val_articles += va; val_summaries += vs

    # SAMSum — dialogue summarization
    try:
        ta, ts = load_samsum("train", min(split_examples_per_source, 14000))
        va, vs = load_samsum("validation", val_examples_per_source)
        train_articles += ta; train_summaries += ts
        val_articles += va; val_summaries += vs
    except Exception as e:
        print(f"  (skipping SAMSum: {e})")

    # Your custom Tavily dataset
    ta, ts, _tq = load_custom_dataset()
    train_articles += ta; train_summaries += ts

    print(f"\n  Combined total: {len(train_articles)} train, {len(val_articles)} val")
    return (train_articles, train_summaries), (val_articles, val_summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="distilbart")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to training/finetuned_<model>/")
    parser.add_argument("--max-per-source", type=int, default=15000,
                        help="Max TRAIN examples taken from each dataset source")
    parser.add_argument("--val-per-source", type=int, default=500)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-target-len", type=int, default=128)
    args = parser.parse_args()

    spec = MODEL_REGISTRY[args.model]
    hf_name = spec["hf_name"]
    prefix = spec["prefix"]
    out_dir = args.output_dir or os.path.join(PROJECT_ROOT, "training", f"finetuned_{args.model}")

    print(f"=== Fine-tuning {args.model} ({hf_name}) ===")
    print(f"Output dir: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)

    # --- Build data ---
    (tr_a, tr_s), (va_a, va_s) = build_combined(args.max_per_source, args.val_per_source)
    train_ds = Dataset.from_dict({"article": tr_a, "summary": tr_s})
    val_ds = Dataset.from_dict({"article": va_a, "summary": va_s})

    def preprocess(batch):
        inputs = [prefix + (a or "") for a in batch["article"]]
        model_inputs = tokenizer(
            inputs, max_length=args.max_input_len, truncation=True
        )
        labels = tokenizer(
            text_target=[s or "" for s in batch["summary"]],
            max_length=args.max_target_len, truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\nTokenizing...")
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    device = get_device()
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(out_dir, "_checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        predict_with_generate=True,
        fp16=(device.type == "cuda"),
        bf16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("\nTraining...")
    trainer.train()

    # --- Save the final model in the flat folder the agent loads ---
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n✓ Saved fine-tuned model to: {out_dir}")
    print("  Load it via agents/finetuned_agent.py")


if __name__ == "__main__":
    main()

"""LoRA fine-tune a small causal LM on the cleaned instruction data.

Model: distilgpt2 (82M) -- small enough to smoke-test on CPU and to train
for real on a free Colab T4. LoRA freezes the base model and trains only
low-rank adapters, so this is a realistic small-scale version of the
production LLM fine-tuning workflow, not a toy re-implementation.

Run (Colab T4 recommended):
    python -m src.train_lora --data data/processed/train.jsonl \
        --output-dir outputs/lora-distilgpt2 --max-examples 2000 --epochs 2

Smoke test (CPU, proves the code path end to end, ~20 examples):
    python -m src.train_lora --data data/processed/train.jsonl \
        --output-dir /tmp/lora-smoke --max-examples 20 --epochs 1 --max-steps 5

Training was NOT run to completion on the CPU-only VM used to build this
repo. Do not claim trained-model results until this script has been run
and src/evaluate.py has been run on its output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: str, max_examples: int) -> list[str]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line)["text"])
            if len(rows) >= max_examples:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/train.jsonl")
    ap.add_argument("--output-dir", default="outputs/lora-distilgpt2")
    ap.add_argument("--base-model", default="distilgpt2")
    ap.add_argument("--max-examples", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    texts = load_rows(args.data, args.max_examples)
    print(f"Loaded {len(texts)} training texts")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, logging_steps=10, save_strategy="epoch",
        report_to=[], bf16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds,
                      data_collator=collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()

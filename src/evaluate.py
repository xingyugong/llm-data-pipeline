"""Evaluate base vs LoRA model on the held-out test split.

Reports, for the same held-out prompts:
  * mean token-level loss / perplexity of the gold response, and
  * greedy generations for a small fixed prompt set, saved verbatim.

Run only after src/train_lora.py has produced an adapter:
    python -m src.evaluate --adapter outputs/lora-distilgpt2 \
        --test data/processed/test.jsonl --out reports/evaluation.md
"""
from __future__ import annotations

import argparse
import json
import math

PROMPTS = [
    "### Instruction:\nGive three tips for staying hydrated during exercise.\n\n### Response:",
    "### Instruction:\nWhat is the difference between a virus and a bacterium?\n\n### Response:",
    "### Instruction:\nSummarize why the sky is blue in one sentence.\n\n### Response:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default="distilgpt2")
    ap.add_argument("--test", default="data/processed/test.jsonl")
    ap.add_argument("--out", default="reports/evaluation.md")
    ap.add_argument("--max-examples", type=int, default=100)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def perplexity(model) -> float:
        total_loss, total_tokens = 0.0, 0
        with open(args.test, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.max_examples:
                    break
                text = json.loads(line)["text"]
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    out = model(**enc, labels=enc["input_ids"])
                total_loss += out.loss.item() * enc["input_ids"].shape[1]
                total_tokens += enc["input_ids"].shape[1]
        return math.exp(total_loss / total_tokens)

    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_ppl = perplexity(base)
    tuned = PeftModel.from_pretrained(base, args.adapter)
    tuned_ppl = perplexity(tuned)

    lines = ["# Evaluation (held-out test split)", "",
             f"- Base distilgpt2 perplexity: {base_ppl:.2f}",
             f"- LoRA-tuned perplexity: {tuned_ppl:.2f}",
             f"- Examples evaluated: {args.max_examples}", "", "## Generations (tuned model, greedy)", ""]
    tuned.eval()
    for prompt in PROMPTS:
        enc = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            gen = tuned.generate(**enc, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        lines.append(f"Prompt: {prompt.splitlines()[1]}")
        lines.append("")
        lines.append(tokenizer.decode(gen[0], skip_special_tokens=True))
        lines.append("")
    Path_out = args.out
    from pathlib import Path
    Path(Path_out).parent.mkdir(parents=True, exist_ok=True)
    Path(Path_out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()

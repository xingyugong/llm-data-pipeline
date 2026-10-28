"""Data cleaning / decontamination for LLM instruction fine-tuning.

Source: Databricks Dolly 15k (CC BY-SA 3.0, human-written instruction/response
pairs). No synthetic data is generated here: every kept example is a real
source example that survived the filters below.

Pipeline, in order:
  1. Load raw examples (instruction, context, response, category).
  2. Exact de-duplication on normalized (instruction, context, response).
  3. Quality filters (length, language, PII-like patterns, repetition).
  4. Near-duplicate removal on (instruction, context) with shingle Jaccard.
  5. Hash-based train/test split, so no near-duplicate can leak across the
     split: the split key is the normalized instruction+context.
  6. Format to a single instruction-tuning JSONL text field.

Run:
    python -m src.clean_data --limit 2000 --test-frac 0.05 \
        --out-dir data/processed --report reports/data_quality_report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip().lower())


def example_key(instruction: str, context: str, response: str) -> str:
    return hashlib.sha256(
        "\x00".join(
            [normalize(instruction), normalize(context), normalize(response)]
        ).encode("utf-8")
    ).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    words = normalize(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_repetitive(text: str) -> bool:
    """Flag text dominated by one repeated character or token."""
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    chars = Counter(stripped)
    return chars.most_common(1)[0][1] / len(stripped) > 0.5


def quality_ok(instruction: str, context: str, response: str) -> str | None:
    """Return None if the example passes, else the rejection reason."""
    if len(instruction.strip()) < 5:
        return "instruction_too_short"
    if len(response.strip()) < 10:
        return "response_too_short"
    if len(response) > 4000:
        return "response_too_long"
    if len(instruction) > 2000:
        return "instruction_too_long"
    combined = f"{instruction} {context} {response}"
    ascii_ratio = sum(ord(c) < 128 for c in combined) / max(len(combined), 1)
    if ascii_ratio < 0.90:
        return "non_english_heavy"
    if EMAIL_RE.search(combined) or PHONE_RE.search(combined):
        return "pii_like_pattern"
    if URL_RE.search(response):
        return "url_in_response"
    if is_repetitive(response):
        return "repetitive_response"
    return None


def format_prompt(instruction: str, context: str, response: str) -> str:
    parts = ["### Instruction:", instruction.strip()]
    if context.strip():
        parts += ["", "### Input:", context.strip()]
    parts += ["", "### Response:", response.strip()]
    return "\n".join(parts)


def clean_records(records: list[dict], test_frac: float = 0.05):
    """Run dedup + filters + near-dup removal + hash split. Returns dict."""
    stats: Counter = Counter()
    seen_exact: set[str] = set()
    kept: list[dict] = []

    for rec in records:
        stats["raw"] += 1
        instruction = rec.get("instruction", "")
        context = rec.get("context", "")
        response = rec.get("response", "")
        key = example_key(instruction, context, response)
        if key in seen_exact:
            stats["exact_duplicate"] += 1
            continue
        seen_exact.add(key)
        reason = quality_ok(instruction, context, response)
        if reason:
            stats[reason] += 1
            continue
        kept.append(
            {
                "instruction": instruction.strip(),
                "context": context.strip(),
                "response": response.strip(),
                "category": rec.get("category", ""),
                "split_key": hashlib.sha256(
                    normalize(f"{instruction} {context}").encode("utf-8")
                ).hexdigest(),
            }
        )

    # Near-duplicate removal, greedy in source order, on instruction+context.
    seen_shingles: list[set[str]] = []
    deduped: list[dict] = []
    for rec in kept:
        sh = shingles(f"{rec['instruction']} {rec['context']}")
        if any(jaccard(sh, prev) >= 0.85 for prev in seen_shingles):
            stats["near_duplicate"] += 1
            continue
        seen_shingles.append(sh)
        deduped.append(rec)
    stats["kept_before_split"] = len(deduped)

    # Hash split: deterministic, no split-key overlap between train/test.
    train, test = [], []
    for rec in deduped:
        bucket = int(rec["split_key"][:8], 16) / 0xFFFFFFFF
        (test if bucket < test_frac else train).append(rec)
    stats["train"] = len(train)
    stats["test"] = len(test)
    return {"train": train, "test": test, "stats": dict(stats)}


def load_dolly(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(r) for r in ds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--report", default="reports/data_quality_report.md")
    args = ap.parse_args()

    records = load_dolly(args.limit)
    result = clean_records(records, test_frac=args.test_frac)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        rows = [
            {"text": format_prompt(r["instruction"], r["context"], r["response"]),
             "category": r["category"]}
            for r in result[split]
        ]
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data quality report", "",
             f"Source: databricks/databricks-dolly-15k (CC BY-SA 3.0), limit={args.limit}",
             f"Kept: {result['stats'].get('kept_before_split', 0)} / {result['stats'].get('raw', 0)} raw",
             f"Train: {result['stats'].get('train', 0)} · Test: {result['stats'].get('test', 0)} (hash split, no split-key overlap)",
             "", "## Rejections", ""]
    for k, v in sorted(result["stats"].items()):
        lines.append(f"- {k}: {v}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["stats"], indent=2))
    print(f"Wrote {out}/train.jsonl, {out}/test.jsonl, {report}")


if __name__ == "__main__":
    main()

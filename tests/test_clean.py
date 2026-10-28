import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean_data import clean_records, format_prompt, quality_ok, shingles


def test_exact_duplicate_removed():
    rec = {"instruction": "Say hi", "context": "", "response": "Hello there, friend.", "category": "open_qa"}
    out = clean_records([rec, rec, rec], test_frac=0.0)
    assert out["stats"]["kept_before_split"] == 1
    assert out["stats"]["exact_duplicate"] == 2


def test_quality_filters_reject_pii_and_short():
    assert quality_ok("What is the answer?", "", "x") == "response_too_short"
    assert quality_ok("What is 2+2?", "", "Call me at test@example.com please.") == "pii_like_pattern"
    assert quality_ok("What is 2+2?", "", "Four, because two plus two equals four.") is None


def test_format_prompt_includes_input_only_when_present():
    assert "### Input:" not in format_prompt("Do it", "", "Done.")
    assert "### Input:" in format_prompt("Do it", "some context", "Done.")


def test_near_duplicate_and_split_no_leak():
    a = {"instruction": "Explain photosynthesis in detail please",
         "context": "biology class notes about plants and sunlight",
         "response": "Plants convert sunlight, water and carbon dioxide into glucose and oxygen.",
         "category": "general_qa"}
    b = dict(a, response="Plants use sunlight, water and carbon dioxide to make glucose and oxygen.")
    out = clean_records([a, b], test_frac=0.0)
    assert out["stats"]["kept_before_split"] == 1
    # Hash split: same split key can never appear on both sides.
    out = clean_records([a, b, dict(a, instruction="Something completely different about history")], test_frac=0.5)
    train_keys = {r["split_key"] for r in out["train"]}
    test_keys = {r["split_key"] for r in out["test"]}
    assert not train_keys & test_keys


def test_shingles_jaccard():
    assert len(shingles("one two three four five six")) == 2

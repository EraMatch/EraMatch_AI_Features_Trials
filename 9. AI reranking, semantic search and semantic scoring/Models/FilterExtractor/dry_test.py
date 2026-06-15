"""
Dry-test suite for the filter-extraction fine-tuning pipeline.

Validates:
  1. Dataset integrity (all lines load, all JSON valid, required keys present)
  2. Sequence-length distribution (warns if many examples exceed max_seq_len)
  3. Prompt formatting (ChatML round-trip via transformers tokenizer only — no GPU needed)
  4. [Optional, --with-model] Full model load + sample inference via Unsloth

Usage:
    # Fast checks only (no GPU required)
    python dry_test.py

    # Include model load + inference (requires GPU + unsloth installed)
    python dry_test.py --with-model
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Inline config (mirrors train.py)
# ---------------------------------------------------------------------------
DATASET_PATH = "deepseek_tech_filters_5k.jsonl"
MODEL_NAME = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024
REQUIRED_KEYS = {"query", "target_filters"}
VALID_FILTER_KEYS = {
    # Core fields
    "role", "skills", "preferred_skills", "min_experience_years", "remote_ok",
    "timezone", "location", "employment_type", "seniority_level", "industry",
    "start_date", "visa_sponsorship", "certifications", "education_level",
    # Common HR-specific fields (high frequency in dataset)
    "no_c2c", "citizenship_required", "contract_duration", "direct_hire",
    "experience_domain", "security_clearance", "startup_experience",
    "preferred_certifications", "preferred_experience_years",
    "preferred_experience_domain", "max_experience_years",
}

SYSTEM_PROMPT = (
    "You are an expert HR filter extractor. "
    "Given a natural language job requirement query, extract structured filters as a valid JSON object.\n\n"
    "Output ONLY valid JSON. Include only fields that are clearly present in the query. Common fields:\n"
    "  role, skills, preferred_skills, min_experience_years, remote_ok, timezone, location,\n"
    "  employment_type, seniority_level, industry, start_date, visa_sponsorship,\n"
    "  certifications, education_level, no_c2c, citizenship_required, contract_duration,\n"
    "  direct_hire, experience_domain, security_clearance, startup_experience"
)

SAMPLE_QUERIES = [
    "Need a senior Python backend dev with FastAPI and PostgreSQL, 5+ yrs, remote OK, full-time.",
    "looking 4 ml engineer pytorch sklearn 3yrs exp nyc onsite contract",
    "Sr. DevOps with AWS + K8s + Terraform. Must have 4+ years. EST timezone, full-time, no visa sponsorship.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

_failures = 0


def ok(msg: str):
    print(f"  {PASS} {msg}")


def fail(msg: str):
    global _failures
    _failures += 1
    print(f"  {FAIL} {msg}")


def warn(msg: str):
    print(f"  {WARN} {msg}")


def info(msg: str):
    print(f"  {INFO} {msg}")


# ---------------------------------------------------------------------------
# Test 1: Dataset integrity
# ---------------------------------------------------------------------------

def test_dataset_integrity(path: str) -> list[dict]:
    print("\n=== Test 1: Dataset integrity ===")
    p = Path(path)
    if not p.exists():
        fail(f"Dataset file not found: {path}")
        sys.exit(1)

    records = []
    bad_json = 0
    missing_keys = 0
    unknown_filter_keys: Counter = Counter()

    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                bad_json += 1
                fail(f"Line {line_no}: JSON decode error — {e}")
                continue

            missing = REQUIRED_KEYS - rec.keys()
            if missing:
                missing_keys += 1
                fail(f"Line {line_no}: missing keys {missing}")
                continue

            if not isinstance(rec["target_filters"], dict):
                fail(f"Line {line_no}: target_filters is not a dict")
                continue

            unknown = set(rec["target_filters"].keys()) - VALID_FILTER_KEYS
            for k in unknown:
                unknown_filter_keys[k] += 1

            records.append(rec)

    ok(f"Loaded {len(records)} valid records")
    if bad_json:
        fail(f"{bad_json} lines with bad JSON")
    if missing_keys:
        fail(f"{missing_keys} records missing required keys")
    if unknown_filter_keys:
        warn(f"Unknown filter keys (not in VALID_FILTER_KEYS): {dict(unknown_filter_keys)}")
    else:
        ok("All filter keys are within the expected schema")

    # Field coverage stats
    field_counts: Counter = Counter()
    for r in records:
        for k in r["target_filters"]:
            field_counts[k] += 1
    info("Filter field frequency:")
    for k, v in field_counts.most_common():
        pct = 100 * v / len(records)
        print(f"      {k:<25} {v:>5}  ({pct:.1f}%)")

    return records


# ---------------------------------------------------------------------------
# Test 2: Sequence length analysis (tokenizer only)
# ---------------------------------------------------------------------------

def test_sequence_lengths(records: list[dict], tokenizer):
    print("\n=== Test 2: Sequence length distribution ===")

    def make_text(r):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["query"]},
            {"role": "assistant", "content": json.dumps(r["target_filters"], ensure_ascii=False)},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    lengths = []
    for r in records:
        text = make_text(r)
        tokens = tokenizer(text, return_length=True)["length"]
        lengths.append(tokens[0] if isinstance(tokens, list) else tokens)

    lengths.sort()
    n = len(lengths)
    too_long = sum(1 for l in lengths if l > MAX_SEQ_LENGTH)

    info(f"Min length   : {lengths[0]} tokens")
    info(f"Median length: {lengths[n // 2]} tokens")
    info(f"p95 length   : {lengths[int(0.95 * n)]} tokens")
    info(f"Max length   : {lengths[-1]} tokens")

    if too_long == 0:
        ok(f"All examples fit within max_seq_len={MAX_SEQ_LENGTH}")
    else:
        pct = 100 * too_long / n
        warn(f"{too_long} examples ({pct:.1f}%) exceed max_seq_len={MAX_SEQ_LENGTH} and will be truncated")


# ---------------------------------------------------------------------------
# Test 3: Prompt formatting round-trip
# ---------------------------------------------------------------------------

def test_prompt_format(records: list[dict], tokenizer):
    print("\n=== Test 3: Prompt formatting ===")

    for i, r in enumerate(records[:3]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["query"]},
            {"role": "assistant", "content": json.dumps(r["target_filters"], ensure_ascii=False)},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Basic sanity checks
        if "<|im_start|>" not in text and "<|system|>" not in text:
            fail(f"Record {i}: chat template does not contain expected tokens")
        elif len(text) < 50:
            fail(f"Record {i}: formatted text is suspiciously short ({len(text)} chars)")
        else:
            ok(f"Record {i}: formatted OK ({len(text)} chars)")

    # Print one full example so the user can visually inspect it
    sample = records[0]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["query"]},
        {"role": "assistant", "content": json.dumps(sample["target_filters"], ensure_ascii=False)},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    print("\n--- Sample formatted example ---")
    print(rendered[:800])
    if len(rendered) > 800:
        print(f"  ... ({len(rendered) - 800} more chars)")
    print("--- end ---\n")


# ---------------------------------------------------------------------------
# Test 4: Model load + inference (optional, requires GPU)
# ---------------------------------------------------------------------------

def test_model_inference():
    print("\n=== Test 4: Model load + inference (GPU) ===")
    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError:
        fail("unsloth not installed — skipping model test")
        return

    if not torch.cuda.is_available():
        warn("No CUDA device found — skipping model inference test")
        return

    print(f"  Loading {MODEL_NAME} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    ok("Model loaded successfully")

    for i, query in enumerate(SAMPLE_QUERIES):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"\n  --- Sample {i + 1} ---")
        print(f"  Query : {query}")
        print(f"  Output: {raw_output}")

        # Try to parse the output as JSON
        try:
            parsed = json.loads(raw_output)
            ok(f"Sample {i + 1}: output is valid JSON with keys: {list(parsed.keys())}")
        except json.JSONDecodeError:
            warn(f"Sample {i + 1}: output is NOT valid JSON (base model — expected after fine-tuning)")


# ---------------------------------------------------------------------------
# Tokenizer-only loader (no GPU, no unsloth)
# ---------------------------------------------------------------------------

def load_tokenizer_only():
    """Load just the tokenizer using HuggingFace transformers (no GPU needed)."""
    try:
        from transformers import AutoTokenizer
        hf_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        print(f"  Loading tokenizer from HuggingFace: {hf_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        ok("Tokenizer loaded")
        return tokenizer
    except Exception as e:
        fail(f"Could not load tokenizer: {e}")
        print("  Hint: run `pip install transformers` and ensure internet access.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Dry-test the filter extraction pipeline")
    parser.add_argument(
        "--dataset", default=DATASET_PATH, help="Path to the .jsonl dataset"
    )
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Also load the model and run sample inference (requires GPU + unsloth)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Filter Extraction Fine-Tuning — Dry Test")
    print("=" * 60)

    # Tests 1–3 only need the tokenizer (CPU-friendly)
    records = test_dataset_integrity(args.dataset)

    print("\n  Loading tokenizer (no GPU needed) ...")
    tokenizer = load_tokenizer_only()

    test_sequence_lengths(records, tokenizer)
    test_prompt_format(records, tokenizer)

    # Test 4 is opt-in
    if args.with_model:
        test_model_inference()
    else:
        print("\n  [Skipped] Test 4: Model inference (pass --with-model to enable)")

    print("\n" + "=" * 60)
    if _failures == 0:
        print(f"  {PASS} All tests passed — pipeline looks good!")
    else:
        print(f"  {FAIL} {_failures} failure(s) — fix before training.")
    print("=" * 60)

    sys.exit(0 if _failures == 0 else 1)


if __name__ == "__main__":
    main()

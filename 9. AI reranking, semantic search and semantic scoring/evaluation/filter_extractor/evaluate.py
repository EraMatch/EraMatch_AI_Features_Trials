"""
Evaluate fine-tuned FilterExtractor checkpoints against the held-out test split.

Default: evaluates ALL checkpoints found under Models/FilterExtractor/.

Usage:
    # All checkpoints (default — no flags needed)
    python evaluate.py

    # All checkpoints, custom models dir
    python evaluate.py --models_dir ../../Models/FilterExtractor

    # Single checkpoint only
    python evaluate.py --model_path ../../Models/FilterExtractor/.../checkpoint-948

    # Quick sanity check (50 samples)
    python evaluate.py --max_samples 50

    # Include LLM-as-judge (reads ollama_api_keys.txt)
    python evaluate.py --with_llm_judge

    # Custom output path
    python evaluate.py --output evaluation_results.md
"""

import argparse
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (relative to this file's location)
# ---------------------------------------------------------------------------
_HERE             = Path(__file__).parent
_REPO_ROOT        = _HERE.parent.parent
_MODELS_DIR       = _REPO_ROOT / "Models" / "FilterExtractor"
_DATASET          = _REPO_ROOT / "Data generation" / "deepseek_tech_filters_5k.jsonl"
_OLLAMA_KEYS_FILE = _REPO_ROOT / "Data generation" / "ollama_api_keys.txt"

OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
OLLAMA_JUDGE_MODEL    = "gpt-oss:120b-cloud"

OLLAMA_LOCAL_BASE_URL = "http://localhost:11434/v1"
OLLAMA_LOCAL_JUDGE_MODEL = "qwen2.5-coder:7b"

# Reuse load_model / extract_filters from the existing inference script
sys.path.insert(0, str(_MODELS_DIR))
from inference import load_model, extract_filters  # noqa: E402

# ---------------------------------------------------------------------------
# Field-type classification
# ---------------------------------------------------------------------------
LIST_FIELDS = {
    "skills", "preferred_skills", "certifications",
    "preferred_certifications", "languages",
}
INT_FIELDS = {
    "min_experience_years", "preferred_experience_years", "min_automation_years",
}
BOOL_FIELDS = {
    "remote_ok", "visa_sponsorship", "no_c2c", "direct_hire", "startup_experience",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_test_split(
    data_path: Path,
    test_size: float = 0.05,
    seed: int = 42,
    max_samples: int = None,
) -> list[dict]:
    """Reproduce the exact held-out split from train.py (seed=42, test_size=0.05)."""
    records = load_jsonl(data_path)
    ds = Dataset.from_list([{"idx": i} for i in range(len(records))])
    split = ds.train_test_split(test_size=test_size, seed=seed)
    test_indices = split["test"]["idx"]
    test_records = [records[i] for i in test_indices]
    if max_samples:
        test_records = test_records[:max_samples]
    return test_records


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _normalize(v):
    """Lowercase strings; convert list to a lowercase set."""
    if isinstance(v, list):
        return {str(x).lower().strip() for x in v}
    if isinstance(v, str):
        return v.lower().strip()
    return v


def _values_match(pred_v, target_v) -> bool:
    return _normalize(pred_v) == _normalize(target_v)


def _list_f1(pred_v, target_v) -> float:
    """Set-intersection F1 for list-valued filter fields."""
    to_set = lambda v: {str(x).lower().strip() for x in (v if isinstance(v, list) else [v])}
    pred_s, tgt_s = to_set(pred_v), to_set(target_v)
    if not pred_s and not tgt_s:
        return 1.0
    if not pred_s or not tgt_s:
        return 0.0
    inter = pred_s & tgt_s
    p = len(inter) / len(pred_s)
    r = len(inter) / len(tgt_s)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _token_f1(pred_str: str, target_str: str) -> float:
    """Multiset token F1 between two serialised JSON strings."""
    tokens = lambda s: re.findall(r"\w+", s.lower())
    pc, tc = Counter(tokens(pred_str)), Counter(tokens(target_str))
    if not pc and not tc:
        return 1.0
    if not pc or not tc:
        return 0.0
    common = sum((pc & tc).values())
    p = common / sum(pc.values())
    r = common / sum(tc.values())
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list[dict],
    queries: list[str],
    targets: list[dict],
    with_llm_judge: bool = False,
    api_keys: list[str] = None,
    judge_base_url: str = OLLAMA_CLOUD_BASE_URL,
    judge_model: str = OLLAMA_JUDGE_MODEL,
) -> dict:
    """Compute all three metric groups and return a flat results dict."""
    n = len(predictions)

    # --- Group 1 accumulators ---
    valid_count = 0
    key_precisions, key_recalls = [], []
    val_match_counts, val_total_counts = [], []
    list_f1s_per_sample = []
    numeric_errors: dict[str, list] = defaultdict(list)
    per_field_correct: dict[str, int] = defaultdict(int)
    per_field_total: dict[str, int] = defaultdict(int)

    # --- NLP accumulators ---
    pred_strings, target_strings = [], []

    for pred, target in zip(predictions, targets):
        is_valid = "_parse_error" not in pred
        if is_valid:
            valid_count += 1

        pred_keys   = set(pred.keys())   if is_valid else set()
        target_keys = set(target.keys())

        # Key-set metrics
        if is_valid and pred_keys:
            tp = pred_keys & target_keys
            key_precisions.append(len(tp) / len(pred_keys))
            key_recalls.append(len(tp) / len(target_keys) if target_keys else 1.0)
        else:
            key_precisions.append(0.0)
            key_recalls.append(0.0)

        # Value and per-field metrics
        v_match, v_total, sample_list_f1s = 0, 0, []
        for key in target_keys:
            per_field_total[key] += 1
            if not is_valid or key not in pred:
                continue
            pred_v, target_v = pred[key], target[key]
            v_total += 1

            if key in LIST_FIELDS:
                f1 = _list_f1(pred_v, target_v)
                sample_list_f1s.append(f1)
                if f1 == 1.0:
                    v_match += 1
                    per_field_correct[key] += 1

            elif key in INT_FIELDS:
                try:
                    err = abs(int(pred_v) - int(target_v))
                    numeric_errors[key].append(err)
                    if err == 0:
                        v_match += 1
                        per_field_correct[key] += 1
                except (ValueError, TypeError):
                    pass

            else:  # strings, bools
                if _values_match(pred_v, target_v):
                    v_match += 1
                    per_field_correct[key] += 1

        val_match_counts.append(v_match)
        val_total_counts.append(v_total)
        if sample_list_f1s:
            list_f1s_per_sample.append(sum(sample_list_f1s) / len(sample_list_f1s))

        # Serialise for NLP metrics (sorted keys → deterministic)
        pred_str   = json.dumps(pred if is_valid else {}, sort_keys=True, ensure_ascii=False)
        target_str = json.dumps(target,                   sort_keys=True, ensure_ascii=False)
        pred_strings.append(pred_str)
        target_strings.append(target_str)

    # Aggregate Group 1
    avg_prec   = sum(key_precisions) / n
    avg_recall = sum(key_recalls)    / n
    denom      = avg_prec + avg_recall
    key_f1     = 2 * avg_prec * avg_recall / denom if denom > 0 else 0.0

    total_val = sum(val_total_counts)
    value_match_rate = sum(val_match_counts) / total_val if total_val > 0 else 0.0
    list_field_f1    = sum(list_f1s_per_sample) / len(list_f1s_per_sample) if list_f1s_per_sample else 0.0

    numeric_mae = {k: sum(v) / len(v) for k, v in numeric_errors.items()}
    avg_numeric_mae = sum(numeric_mae.values()) / len(numeric_mae) if numeric_mae else 0.0

    per_field_acc = {
        k: per_field_correct[k] / per_field_total[k]
        for k in per_field_total
    }

    # --- Group 2: NLP ---
    from rouge_score import rouge_scorer as rouge_lib
    rouge_scorer = rouge_lib.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1s, rLs, tf1s = [], [], []
    for ps, ts in zip(pred_strings, target_strings):
        scores = rouge_scorer.score(ts, ps)
        r1s.append(scores["rouge1"].fmeasure)
        rLs.append(scores["rougeL"].fmeasure)
        tf1s.append(_token_f1(ps, ts))

    rouge_1  = sum(r1s)  / n
    rouge_l  = sum(rLs)  / n
    token_f1 = sum(tf1s) / n

    # BERTScore (uses GPU if available)
    from bert_score import score as bert_score_fn
    _, _, bert_f1_tensor = bert_score_fn(
        pred_strings, target_strings,
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    bert_f1 = bert_f1_tensor.mean().item()

    # --- Group 3: LLM-as-judge (optional) ---
    llm_judge_mean = None
    llm_judge_pass_rate = None
    llm_judge_reasons: list[str] = []

    is_local = judge_base_url == OLLAMA_LOCAL_BASE_URL
    if with_llm_judge and (api_keys or is_local):
        from openai import OpenAI

        # Local Ollama: single client with a dummy key, no rotation.
        # Cloud: rotate across multiple API keys so requests are spread evenly.
        if is_local:
            clients = itertools.cycle([OpenAI(api_key="ollama", base_url=judge_base_url)])
        else:
            clients = itertools.cycle([
                OpenAI(api_key=k, base_url=judge_base_url) for k in api_keys
            ])

        JUDGE_SYS = (
            "You are evaluating a filter extraction model.\n"
            "Given a recruiter query, the model's predicted filters, and the ground-truth "
            "filters, score the prediction:\n"
            "  3 = Correct (all critical filters present; minor wording differences are OK)\n"
            "  2 = Mostly correct (1–2 minor errors or omissions)\n"
            "  1 = Partially correct (major fields missing or wrong)\n"
            "  0 = Completely wrong or unparseable\n"
            'Output ONLY valid JSON: {"score": int, "reason": str}'
        )

        judge_scores = []
        for query, pred, target in tqdm(
            zip(queries, predictions, targets), total=n, desc="LLM Judge"
        ):
            try:
                user_msg = (
                    f'Query: "{query}"\n\n'
                    f"Predicted: {json.dumps(pred, ensure_ascii=False)}\n\n"
                    f"Target: {json.dumps(target, ensure_ascii=False)}"
                )
                resp = next(clients).chat.completions.create(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYS},
                        {"role": "user",   "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                verdict = json.loads(resp.choices[0].message.content)
                judge_scores.append(int(verdict.get("score", 0)))
                llm_judge_reasons.append(verdict.get("reason", ""))
            except Exception as exc:
                judge_scores.append(0)
                llm_judge_reasons.append(f"[Error: {exc}]")

        llm_judge_mean      = sum(judge_scores) / len(judge_scores)
        llm_judge_pass_rate = sum(1 for s in judge_scores if s >= 2) / len(judge_scores)

    # --- Aggregate scores ---
    # Group 1: structural quality (key coverage + value correctness)
    g1_score = (
        0.35 * key_f1
        + 0.35 * value_match_rate
        + 0.20 * list_field_f1
        + 0.10 * (valid_count / n)
    )

    # Group 2: NLP similarity (BERTScore weighted highest as most semantic)
    g2_score = (
        0.45 * bert_f1
        + 0.30 * rouge_l
        + 0.15 * token_f1
        + 0.10 * rouge_1
    )

    # Overall: LLM judge carries 50 % when available; redistributed when absent.
    # llm_judge_mean is 0–3 → normalise to 0–1 before weighting.
    if llm_judge_mean is not None:
        judge_norm   = llm_judge_mean / 3.0
        overall_score = 0.25 * g1_score + 0.25 * g2_score + 0.50 * judge_norm
    else:
        overall_score = 0.55 * g1_score + 0.45 * g2_score

    return {
        # Group 1 — structural
        "json_valid_rate":    valid_count / n,
        "key_precision":      avg_prec,
        "key_recall":         avg_recall,
        "key_f1":             key_f1,
        "value_match_rate":   value_match_rate,
        "list_field_f1":      list_field_f1,
        "avg_numeric_mae":    avg_numeric_mae,
        "numeric_mae":        numeric_mae,
        "per_field_accuracy": per_field_acc,
        # Group 2 — NLP
        "rouge_1":            rouge_1,
        "rouge_l":            rouge_l,
        "token_f1":           token_f1,
        "bert_score_f1":      bert_f1,
        # Group 3 — LLM judge
        "llm_judge_mean":     llm_judge_mean,
        "llm_judge_pass_rate": llm_judge_pass_rate,
        "llm_judge_reasons":  llm_judge_reasons,
        # Aggregates
        "g1_score":           g1_score,
        "g2_score":           g2_score,
        "overall_score":      overall_score,
    }


# ---------------------------------------------------------------------------
# Checkpoint runner
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    ckpt_path: Path,
    test_data: list[dict],
    with_llm_judge: bool = False,
    api_keys: list[str] = None,
    judge_base_url: str = OLLAMA_CLOUD_BASE_URL,
    judge_model: str = OLLAMA_JUDGE_MODEL,
) -> dict:
    """Load model → infer → compute metrics → free VRAM → return results."""
    model, tokenizer = load_model(str(ckpt_path))
    queries = [r["query"]          for r in test_data]
    targets = [r["target_filters"] for r in test_data]
    preds   = [
        extract_filters(q, model, tokenizer)
        for q in tqdm(queries, desc="Inference", unit="query")
    ]
    del model
    torch.cuda.empty_cache()
    return compute_metrics(
        preds, queries, targets,
        with_llm_judge=with_llm_judge,
        api_keys=api_keys,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
    )


def find_all_checkpoints(models_dir: Path) -> list[Path]:
    """Return all checkpoint-* subdirs sorted by model name then step number."""
    return sorted(models_dir.glob("*/checkpoint-*"))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    # Group 1 — structural
    "json_valid_rate", "key_precision", "key_recall", "key_f1",
    "value_match_rate", "list_field_f1",
    # Group 2 — NLP
    "rouge_1", "rouge_l", "token_f1", "bert_score_f1",
    # Group 3 — LLM judge
    "llm_judge_mean",
    # Aggregates
    "g1_score", "g2_score", "overall_score",
]


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _label(ckpt_path: Path) -> str:
    """Short human-readable label for a checkpoint path."""
    model = ckpt_path.parent.name
    for strip in ("qwen_filter_extractor_", "-instruct-bnb-4bit", "-Instruct-bnb-4bit"):
        model = model.replace(strip, "")
    return f"{model}/{ckpt_path.name}"


def print_comparison_table(results: dict) -> None:
    labels = list(results.keys())

    # Best value per column
    best: dict[str, float] = {}
    for col in _SUMMARY_COLS:
        vals = [results[l][col] for l in labels
                if isinstance(results[l].get(col), float)]
        if vals:
            best[col] = max(vals)

    col_w = 15
    pad   = 38

    print("\n" + "=" * (pad + col_w * len(_SUMMARY_COLS)))
    print("EVALUATION SUMMARY")
    print("=" * (pad + col_w * len(_SUMMARY_COLS)))
    hdr = f"{'Checkpoint':<{pad}}" + "".join(f"{c[:col_w]:>{col_w}}" for c in _SUMMARY_COLS)
    print(hdr)
    print("-" * len(hdr))

    for label, metrics in results.items():
        row = f"{_label(Path(label)):<{pad}}"
        for col in _SUMMARY_COLS:
            v = metrics.get(col)
            s = _fmt(v)
            is_best = (isinstance(v, float) and col in best
                       and abs(v - best[col]) < 1e-9)
            row += f"{'*' + s if is_best else ' ' + s:>{col_w}}"
        print(row)
    print("(* = best per column)\n")


def build_markdown_report(
    results: dict,
    test_n: int,
    with_llm_judge: bool,
    output_path: Path,
) -> None:
    today  = date.today().isoformat()
    labels = list(results.keys())

    # Best per column for bolding
    best: dict[str, float] = {}
    for col in _SUMMARY_COLS:
        vals = [results[l][col] for l in labels
                if isinstance(results[l].get(col), float)]
        if vals:
            best[col] = max(vals)

    short = [_label(Path(l)) for l in labels]
    lines: list[str] = []

    judge_note = (
        "LLM judge present — overall = 0.25×G1 + 0.25×G2 + 0.50×judge"
        if with_llm_judge
        else "no LLM judge — overall = 0.55×G1 + 0.45×G2"
    )

    # Header
    lines += [
        f"# FilterExtractor Evaluation — {today}",
        "",
        f"**Test samples:** {test_n} &nbsp;|&nbsp; "
        f"**Dataset:** `deepseek_tech_filters_5k.jsonl`",
        "",
        f"> **Scoring:** G1 = 0.35×key\\_f1 + 0.35×value\\_match + 0.20×list\\_f1 + 0.10×json\\_valid &nbsp;|&nbsp; "
        f"G2 = 0.45×bert\\_f1 + 0.30×rouge\\_l + 0.15×token\\_f1 + 0.10×rouge\\_1 &nbsp;|&nbsp; {judge_note}",
        "",
    ]

    # Summary table
    lines += ["## Summary", ""]
    header_row = ["Checkpoint"] + _SUMMARY_COLS
    lines.append("| " + " | ".join(header_row) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")
    for label, metrics in results.items():
        cells = [_label(Path(label))]
        for col in _SUMMARY_COLS:
            v = metrics.get(col)
            s = _fmt(v)
            if isinstance(v, float) and col in best and abs(v - best[col]) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Per-field accuracy table
    lines += ["## Per-Field Accuracy", ""]
    all_fields = sorted({
        k for m in results.values()
        for k in m.get("per_field_accuracy", {})
    })
    lines.append("| Field | " + " | ".join(short) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(short)) + " |")
    for field in all_fields:
        row = [field]
        for label in labels:
            row.append(_fmt(results[label].get("per_field_accuracy", {}).get(field)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Numeric MAE table
    all_num = sorted({
        k for m in results.values()
        for k in m.get("numeric_mae", {})
    })
    if all_num:
        lines += ["## Numeric Field MAE", ""]
        lines.append("| Field | " + " | ".join(short) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(short)) + " |")
        for field in all_num:
            row = [field]
            for label in labels:
                row.append(_fmt(results[label].get("numeric_mae", {}).get(field)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # LLM Judge section
    if with_llm_judge:
        lines += ["## LLM Judge (DeepSeek)", ""]
        lines.append("| Checkpoint | Mean Score (0–3) | Pass Rate (≥2) |")
        lines.append("| --- | --- | --- |")
        for label, metrics in results.items():
            lines.append(
                f"| {_label(Path(label))} "
                f"| {_fmt(metrics.get('llm_judge_mean'))} "
                f"| {_fmt(metrics.get('llm_judge_pass_rate'))} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate FilterExtractor checkpoints. "
                    "Default: all checkpoints under --models_dir. "
                    "Pass --model_path to evaluate a single checkpoint instead."
    )
    p.add_argument("--model_path",  default=None,
                   help="Evaluate a single checkpoint directory (overrides default all-checkpoints mode)")
    p.add_argument("--models_dir",  default=str(_MODELS_DIR),
                   help="Root dir scanned for all checkpoints when --model_path is not given")
    p.add_argument("--data",        default=str(_DATASET),
                   help="Path to the ground-truth JSONL dataset")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap test samples (useful for quick sanity checks)")
    p.add_argument("--with_llm_judge", action="store_true",
                   help="Run LLM-as-judge via Ollama Cloud API (reads keys from ollama_api_keys.txt)")
    p.add_argument("--local_judge", action="store_true",
                   help="Use a local Ollama model as judge instead of the cloud API (no API keys needed)")
    p.add_argument("--judge_model", default=None,
                   help="Override the judge model name (default: cloud=gpt-oss:120b-cloud, local=qwen2.5-coder:7b)")
    p.add_argument("--output",      default="evaluation_results.md",
                   help="Output markdown report file (default: evaluation_results.md)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve judge backend (local Ollama vs cloud)
    if args.local_judge:
        args.with_llm_judge = True  # --local_judge implies --with_llm_judge

    judge_base_url = OLLAMA_LOCAL_BASE_URL if args.local_judge else OLLAMA_CLOUD_BASE_URL
    judge_model    = args.judge_model or (
        OLLAMA_LOCAL_JUDGE_MODEL if args.local_judge else OLLAMA_JUDGE_MODEL
    )

    # API keys — only needed for cloud judge
    api_keys: list[str] = []
    if args.with_llm_judge and not args.local_judge:
        if _OLLAMA_KEYS_FILE.exists():
            api_keys = [
                line.strip()
                for line in _OLLAMA_KEYS_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if not api_keys:
            print("Warning: ollama_api_keys.txt not found or empty — skipping LLM judge.")
            args.with_llm_judge = False

    if args.with_llm_judge:
        backend = f"local ({judge_base_url})" if args.local_judge else "cloud"
        print(f"LLM judge: {judge_model}  [{backend}]")

    print(f"Loading test split from: {args.data}")
    test_data = load_test_split(Path(args.data), max_samples=args.max_samples)
    print(f"Test samples: {len(test_data)}")

    if args.model_path:
        checkpoints = [Path(args.model_path)]
    else:
        checkpoints = find_all_checkpoints(Path(args.models_dir))
    print(f"Checkpoints to evaluate: {len(checkpoints)}")

    results: dict[str, dict] = {}
    for ckpt in checkpoints:
        lbl = _label(ckpt)
        print(f"\n{'=' * 60}\nEvaluating: {lbl}\n{'=' * 60}")
        metrics = evaluate_checkpoint(
            ckpt, test_data,
            with_llm_judge=args.with_llm_judge,
            api_keys=api_keys,
            judge_base_url=judge_base_url,
            judge_model=judge_model,
        )
        results[str(ckpt)] = metrics
        print(
            f"  key_f1={metrics['key_f1']:.3f}  "
            f"value_match={metrics['value_match_rate']:.3f}  "
            f"bert_f1={metrics['bert_score_f1']:.3f}  "
            f"g1={metrics['g1_score']:.3f}  g2={metrics['g2_score']:.3f}  "
            f"overall={metrics['overall_score']:.3f}"
        )

    print_comparison_table(results)
    build_markdown_report(results, len(test_data), args.with_llm_judge, Path(args.output))


if __name__ == "__main__":
    main()

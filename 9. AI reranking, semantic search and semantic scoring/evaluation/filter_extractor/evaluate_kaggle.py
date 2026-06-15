"""
Kaggle-compatible FilterExtractor evaluator.

Setup (run these cells BEFORE this script in your notebook):

    # Cell 1 – system deps + Ollama
    !apt-get update && apt-get install -y zstd
    !curl -fsSL https://ollama.com/install.sh | sh

    # Cell 2 – Python packages
    !pip install ollama gdown rouge-score bert-score -q

    # Cell 3 – start Ollama server
    import os, subprocess, time
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia'
    os.environ['OLLAMA_NUM_PARALLEL'] = '4'
    subprocess.Popen("nohup ollama serve > ollama.log 2>&1 &", shell=True)
    time.sleep(5)

    # Cell 4 – pull the judge model (once)
    import ollama
    ollama.pull("qwen2.5-coder:7b")

Usage examples:

    # Evaluate a checkpoint downloaded from Google Drive (folder ID)
    python evaluate_kaggle.py --drive_folder_id 1ABC...xyz

    # If the checkpoint is a zip on Drive (file ID)
    python evaluate_kaggle.py --drive_file_id 1ABC...xyz

    # Point at a checkpoint already on disk
    python evaluate_kaggle.py --model_path /kaggle/working/my_checkpoint

    # Dataset from Drive (file ID of the .jsonl)
    python evaluate_kaggle.py --drive_folder_id 1ABC...xyz --data_drive_file_id 1DEF...uvw

    # Skip LLM judge (faster)
    python evaluate_kaggle.py --drive_folder_id 1ABC...xyz --no_judge

    # Quick sanity-check (50 samples)
    python evaluate_kaggle.py --drive_folder_id 1ABC...xyz --max_samples 50
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Kaggle / Ollama environment setup
# ---------------------------------------------------------------------------

def setup_kaggle_env() -> None:
    """Point the dynamic linker at Kaggle's NVIDIA drivers so Ollama can see the GPU."""
    os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib64-nvidia")
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")


def ensure_ollama_running(timeout: int = 15) -> None:
    """Start the Ollama server if it is not already listening on port 11434."""
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        print("Ollama already running.")
        return
    except Exception:
        pass

    print("Starting Ollama server...")
    subprocess.Popen(
        "nohup ollama serve > /kaggle/working/ollama.log 2>&1 &",
        shell=True,
    )
    for _ in range(timeout):
        time.sleep(1)
        try:
            urllib.request.urlopen("http://localhost:11434", timeout=1)
            print("Ollama server ready.")
            return
        except Exception:
            pass
    print("Warning: Ollama server did not respond within timeout — continuing anyway.")


def pull_judge_model(model_name: str) -> None:
    """Pull the Ollama judge model if it is not already cached."""
    import ollama as _ollama
    existing = {m["name"] for m in _ollama.list().get("models", [])}
    if any(model_name.split(":")[0] in e for e in existing):
        print(f"Judge model '{model_name}' already available.")
        return
    print(f"Pulling judge model '{model_name}' …")
    _ollama.pull(model_name)
    print("Judge model ready.")


# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

_DEFAULT_CKPT_DIR = Path("/kaggle/working/filter_extractor_checkpoint")
_DEFAULT_DATA_DIR = Path("/kaggle/working/eval_data")


def download_from_drive_folder(folder_id: str, dest: Path) -> Path:
    """
    Download a Google Drive *folder* with gdown.
    Returns the destination directory.
    """
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading Drive folder {folder_id} → {dest} …")
    ret = subprocess.run(
        ["gdown", "--folder", url, "-O", str(dest), "--remaining-ok"],
        check=True,
    )
    # gdown creates a sub-folder named after the Drive folder; find it
    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    return subdirs[0] if len(subdirs) == 1 else dest


def download_from_drive_file(file_id: str, dest: Path, filename: str = "download") -> Path:
    """
    Download a single Google Drive *file* (e.g. a zip archive or .jsonl).
    Returns the path of the downloaded file.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / filename
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading Drive file {file_id} → {out_path} …")
    subprocess.run(["gdown", url, "-O", str(out_path), "--remaining-ok"], check=True)
    return out_path


def maybe_unzip(path: Path) -> Path:
    """If path is a zip file, extract it next to it and return the extracted dir."""
    if path.suffix != ".zip":
        return path
    out_dir = path.parent / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {path} → {out_dir} …")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Inline inference (avoids sys.path juggling on Kaggle)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert HR filter extractor. "
    "Given a natural language job requirement query, extract structured filters as a valid JSON object.\n\n"
    "Output ONLY valid JSON. Include only fields that are clearly present in the query. Common fields:\n"
    "  role, skills, preferred_skills, min_experience_years, remote_ok, timezone, location,\n"
    "  employment_type, seniority_level, industry, start_date, visa_sponsorship,\n"
    "  certifications, education_level, no_c2c, citizenship_required, contract_duration,\n"
    "  direct_hire, experience_domain, security_clearance, startup_experience"
)

_MAX_NEW_TOKENS = 512


def load_model(model_path: str):
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        quantization_config=bnb_config,
    )
    print("Model ready.")
    return model, tokenizer


def extract_filters(query: str, model, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"_raw_output": raw, "_parse_error": True}


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_HERE     = Path(__file__).parent
_REPO     = _HERE.parent.parent
_DATASET  = _REPO / "Data generation" / "deepseek_tech_filters_5k.jsonl"

OLLAMA_LOCAL_BASE_URL    = "http://localhost:11434/v1"
OLLAMA_LOCAL_JUDGE_MODEL = "qwen2.5-coder:7b"

# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

LIST_FIELDS = {
    "skills", "preferred_skills", "certifications",
    "preferred_certifications", "languages",
}
INT_FIELDS  = {
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
    """Reproduce the exact held-out split used in train.py (seed=42, test_size=0.05)."""
    records = load_jsonl(data_path)
    ds      = Dataset.from_list([{"idx": i} for i in range(len(records))])
    split   = ds.train_test_split(test_size=test_size, seed=seed)
    test_records = [records[i] for i in split["test"]["idx"]]
    if max_samples:
        test_records = test_records[:max_samples]
    return test_records


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _normalize(v):
    if isinstance(v, list):
        return {str(x).lower().strip() for x in v}
    if isinstance(v, str):
        return v.lower().strip()
    return v


def _values_match(pred_v, target_v) -> bool:
    return _normalize(pred_v) == _normalize(target_v)


def _list_f1(pred_v, target_v) -> float:
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
    judge_model: str = OLLAMA_LOCAL_JUDGE_MODEL,
) -> dict:
    n = len(predictions)

    valid_count = 0
    key_precisions, key_recalls = [], []
    val_match_counts, val_total_counts = [], []
    list_f1s_per_sample = []
    numeric_errors: dict[str, list] = defaultdict(list)
    per_field_correct: dict[str, int] = defaultdict(int)
    per_field_total:   dict[str, int] = defaultdict(int)
    pred_strings, target_strings = [], []

    for pred, target in zip(predictions, targets):
        is_valid = "_parse_error" not in pred
        if is_valid:
            valid_count += 1

        pred_keys   = set(pred.keys())   if is_valid else set()
        target_keys = set(target.keys())

        if is_valid and pred_keys:
            tp = pred_keys & target_keys
            key_precisions.append(len(tp) / len(pred_keys))
            key_recalls.append(len(tp) / len(target_keys) if target_keys else 1.0)
        else:
            key_precisions.append(0.0)
            key_recalls.append(0.0)

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
            else:
                if _values_match(pred_v, target_v):
                    v_match += 1
                    per_field_correct[key] += 1

        val_match_counts.append(v_match)
        val_total_counts.append(v_total)
        if sample_list_f1s:
            list_f1s_per_sample.append(sum(sample_list_f1s) / len(sample_list_f1s))

        pred_str   = json.dumps(pred if is_valid else {}, sort_keys=True, ensure_ascii=False)
        target_str = json.dumps(target,                   sort_keys=True, ensure_ascii=False)
        pred_strings.append(pred_str)
        target_strings.append(target_str)

    # Group 1 aggregation
    avg_prec   = sum(key_precisions) / n
    avg_recall = sum(key_recalls)    / n
    denom      = avg_prec + avg_recall
    key_f1     = 2 * avg_prec * avg_recall / denom if denom > 0 else 0.0

    total_val        = sum(val_total_counts)
    value_match_rate = sum(val_match_counts) / total_val if total_val > 0 else 0.0
    list_field_f1    = (
        sum(list_f1s_per_sample) / len(list_f1s_per_sample) if list_f1s_per_sample else 0.0
    )
    numeric_mae     = {k: sum(v) / len(v) for k, v in numeric_errors.items()}
    avg_numeric_mae = sum(numeric_mae.values()) / len(numeric_mae) if numeric_mae else 0.0
    per_field_acc   = {k: per_field_correct[k] / per_field_total[k] for k in per_field_total}

    # Group 2 – NLP
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

    from bert_score import score as bert_score_fn
    _, _, bert_f1_tensor = bert_score_fn(
        pred_strings, target_strings,
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    bert_f1 = bert_f1_tensor.mean().item()

    # Group 3 – LLM judge via local Ollama
    llm_judge_mean      = None
    llm_judge_pass_rate = None
    llm_judge_reasons: list[str] = []

    if with_llm_judge:
        from openai import OpenAI
        client = OpenAI(api_key="ollama", base_url=OLLAMA_LOCAL_BASE_URL)

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
                resp = client.chat.completions.create(
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

    # Aggregate scores
    g1_score = (
        0.35 * key_f1
        + 0.35 * value_match_rate
        + 0.20 * list_field_f1
        + 0.10 * (valid_count / n)
    )
    g2_score = (
        0.45 * bert_f1
        + 0.30 * rouge_l
        + 0.15 * token_f1
        + 0.10 * rouge_1
    )

    if llm_judge_mean is not None:
        judge_norm    = llm_judge_mean / 3.0
        overall_score = 0.25 * g1_score + 0.25 * g2_score + 0.50 * judge_norm
    else:
        overall_score = 0.55 * g1_score + 0.45 * g2_score

    return {
        "json_valid_rate":     valid_count / n,
        "key_precision":       avg_prec,
        "key_recall":          avg_recall,
        "key_f1":              key_f1,
        "value_match_rate":    value_match_rate,
        "list_field_f1":       list_field_f1,
        "avg_numeric_mae":     avg_numeric_mae,
        "numeric_mae":         numeric_mae,
        "per_field_accuracy":  per_field_acc,
        "rouge_1":             rouge_1,
        "rouge_l":             rouge_l,
        "token_f1":            token_f1,
        "bert_score_f1":       bert_f1,
        "llm_judge_mean":      llm_judge_mean,
        "llm_judge_pass_rate": llm_judge_pass_rate,
        "llm_judge_reasons":   llm_judge_reasons,
        "g1_score":            g1_score,
        "g2_score":            g2_score,
        "overall_score":       overall_score,
    }


# ---------------------------------------------------------------------------
# Checkpoint runner
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    ckpt_path: Path,
    test_data: list[dict],
    with_llm_judge: bool = False,
    judge_model: str = OLLAMA_LOCAL_JUDGE_MODEL,
) -> dict:
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
        judge_model=judge_model,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    "json_valid_rate", "key_precision", "key_recall", "key_f1",
    "value_match_rate", "list_field_f1",
    "rouge_1", "rouge_l", "token_f1", "bert_score_f1",
    "llm_judge_mean",
    "g1_score", "g2_score", "overall_score",
]


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _label(ckpt_path: Path) -> str:
    model = ckpt_path.parent.name
    for strip in ("qwen_filter_extractor_", "-instruct-bnb-4bit", "-Instruct-bnb-4bit"):
        model = model.replace(strip, "")
    return f"{model}/{ckpt_path.name}"


def print_comparison_table(results: dict) -> None:
    labels = list(results.keys())
    best: dict[str, float] = {}
    for col in _SUMMARY_COLS:
        vals = [results[l][col] for l in labels if isinstance(results[l].get(col), float)]
        if vals:
            best[col] = max(vals)

    col_w, pad = 15, 38
    print("\n" + "=" * (pad + col_w * len(_SUMMARY_COLS)))
    print("EVALUATION SUMMARY")
    print("=" * (pad + col_w * len(_SUMMARY_COLS)))
    hdr = f"{'Checkpoint':<{pad}}" + "".join(f"{c[:col_w]:>{col_w}}" for c in _SUMMARY_COLS)
    print(hdr)
    print("-" * len(hdr))

    for label, metrics in results.items():
        row = f"{_label(Path(label)):<{pad}}"
        for col in _SUMMARY_COLS:
            v  = metrics.get(col)
            s  = _fmt(v)
            is_best = isinstance(v, float) and col in best and abs(v - best[col]) < 1e-9
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

    best: dict[str, float] = {}
    for col in _SUMMARY_COLS:
        vals = [results[l][col] for l in labels if isinstance(results[l].get(col), float)]
        if vals:
            best[col] = max(vals)

    short = [_label(Path(l)) for l in labels]
    lines: list[str] = []

    judge_note = (
        f"LLM judge present ({OLLAMA_LOCAL_JUDGE_MODEL}) — overall = 0.25×G1 + 0.25×G2 + 0.50×judge"
        if with_llm_judge
        else "no LLM judge — overall = 0.55×G1 + 0.45×G2"
    )

    lines += [
        f"# FilterExtractor Evaluation — {today}",
        "",
        f"**Test samples:** {test_n} &nbsp;|&nbsp; "
        f"**Dataset:** `deepseek_tech_filters_5k.jsonl`",
        "",
        f"> **Scoring:** G1 = 0.35×key\\_f1 + 0.35×value\\_match + 0.20×list\\_f1 + 0.10×json\\_valid"
        f" &nbsp;|&nbsp; G2 = 0.45×bert\\_f1 + 0.30×rouge\\_l + 0.15×token\\_f1 + 0.10×rouge\\_1"
        f" &nbsp;|&nbsp; {judge_note}",
        "",
    ]

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

    lines += ["## Per-Field Accuracy", ""]
    all_fields = sorted({k for m in results.values() for k in m.get("per_field_accuracy", {})})
    lines.append("| Field | " + " | ".join(short) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(short)) + " |")
    for field in all_fields:
        row = [field]
        for label in labels:
            row.append(_fmt(results[label].get("per_field_accuracy", {}).get(field)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    all_num = sorted({k for m in results.values() for k in m.get("numeric_mae", {})})
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

    if with_llm_judge:
        lines += [f"## LLM Judge (`{OLLAMA_LOCAL_JUDGE_MODEL}` via local Ollama)", ""]
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
        description="Evaluate a FilterExtractor checkpoint on Kaggle using local Ollama."
    )

    # Model source (mutually exclusive)
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--drive_folder_id", default=None,
        help="Google Drive *folder* ID containing the checkpoint (downloaded via gdown)",
    )
    src.add_argument(
        "--drive_file_id", default=None,
        help="Google Drive *file* ID of a zip archive containing the checkpoint",
    )
    src.add_argument(
        "--model_path", default=None,
        help="Path to a checkpoint already on disk (skips Drive download)",
    )

    # Dataset source
    p.add_argument(
        "--data_drive_file_id", default=None,
        help="Google Drive file ID of the evaluation .jsonl dataset "
             "(downloads to /kaggle/working/eval_data/dataset.jsonl)",
    )
    p.add_argument(
        "--data", default=str(_DATASET),
        help="Local path to the evaluation .jsonl dataset "
             "(ignored when --data_drive_file_id is given)",
    )

    # Evaluation options
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap test samples for quick sanity checks",
    )
    p.add_argument(
        "--no_judge", action="store_true",
        help="Skip the LLM-as-judge step (faster; uses only structural + NLP metrics)",
    )
    p.add_argument(
        "--judge_model", default=OLLAMA_LOCAL_JUDGE_MODEL,
        help=f"Ollama model to use as judge (default: {OLLAMA_LOCAL_JUDGE_MODEL})",
    )
    p.add_argument(
        "--output", default="/kaggle/working/evaluation_results.md",
        help="Output markdown report path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with_llm_judge = not args.no_judge

    # --- Environment setup ---
    setup_kaggle_env()
    if with_llm_judge:
        ensure_ollama_running()
        pull_judge_model(args.judge_model)

    # --- Resolve model checkpoint ---
    if args.model_path:
        ckpt_path = Path(args.model_path)
    elif args.drive_folder_id:
        ckpt_path = download_from_drive_folder(args.drive_folder_id, _DEFAULT_CKPT_DIR)
    elif args.drive_file_id:
        archive = download_from_drive_file(
            args.drive_file_id, _DEFAULT_CKPT_DIR, filename="checkpoint.zip"
        )
        ckpt_path = maybe_unzip(archive)
    else:
        p = argparse.ArgumentParser()
        p.error(
            "Provide one of: --model_path, --drive_folder_id, or --drive_file_id"
        )

    print(f"Checkpoint: {ckpt_path}")

    # --- Resolve dataset ---
    if args.data_drive_file_id:
        data_path = download_from_drive_file(
            args.data_drive_file_id, _DEFAULT_DATA_DIR, filename="dataset.jsonl"
        )
    else:
        data_path = Path(args.data)
        if not data_path.exists():
            sys.exit(
                f"Dataset not found at {data_path}.\n"
                "Pass --data_drive_file_id to download it from Google Drive, "
                "or --data <local_path> if it is already on disk."
            )

    # --- Load test split ---
    print(f"Loading test split from: {data_path}")
    test_data = load_test_split(data_path, max_samples=args.max_samples)
    print(f"Test samples: {len(test_data)}")

    if with_llm_judge:
        print(f"LLM judge: {args.judge_model} (local Ollama at {OLLAMA_LOCAL_BASE_URL})")

    # --- Evaluate ---
    print(f"\n{'=' * 60}\nEvaluating: {ckpt_path}\n{'=' * 60}")
    metrics = evaluate_checkpoint(
        ckpt_path, test_data,
        with_llm_judge=with_llm_judge,
        judge_model=args.judge_model,
    )

    results = {str(ckpt_path): metrics}

    print(
        f"  key_f1={metrics['key_f1']:.3f}  "
        f"value_match={metrics['value_match_rate']:.3f}  "
        f"bert_f1={metrics['bert_score_f1']:.3f}  "
        f"g1={metrics['g1_score']:.3f}  g2={metrics['g2_score']:.3f}  "
        f"overall={metrics['overall_score']:.3f}"
    )

    print_comparison_table(results)
    build_markdown_report(results, len(test_data), with_llm_judge, Path(args.output))


if __name__ == "__main__":
    main()

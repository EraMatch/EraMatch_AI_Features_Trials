"""
Run after the two focused Qwen pending shards complete.
Merges all shard files, fuses into NuExtract3 full-train,
evaluates fused result, and runs 5-fold OOF calibration.

Usage:
    uv run python scripts/post_qwen_merge_fuse.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ROUTER = ROOT / "artifacts" / "trials" / "router"
TRAIN_NUE_RESULTS = (
    ROOT
    / "artifacts"
    / "trials"
    / "router"
    / "train-nuextract3-evidence-ingested"
    / "nuextract3-nuextract3_visual-b16fdd6e13"
    / "results.jsonl"
)


def merge_qwen_shards() -> Path:
    req_path = ROUTER / "train.qwen-focused.requests.jsonl"
    req_ids = [json.loads(x)["cv_id"] for x in req_path.read_text().splitlines()]

    merged: dict[str, dict] = {}
    for fname in [
        "train.qwen-focused.responses.jsonl",
        "train.qwen-pending-0.responses.jsonl",
        "train.qwen-pending-1.responses.jsonl",
    ]:
        p = ROUTER / fname
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            merged[r["cv_id"]] = r

    missing = [cv_id for cv_id in req_ids if cv_id not in merged]
    if missing:
        print(f"ERROR: {len(missing)} focused requests still missing responses.")
        print("  First missing:", missing[:5])
        sys.exit(1)

    out = ROUTER / "train.qwen-focused.all.responses.jsonl"
    out.write_text("".join(json.dumps(merged[cv_id]) + "\n" for cv_id in req_ids))
    print(f"merged {len(merged)} focused Qwen responses -> {out}")
    return out


def fuse_and_evaluate(qwen_all: Path) -> Path:
    fused_out = ROUTER / "train.focused-fusion.all.responses.jsonl"
    cmd = [
        "uv", "run", "eraparse", "trials", "router-fuse-focused",
        "--primary-results", str(TRAIN_NUE_RESULTS),
        "--focused-requests", str(ROUTER / "train.qwen-focused.requests.jsonl"),
        "--specialist-responses", str(qwen_all),
        "--output", str(fused_out),
        "--json",
    ]
    print("fusing ->", fused_out)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FUSE FAILED:", result.stderr[-2000:])
        sys.exit(1)
    print(result.stdout.strip())

    ingested_dir = ROUTER / "train-focused-fusion-ingested"
    eval_cmd = [
        "uv", "run", "eraparse", "trials", "ingest-nuextract3",
        "--requests", str(ROUTER / "train.nuextract3-full.requests.jsonl"),
        "--responses", str(fused_out),
        "--output-dir", str(ingested_dir),
        "--run-db", str(ROUTER / "train-focused-fusion.duckdb"),
        "--model-id", "eraparse/focused-selective-router",
        "--revision", "train-calibrated-v1",
        "--repair-work-records",
        "--full-schema",
        "--json",
    ]
    print("evaluating full-train fused ->", ingested_dir)
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("EVAL FAILED:", result.stderr[-2000:])
        sys.exit(1)
    summary = json.loads(result.stdout)
    agg = summary.get("aggregate", {})
    print(
        f"full-train fused: macro={agg.get('macro_score', '?'):.6f}"
        f"  schema_valid={agg.get('schema_valid_rate', '?')}"
        f"  unsupported={agg.get('unsupported_evidence_rate', '?'):.4f}"
    )
    return Path(summary["results_path"])


def run_calibration(primary_results: Path, specialist_results: Path) -> None:
    from eraparse.io import atomic_write_json
    from eraparse.router_calibration import (
        extract_primary_only_features,
        extract_training_labels,
        summarize_train_oof_calibration,
    )
    from eraparse.trials import read_rows

    primary = read_rows(primary_results)
    specialist = read_rows(specialist_results)
    fields = ["work_experience", "certifications"]

    features = extract_primary_only_features(primary, fields=fields)
    labels = extract_training_labels(primary, specialist, fields=fields)
    report = summarize_train_oof_calibration(features, labels, partition="train_oof")

    out = ROUTER / "train.calibration-report.json"
    atomic_write_json(out, report)
    print(f"calibration report -> {out}")
    print("row_count:", report["row_count"])
    print("fold_outcomes:", report["fold_outcomes"])
    for field in fields:
        top = sorted(
            [
                c
                for c in report["candidate_thresholds"]
                if c["field"] == field
                and c["coverage"] <= 0.45
                and c["win_precision"] >= 0.6
            ],
            key=lambda x: (x["wins"] - x["losses"], x["win_precision"]),
            reverse=True,
        )[:3]
        print(f"\n{field} top thresholds:")
        for t in top:
            print(
                f"  {t['feature']} {t['direction']} {t['threshold']}"
                f"  cov={t['coverage']:.3f}  prec={t['win_precision']:.3f}"
                f"  wins={t['wins']}  losses={t['losses']}"
            )


if __name__ == "__main__":
    print("=== Step 1: merge Qwen shards ===")
    qwen_all = merge_qwen_shards()

    print("\n=== Step 2: fuse and evaluate full-train ===")
    fused_results = fuse_and_evaluate(qwen_all)

    print("\n=== Step 3: 5-fold OOF calibration ===")
    run_calibration(TRAIN_NUE_RESULTS, fused_results)

    print("\nDone. Next: run NuExtract3 on validation split (310 CVs).")
    print("  uv run --group modal modal run modal_apps/nuextract3_vllm_trial.py \\")
    print(
        "    --requests-path "
        "artifacts/trials/router/validation.nuextract3-full.requests.jsonl \\"
    )
    print("    --dataset-root ../eramatch_benchmark_v4 \\")
    print(
        "    --output-path "
        "artifacts/trials/router/validation.nuextract3-mtp-evidence.responses.jsonl \\"
    )
    print("    --use-mtp --include-evidence-text --max-records 310 \\")
    print("    --chunk-size 20 --max-new-tokens 900")

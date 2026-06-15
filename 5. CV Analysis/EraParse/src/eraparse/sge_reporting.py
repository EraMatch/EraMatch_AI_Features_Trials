from __future__ import annotations

from pathlib import Path
from typing import Any

from eraparse.io import atomic_write_json, atomic_write_text, read_json


def _metric(evaluation: Any, key: str) -> float | None:
    if not isinstance(evaluation, dict):
        return None
    value = evaluation.get(key)
    return float(value) if isinstance(value, int | float) else None


def collect_trial_summaries(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("summary.json")):
        summary = read_json(path)
        if not isinstance(summary, dict) or "mode" not in summary:
            continue
        training = summary.get("training_evaluation")
        validation = summary.get("validation_evaluation")
        rows.append(
            {
                "run": str(path.parent.relative_to(root)),
                "mode": summary.get("mode"),
                "records_path": summary.get("records_path"),
                "steps": summary.get("steps"),
                "completed_steps": summary.get("completed_steps", summary.get("steps")),
                "unfreeze_final_layers": summary.get("unfreeze_final_layers"),
                "query_layers": summary.get("query_layers"),
                "loss_weights": summary.get("loss_weights"),
                "primary_decoder": summary.get("primary_decoder"),
                "training_macro": _metric(training, "macro_score"),
                "validation_macro": _metric(validation, "macro_score"),
                "training_schema_valid": _metric(training, "schema_valid_rate"),
                "validation_schema_valid": _metric(validation, "schema_valid_rate"),
                "training_unsupported": _metric(training, "unsupported_evidence_rate"),
                "validation_unsupported": _metric(validation, "unsupported_evidence_rate"),
                "runtime_seconds": summary.get("runtime_seconds"),
                "training_seconds": summary.get("training_seconds"),
                "evaluation_seconds": summary.get("evaluation_seconds"),
                "source_summary": str(path),
            }
        )
    return rows


def _format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def render_trial_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SG-ESE Trial Comparison",
        "",
        "| Run | Mode | Steps | Decoder | Train macro | Validation macro | "
        "Validation schema | Runtime (s) |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {mode} | {steps} | {decoder} | {train} | {validation} | "
            "{schema} | {runtime} |".format(
                run=row["run"],
                mode=row["mode"],
                steps=row["completed_steps"] or "",
                decoder=row["primary_decoder"] or "",
                train=_format_metric(row["training_macro"]),
                validation=_format_metric(row["validation_macro"]),
                schema=_format_metric(row["validation_schema_valid"]),
                runtime=_format_metric(row["runtime_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "Training-only scores are diagnostics and are not held-out evidence. "
            "Use validation scores for model selection.",
            "",
        ]
    )
    return "\n".join(lines)


def write_trial_report(root: Path, output: Path) -> dict[str, Any]:
    rows = collect_trial_summaries(root)
    payload = {"root": str(root), "trial_count": len(rows), "trials": rows}
    atomic_write_json(output, payload)
    atomic_write_text(output.with_suffix(".md"), render_trial_markdown(rows))
    return payload

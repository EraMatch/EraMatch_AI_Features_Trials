"""Generate standard thesis analysis plots from the run store.

Produces after every eval:
  artifacts/analysis/<run_id>/field_breakdown.png   per-field F1 bar chart
  artifacts/analysis/pareto_latest.png              accuracy vs latency scatter
  artifacts/analysis/faithfulness_latest.png        hallucination rate bars
  artifacts/analysis/<run_id>/loss_curve.png        training loss (if available)
  artifacts/analysis/<run_id>/field_breakdown.csv   underlying data

Run:
    uv run python scripts/plot_results.py                  # all plots from run store
    uv run python scripts/plot_results.py --run-id <id>    # single run breakdown only
"""
import argparse
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "artifacts" / "runs.duckdb"
ANALYSIS_DIR = ROOT / "artifacts" / "analysis"

FIELD_ORDER = [
    "full_name", "email", "location", "phone",
    "linkedin_url", "github_url", "summary", "skills",
    "work_experience", "education", "projects", "certifications",
]

MODEL_COLORS = {
    "nuextract3": "#4C72B0",
    "router":     "#DD8452",
    "qwen3-4b":   "#55A868",
    "gemma3-1b":  "#C44E52",
    "nuextract-tiny": "#8172B3",
    "nuextract-15":   "#937860",
    "smolvlm2":   "#DA8BC3",
}

KNOWN_BASELINES = {
    "NuExtract3 (B)":    {"clean_macro": 0.9402, "latency_p50": 1.29,  "unsupported": 0.036},
    "Router (C)":        {"clean_macro": 0.9510, "latency_p50": 10.35, "unsupported": 0.039},
    "Qwen3-4B (A)":      {"clean_macro": 0.8921, "latency_p50": 15.83, "unsupported": 0.006},
}


def _color(model_id: str) -> str:
    for key, col in MODEL_COLORS.items():
        if key in model_id.lower():
            return col
    return "#999999"


# =========================================================
# Field breakdown bar chart
# =========================================================

def plot_field_breakdown(run_id: str, field_scores: dict[str, float], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [f for f in FIELD_ORDER if f in field_scores]
    scores = [field_scores[f] for f in fields]

    df = pd.DataFrame({"field": fields, "f1": scores})
    df.to_csv(out_dir / "field_breakdown.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0" if s >= 0.90 else "#DD8452" if s >= 0.80 else "#C44E52" for s in scores]
    bars = ax.barh(fields, scores, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.90, color="gray", linestyle="--", linewidth=0.8, label="0.90 target")
    ax.set_xlabel("Clean F1")
    ax.set_title(f"Field-level breakdown — {run_id}", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "field_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir / 'field_breakdown.png'}")


# =========================================================
# Pareto scatter: accuracy vs latency
# =========================================================

def plot_pareto(runs: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot known baselines first (hollow markers)
    for label, b in KNOWN_BASELINES.items():
        ax.scatter(
            b["latency_p50"], b["clean_macro"],
            s=120, marker="D", color="gray", edgecolors="black", linewidths=1.0, zorder=3
        )
        ax.annotate(label, (b["latency_p50"], b["clean_macro"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="gray")

    # Plot new runs
    for r in runs:
        lat = r.get("latency_p50") or r.get("latency_mean")
        acc = r.get("clean_macro")
        if lat is None or acc is None:
            continue
        col = _color(r.get("model_id", ""))
        ax.scatter(lat, acc, s=100, color=col, edgecolors="white", linewidths=0.5, zorder=4)
        ax.annotate(
            r.get("label") or r.get("model_id", "")[:18],
            (lat, acc), textcoords="offset points", xytext=(6, 4), fontsize=8
        )

    ax.set_xlabel("Latency p50 (s/CV)", fontsize=11)
    ax.set_ylabel("Clean Macro F1", fontsize=11)
    ax.set_title("Accuracy × Latency Pareto — all runs", fontsize=12)
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


# =========================================================
# Faithfulness comparison bar chart
# =========================================================

def plot_faithfulness(runs: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(KNOWN_BASELINES.items())
    labels = [r[0] for r in rows]
    values = [r[1]["unsupported"] for r in rows]
    colors = ["gray"] * len(labels)

    for r in runs:
        v = r.get("unsupported_evidence_rate")
        if v is None:
            continue
        labels.append(r.get("label") or r.get("model_id", "")[:18])
        values.append(v)
        colors.append(_color(r.get("model_id", "")))

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.axhline(0.01, color="green", linestyle="--", linewidth=0.8, label="1% target")
    ax.set_ylabel("Unsupported evidence rate (↓ better)")
    ax.set_title("Faithfulness (hallucination rate) — all models")
    ax.set_ylim(0, max(values) * 1.3 + 0.01)
    ax.legend(fontsize=8)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


# =========================================================
# Training loss curve
# =========================================================

def plot_loss_curve(loss_steps: list[tuple[int, float]], run_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, losses = zip(*loss_steps)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, color="#4C72B0", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"Training loss — {run_id}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"step": steps, "loss": losses}).to_csv(out_dir / "loss_curve.csv", index=False)
    print(f"saved {out_dir / 'loss_curve.png'}")


# =========================================================
# Load from run store
# =========================================================

def load_runs_from_db(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT run_id, model_id, clean_macro, raw_macro, nested_macro,
                   unsupported_evidence_rate, latency_p50, latency_p95,
                   cost_usd, split, schema, constrained, notes
            FROM runs
            WHERE status = 'completed'
            ORDER BY clean_macro DESC NULLS LAST
        """).fetchall()
        cols = ["run_id", "model_id", "clean_macro", "raw_macro", "nested_macro",
                "unsupported_evidence_rate", "latency_p50", "latency_p95",
                "cost_usd", "split", "schema", "constrained", "notes"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []
    finally:
        con.close()


def load_field_scores_from_db(run_id: str, db_path: Path) -> dict[str, float]:
    if not db_path.exists():
        return {}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT field_name, mean_f1
            FROM field_scores
            WHERE run_id = ?
        """, [run_id]).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}
    finally:
        con.close()


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", help="Plot breakdown for a specific run only")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--runs-json", help="JSON file with run list (fallback if no DB)")
    args = ap.parse_args()

    db = Path(args.db)
    runs = load_runs_from_db(db)

    # Fallback: load from a manually-written JSON list
    if not runs and args.runs_json:
        runs = json.loads(Path(args.runs_json).read_text())

    if args.run_id:
        field_scores = load_field_scores_from_db(args.run_id, db)
        if field_scores:
            out_dir = ANALYSIS_DIR / args.run_id
            plot_field_breakdown(args.run_id, field_scores, out_dir)
        else:
            print(f"no field scores in DB for run {args.run_id}")
    else:
        # Full suite
        if runs:
            plot_pareto(runs, ANALYSIS_DIR / "pareto_latest.png")
            plot_faithfulness(runs, ANALYSIS_DIR / "faithfulness_latest.png")
            for r in runs:
                fs = load_field_scores_from_db(r["run_id"], db)
                if fs:
                    plot_field_breakdown(r["run_id"], fs, ANALYSIS_DIR / r["run_id"])
        else:
            # No DB yet — plot baselines only as a starting point
            print("no completed runs in DB yet — plotting baseline Pareto only")
            plot_pareto([], ANALYSIS_DIR / "pareto_latest.png")
            plot_faithfulness([], ANALYSIS_DIR / "faithfulness_latest.png")


if __name__ == "__main__":
    main()

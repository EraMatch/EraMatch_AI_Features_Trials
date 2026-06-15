#!/usr/bin/env bash
# One-command eval pipeline for any fine-tuned model output.
#
# Usage:
#   bash scripts/eval_ft.sh <adapter_name> <requests_jsonl> <responses_jsonl> <run_label>
#
# Examples:
#   bash scripts/eval_ft.sh nuextract-15-reduced \
#       artifacts/trials/router/validation.nuextract-tiny-ft.requests.jsonl \
#       artifacts/trials/ft/validation.nuextract-15-ft.jsonl \
#       nuextract-15-ft-v1
#
#   bash scripts/eval_ft.sh gemma3-1b-reduced \
#       artifacts/trials/router/validation.gemma-ft.requests.jsonl \
#       artifacts/trials/ft/validation.gemma3-1b-ft.jsonl \
#       gemma3-1b-ft-v1
#
# Steps:
#   1. eraparse trials ingest-mapper   → structured JSON per CV
#   2. analyze_labels_and_router.py    → clean_macro F1 + per-field scores
#   3. plot_results.py                 → field breakdown PNG + updated Pareto

set -euo pipefail

ADAPTER_NAME="${1:?Usage: eval_ft.sh <adapter_name> <requests_jsonl> <responses_jsonl> <run_label>}"
REQUESTS="${2:?}"
RESPONSES="${3:?}"
LABEL="${4:?}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INGEST_OUT="$ROOT/artifacts/ingested/$LABEL"

echo "=== eval_ft: $LABEL ==="
echo "adapter   : $ADAPTER_NAME"
echo "requests  : $REQUESTS"
echo "responses : $RESPONSES"
echo "output    : $INGEST_OUT"

mkdir -p "$INGEST_OUT"

# ------------------------------------------------------------------
# 1. Ingest-mapper: raw model outputs → structured per-CV JSON
# ------------------------------------------------------------------
echo ""
echo "[1/3] running ingest-mapper..."
uv run eraparse trials ingest-mapper \
    --model-id "eraparse/$ADAPTER_NAME" \
    --revision ft-v1 \
    --representation pymupdf4llm_markdown \
    --requests "$REQUESTS" \
    --responses "$RESPONSES" \
    --output-dir "$INGEST_OUT" \
    --allow-partial

# ------------------------------------------------------------------
# 2. Compute fully-clean metric
# ------------------------------------------------------------------
echo ""
echo "[2/3] computing clean macro F1..."
uv run python -c "
import json, sys
from pathlib import Path
sys.path.insert(0, '$ROOT/scripts')
import analyze_labels_and_router as A
man = A.load_manifest('validation')
res = next(Path('$INGEST_OUT').rglob('results.jsonl'))
A.clean_macro(res, man, '$LABEL')
"

# ------------------------------------------------------------------
# 3. Plots
# ------------------------------------------------------------------
echo ""
echo "[3/3] generating plots..."
uv run python "$ROOT/scripts/plot_results.py" || echo "plot_results.py skipped (optional)"

echo ""
echo "=== DONE: $LABEL ==="
echo "ingested : $INGEST_OUT"

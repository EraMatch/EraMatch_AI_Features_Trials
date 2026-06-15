#!/usr/bin/env bash
# Eval pipeline for vision fine-tuned model output (Track C SmolVLM2).
#
# Usage:
#   bash scripts/eval_ft_vision.sh <adapter_name> <requests_jsonl> <responses_jsonl> <run_label>
#
# Example:
#   bash scripts/eval_ft_vision.sh smolvlm2-cv-reduced \
#       artifacts/trials/ft/validation.smolvlm2-ft.requests.jsonl \
#       artifacts/trials/ft/validation.smolvlm2-ft.jsonl \
#       smolvlm2-ft-v1

set -euo pipefail

ADAPTER_NAME="${1:?Usage: eval_ft_vision.sh <adapter_name> <requests_jsonl> <responses_jsonl> <run_label>}"
REQUESTS="${2:?}"
RESPONSES="${3:?}"
LABEL="${4:?}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INGEST_OUT="$ROOT/artifacts/ingested/$LABEL"

echo "=== eval_ft_vision: $LABEL ==="
echo "adapter   : $ADAPTER_NAME"
echo "requests  : $REQUESTS"
echo "responses : $RESPONSES"
echo "output    : $INGEST_OUT"

mkdir -p "$INGEST_OUT"

echo ""
echo "[1/3] running ingest-mapper (vision / pymupdf4llm_markdown evidence)..."
uv run eraparse trials ingest-mapper \
    --model-id "eraparse/$ADAPTER_NAME" \
    --revision ft-v1 \
    --representation pymupdf4llm_markdown \
    --requests "$REQUESTS" \
    --responses "$RESPONSES" \
    --output-dir "$INGEST_OUT" \
    --allow-partial

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

echo ""
echo "[3/3] generating plots..."
uv run python "$ROOT/scripts/plot_results.py" || echo "plot_results.py skipped (optional)"

echo ""
echo "=== DONE: $LABEL ==="
echo "ingested : $INGEST_OUT"

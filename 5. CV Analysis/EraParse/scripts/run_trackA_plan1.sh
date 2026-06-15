#!/usr/bin/env bash
# Track A Plan 1 — end-to-end: upload SFT data → fine-tune on Modal → infer → eval
# Run from repo root: bash scripts/run_trackA_plan1.sh
# Safe to re-run: training skips if adapter exists; inference resumes from where it left off.
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER_NAME="gemma3-1b-reduced"
BASE_MODEL="unsloth/gemma-3-1b-it"
REQUESTS="artifacts/trials/router/validation.gemma-ft.requests.jsonl"
RESPONSES="artifacts/trials/router/validation.gemma-ft.responses.jsonl"
INGEST_OUT="artifacts/trials/router/validation-gemma-ft-ingested"
TARGET=310

echo "=== STEP 1: upload SFT train data to Modal volume ==="
uv run modal volume put eraparse-adapters \
    artifacts/sft/train.reduced.sft.jsonl \
    /sft/train.reduced.sft.jsonl || true
echo "upload done (or already existed)"

echo ""
echo "=== STEP 2: fine-tune Gemma-3-1B on Modal L4 GPU (~35 min) ==="
echo "    adapter will be saved to volume: eraparse-adapters/$ADAPTER_NAME"
uv run modal run modal_apps/gemma_finetune.py \
    --model-name "$BASE_MODEL" \
    --out-name "$ADAPTER_NAME"
echo "training done"

echo ""
echo "=== STEP 3: verify adapter saved ==="
uv run modal volume ls eraparse-adapters "$ADAPTER_NAME"

echo ""
echo "=== STEP 4: run inference on 310 validation CVs (resume-loop) ==="
DONE=0
while [ "$DONE" -lt "$TARGET" ]; do
    uv run modal run modal_apps/gemma_adapter_infer.py \
        --requests-path "$REQUESTS" \
        --output-path "$RESPONSES" \
        --base-model "$BASE_MODEL" \
        --adapter-name "$ADAPTER_NAME" \
        --chunk-size 25 || true
    DONE=$(wc -l < "$RESPONSES" 2>/dev/null || echo 0)
    echo "progress: $DONE/$TARGET responses"
    if [ "$DONE" -lt "$TARGET" ]; then
        echo "retrying in 10s..."
        sleep 10
    fi
done
echo "inference done: $DONE/$TARGET"

echo ""
echo "=== STEP 5: evaluate via ingest-mapper ==="
uv run eraparse trials ingest-mapper \
    --model-id "eraparse/$ADAPTER_NAME" \
    --revision ft-v1 \
    --representation pymupdf4llm_markdown \
    --requests "$REQUESTS" \
    --responses "$RESPONSES" \
    --output-dir "$INGEST_OUT" \
    --allow-partial

echo ""
echo "=== STEP 6: compute fully-clean metric ==="
uv run python -c "
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import analyze_labels_and_router as A
man = A.load_manifest('validation')
res = next(Path('$INGEST_OUT').rglob('results.jsonl'))
A.clean_macro(res, man, 'Gemma3-1B fine-tuned (validation)')
"

echo ""
echo "=== ALL DONE — results in $INGEST_OUT ==="

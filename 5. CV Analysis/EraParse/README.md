# EraParse

EraParse is a documentation-first research project for evaluating CV parsing
pipelines, structured extraction models, document VLMs, and visual-token
compression.

Implementation agents must read [AGENTS.md](AGENTS.md) and the required
documents listed there before changing trial behavior.

## Research Goal

Determine how document representation, model architecture, and visual-token
compression affect accurate, evidence-supported CV-to-JSON extraction.

The main experiment corpus is a deterministic, stratified half of the completed
EraMatch V4 dataset:

- Source dataset: `../eramatch_benchmark_v4`
- Completed samples: 4,950
- Working experiment corpus: 2,475
- Locked confirmation corpus: 2,475
- Source dataset policy: read-only

## Repository Status

- Documentation and research protocol: ready
- Dataset audit and deterministic manifest generator: implemented
- Evaluator and run database: implemented
- Trial 2A parser-input ablation: debug-50 completed
- Trial 2B mapper comparison: frozen Qwen3 lane evaluated on full validation,
  ID, and template-OOD splits
- ATS deterministic baseline: completed on full 310 ID and 410 OOD test pools
- ATS prediction comparison: completed for the frozen Qwen3 lane
- Modal CPU parser and A10 mapper jobs: implemented
- Trial 3 OCR-recovery baseline: selected on full validation
- Trial 3 Donut-base adaptation: completed; decoder/schema gate failed, SG-VTC
  blocked
- PyMuPDF4LLM smart-OCR evidence graph and deterministic assembler: implemented
- LayoutLMv3 and SG-ESE local MPS architecture smokes: passed
- Full practical PyMuPDF4LLM train/validation evidence corpus: prepared
- Standard LayoutLMv3 two-epoch held-out validation macro: 0.739158
- SG-ESE query-only two-epoch held-out validation macro: 0.738284
- Best practical modular held-out validation macro: 0.776740
- NuExtract3 + deterministic contract assembly held-out validation macro:
  0.881002 with 100% JSON and schema validity
- SG-ESE-stage Modal GPU smoke spend: approximately $0.0024; current held-out
  training runs use local M1 Pro/MPS
- Locked confirmation corpus: untouched

Current accuracy, speed, and cost comparisons are tracked in
[docs/RESULTS_AND_PARETO.md](docs/RESULTS_AND_PARETO.md).

## Local Development

```bash
uv sync --python 3.11
uv run pytest
uv run ruff check .
uv run mypy src

uv run eraparse data audit --dataset-root ../eramatch_benchmark_v4
uv run eraparse data build-manifests --dataset-root ../eramatch_benchmark_v4
uv run eraparse data validate-manifests
uv run eraparse runs init

# Run the persisted deterministic ATS baseline on the real protected test sizes.
uv run eraparse ats run-baseline --json

# Compare frozen structured predictions against the same ATS contract.
uv run eraparse ats compare-predictions \
  --lane qwen3_pymupdf_frozen_v1 \
  --id-results PATH_TO_ID_RESULTS \
  --ood-results PATH_TO_OOD_RESULTS \
  --json

# Prepare a NuExtract debug-50 representation bundle.
uv run eraparse trials prepare-nuextract --representation pymupdf_text --json

# Generate Docling and PyMuPDF4LLM representations on Modal CPU.
uv run --group modal modal run modal_apps/docling_representations.py \
  --manifest-path artifacts/manifests/debug_50.jsonl \
  --dataset-root ../eramatch_benchmark_v4 \
  --output-root artifacts/representations

# Run the prepared bundle on Modal GPU.
uv run --group modal modal run modal_apps/nuextract_trial.py \
  --requests-path artifacts/trials/nuextract/pymupdf_text/requests.jsonl \
  --output-path artifacts/trials/nuextract/pymupdf_text/responses.jsonl

# Prepare direct visual NuExtract3 requests.
uv run eraparse trials prepare-nuextract3 \
  --manifest artifacts/manifests/debug_50.jsonl \
  --output artifacts/trials/nuextract3/debug_50.jsonl \
  --json

# Run NuExtract3 on page images with the exact reduced schema template.
uv run --group modal modal run modal_apps/nuextract3_trial.py \
  --requests-path artifacts/trials/nuextract3/debug_50.jsonl \
  --dataset-root ../eramatch_benchmark_v4 \
  --output-path artifacts/trials/nuextract3/debug_50.responses.jsonl

# Evaluate and record the NuExtract3 direct visual responses.
uv run eraparse trials ingest-nuextract3 \
  --requests artifacts/trials/nuextract3/debug_50.jsonl \
  --responses artifacts/trials/nuextract3/debug_50.responses.jsonl \
  --json

# Prepare PaddleOCR-VL page-parser requests.
uv run eraparse trials prepare-paddleocr-vl \
  --manifest artifacts/manifests/debug_50.jsonl \
  --output artifacts/trials/paddleocr_vl/debug_50.jsonl \
  --json

# Run PaddleOCR-VL as a parser lane, then materialize its markdown/json outputs.
uv run --group modal modal run modal_apps/paddleocr_vl_trial.py \
  --requests-path artifacts/trials/paddleocr_vl/debug_50.jsonl \
  --dataset-root ../eramatch_benchmark_v4 \
  --output-path artifacts/trials/paddleocr_vl/debug_50.responses.jsonl
uv run eraparse trials materialize-paddleocr-vl \
  --responses artifacts/trials/paddleocr_vl/debug_50.responses.jsonl \
  --json

# Feed PaddleOCR-VL markdown into the frozen Qwen3 mapper lane.
uv run eraparse trials prepare-nuextract \
  --representation paddleocr_vl_markdown \
  --manifest artifacts/manifests/debug_50.jsonl \
  --json

# Evaluate and record the Modal responses.
uv run eraparse trials ingest-nuextract \
  --representation pymupdf_text \
  --requests artifacts/trials/nuextract/pymupdf_text/requests.jsonl \
  --responses artifacts/trials/nuextract/pymupdf_text/responses.jsonl \
  --json

# Prepare exact Donut train/validation records from the audited manifests.
uv run eraparse trials prepare-donut \
  --manifest artifacts/manifests/train.jsonl \
  --output artifacts/trials/donut/train.jsonl \
  --target-format native_tokens \
  --json

# Run the corrected full-token Donut fine-tune on Modal.
uv run --group modal modal run modal_apps/donut_train.py \
  --train-records artifacts/trials/donut/train.jsonl \
  --validation-records artifacts/trials/donut/validation.jsonl \
  --dataset-root ../eramatch_benchmark_v4 \
  --run-name donut-native-v3-full-train-v1 \
  --epochs 3 \
  --max-steps 0 \
  --gradient-accumulation-steps 8
```

Generated manifests, reports, and databases are written under ignored
`artifacts/`.

## Required Reading

1. [AGENTS.md](AGENTS.md)
2. [Project Plan](docs/PROJECT_PLAN.md)
3. [Dataset Contract](docs/DATASET_CONTRACT.md)
4. [Evaluation Protocol](docs/EVALUATION_PROTOCOL.md)
5. [Trial Runbook](docs/TRIAL_RUNBOOK.md)
6. [Model Catalog](docs/MODEL_CATALOG.md)
7. [Versions and Environments](docs/VERSIONS_AND_ENVIRONMENTS.md)
8. [Modal Execution](docs/MODAL_EXECUTION.md)
9. [Initial Trial Results](docs/TRIAL_2_INITIAL_RESULTS.md)
10. [ATS Baseline](docs/ATS_BASELINE.md)
11. [ATS Baseline Results](docs/ATS_BASELINE_RESULTS.md)
12. [Second Contribution Plan](docs/SECOND_CONTRIBUTION_PLAN.md)
13. [Frozen Qwen3 Final Results](docs/QWEN3_FROZEN_RESULTS.md)
14. [Baseline Matrix](docs/BASELINE_MATRIX.md)
15. [Trial 3 Direct Vision](docs/TRIAL_3_DIRECT_VISION.md)
15. [Next Model Action Plan](docs/NEXT_MODEL_ACTION_PLAN.md)
16. [Modal Cost Audit](docs/MODAL_COST_AUDIT.md)
17. [Re-Architected Research Flow](docs/REARCHITECTED_RESEARCH_FLOW.md)
18. [SG-ESE Implementation And Trial Log](docs/SG_ESE_STAGE_LOG.md)
19. [Current Next Model Action Plan](docs/NEXT_MODEL_ACTION_PLAN.md)

Model, library, and paper information was checked against live sources on
**2026-06-09**. Re-verify volatile APIs and model revisions before implementing
or running a trial.

# Phase 1 Finetunes

First attempt at training a small LLM to extract structured CV data.

## `finetune_cv_mapper_gemma3_1b.ipynb`

QLoRA fine-tune of `unsloth/gemma-3-1b-it` on rich CVSchema.

- **Data**: 610 train / 128 test CVs (v3 dataset)
- **Platform**: Kaggle T4
- **Results**: Base zero-shot 31.2% schema-valid. Fine-tuned results computed in-notebook but adapter not exported.
- **Issue**: `max_seq_length=2048` vs ~4,270 token system prompt (heavy truncation)

## `finetune-cv-mapper-gemma3-1b-kaggle.md`

Implementation plan: 19 tasks across 5 waves.

## Benchmark Results

Pre-finetune extraction benchmark (1,000 CV sample):
- `results__n1000_all__gemma3_1b__pymupdf_raw.csv`
- `results__n1000_all__gemma3_1b__pymupdf4llm.csv`
- `sample__n1000_all.csv` — sample manifest
- `extractions__n1000_all__summary.csv` — quality summary
- `extractions__n1000_all.pkl` — extraction cache

## Lessons → Track A

- Switch to reduced CVSchema (fewer fields, shorter prompt)
- Move from Kaggle T4 to Modal Labs (A100/H100)
- Proper adapter export and inference pipeline
- See `EraParse/` for production implementation

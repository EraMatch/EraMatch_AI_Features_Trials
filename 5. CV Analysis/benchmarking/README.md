# CV Parsing Benchmarking

## Notebook

### `cv-parsers-benchmark.ipynb`
Compares PDF extraction + LLM structuring approaches on the v3 10K dataset.

**Extraction methods**: PyMuPDF raw, Docling default, Docling custom (TableFormer ACCURATE + sidebar reorder)

**LLMs**: qwen3.5:cloud (Ollama), qwen3.5:2b (local), Phi-4-mini GGUF (llama-cpp)

**Metrics**: Skill set F1, section-level scoring, extraction quality

## Kaggle

**Notebook**: https://www.kaggle.com/code/anasahmad202202029/cv-parsing-2

**Output**: `kaggle_output/` — 14 benchmark result files across 8 extraction methods (CSV, JSONL, PKL). Downloaded via `kaggle kernels output anasahmad202202029/cv-parsing-2 -p kaggle_output/`.

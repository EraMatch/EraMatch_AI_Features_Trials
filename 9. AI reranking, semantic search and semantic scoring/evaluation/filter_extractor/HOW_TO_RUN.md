# FilterExtractor Evaluation — How to Run

## Prerequisites

```bash
pip install torch datasets tqdm rouge-score bert-score openai
```

Ollama must be running locally if you use `--local_judge`:

```bash
ollama serve
ollama pull qwen2.5-coder:7b   # or whichever model you want as judge
```

---

## Quick reference

| Goal | Command |
|---|---|
| Evaluate all checkpoints, no judge | `python evaluate.py` |
| Evaluate all checkpoints, local judge | `python evaluate.py --local_judge` |
| Evaluate all checkpoints, cloud judge | `python evaluate.py --with_llm_judge` |
| Single checkpoint | `python evaluate.py --model_path <path>` |
| Quick sanity check (50 samples) | `python evaluate.py --max_samples 50` |
| Custom output file | `python evaluate.py --output my_report.md` |

---

## All flags

| Flag | Default | Description |
|---|---|---|
| `--model_path` | *(none)* | Evaluate a single checkpoint instead of all |
| `--models_dir` | `../../Models/FilterExtractor` | Root dir scanned for all checkpoints |
| `--data` | `../../Data generation/deepseek_tech_filters_5k.jsonl` | Ground-truth JSONL dataset |
| `--max_samples` | *(all)* | Cap test samples — useful for quick checks |
| `--with_llm_judge` | off | Enable cloud LLM judge (needs `ollama_api_keys.txt`) |
| `--local_judge` | off | Enable local Ollama judge (no API keys needed) |
| `--judge_model` | *(auto)* | Override judge model name |
| `--output` | `evaluation_results.md` | Output markdown report path |

---

## Common use cases

### 1. Evaluate everything (no judge)
```bash
python evaluate.py
```
Scans all `checkpoint-*` dirs under `Models/FilterExtractor/`, runs inference,
computes structural + NLP metrics, prints a comparison table, and saves
`evaluation_results.md`.

---

### 2. Evaluate with a local Ollama judge
```bash
python evaluate.py --local_judge
```
Hits `http://localhost:11434/v1` with `qwen2.5-coder:7b` as the judge.
No API keys required. Ollama must be running.

Use a different local model:
```bash
python evaluate.py --local_judge --judge_model llama3.1:8b
```

---

### 3. Evaluate with the cloud judge
```bash
python evaluate.py --with_llm_judge
```
Reads API keys from `../../Data generation/ollama_api_keys.txt` (one key per line)
and uses `gpt-oss:120b-cloud` via `https://ollama.com/v1`.

---

### 4. Evaluate a single checkpoint
```bash
python evaluate.py --model_path ../../Models/FilterExtractor/qwen_filter_extractor_Qwen2.5-Coder-1.5B-Instruct-bnb-4bit/checkpoint-948
```

With a local judge on 50 samples:
```bash
python evaluate.py \
  --model_path ../../Models/FilterExtractor/qwen_filter_extractor_Qwen2.5-Coder-1.5B-Instruct-bnb-4bit/checkpoint-948 \
  --local_judge \
  --max_samples 50
```

---

### 5. Custom output path
```bash
python evaluate.py --local_judge --output results/run_01.md
```

---

## Output

**Console** — comparison table printed after all checkpoints finish. `*` marks
the best value per column.

**Markdown report** — saved to `--output` (default `evaluation_results.md`).
Contains:
- Summary table (all metrics side-by-side)
- Per-field accuracy breakdown
- Numeric field MAE (mean absolute error)
- LLM judge section (if judge was enabled)

---

## Scoring formula

```
G1 (structural) = 0.35 × key_f1
                + 0.35 × value_match_rate
                + 0.20 × list_field_f1
                + 0.10 × json_valid_rate

G2 (NLP)        = 0.45 × bert_score_f1
                + 0.30 × rouge_l
                + 0.15 × token_f1
                + 0.10 × rouge_1

Overall (no judge)   = 0.55 × G1 + 0.45 × G2
Overall (with judge) = 0.25 × G1 + 0.25 × G2 + 0.50 × judge_score_normalized
```

LLM judge scores 0–3 per sample; normalised to 0–1 before weighting.
Pass rate = fraction of samples scoring ≥ 2.

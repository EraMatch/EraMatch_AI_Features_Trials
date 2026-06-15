# Prior Kaggle Trials — Approaches, Configs, Results

Documents the author's earlier exploratory Kaggle work that preceded the
`eraparse` rigorous benchmark. Kept for the thesis "prior work / motivation"
section. **Caveat:** these notebooks used exploratory evaluation harnesses with
known limitations; the rigorous numbers are the `eraparse` harness
(`RESULTS_AND_PARETO.md`), not these. They are documented here for the *narrative
of how the problem was found*, not as final results.

---

## Trial 1 — `cv-parsing-2` (parser-comparison benchmark, dataset **v3**)

- **URL:** https://www.kaggle.com/code/anasahmad202202029/cv-parsing-2
- **Dataset:** EraMatch CV dataset **v3** (`anasahmad25/cv-parsing-eramatch`)
- **Goal:** "Phase 1 Extraction Benchmark" — compare many PDF-extraction methods
  while holding the schema mapper fixed, on 1,000 CVs balanced across tier /
  template / domain.

### Approach
Two-stage `parser → fixed mapper`, where the mapper is **`gemma3:1b`** served via
**Ollama** (prompt style `gemma_native_xml`). 9 extractors produced text →
the same 1B mapper structured it into the target schema. Also scaffolded
(optional, mostly disabled on Kaggle): `regex_nuextract` (NuExtract-tiny),
`qwen_vl` (Qwen2.5-VL-3B end-to-end), LayoutLMv1, pyresparser.

### Extractors compared + results (1,000 CVs)

| Extractor | mapper valid% | skills F1 | exp/edu F1 | extract time | total/CV |
|---|---:|---:|---:|---:|---:|
| pymupdf4llm | **100.0%** | **0.416** | ~0.000 | 1.04 s | 4.03 s |
| marker | 99.9% | 0.339 | ~0.000 | 13.22 s | 16.06 s |
| docling_custom | 86.7% | 0.293 | ~0.000 | 0.58 s | 3.11 s |
| docling_default | 86.7% | 0.346 | ~0.000 | 0.37 s | 3.24 s |
| pymupdf_raw | 84.7% | 0.281 | ~0.000 | **0.01 s** | 2.29 s |
| pymupdf_structured | 84.7% | 0.318 | ~0.000 | 1.07 s | 3.48 s |
| unstructured_fast | 84.7% | 0.424 | ~0.000 | 0.18 s | 3.41 s |
| regex_heuristic (pyresparser) | 84.5% | 0.483 | ~0.001 | 0.01 s | 3.13 s |
| regex_ner (spaCy) | 82.5% | **0.513** | ~0.002 | 0.43 s | 3.60 s |

### What is reliable vs not
- **Reliable — parser speed:** `pymupdf_raw` fastest (0.01 s), `pymupdf4llm`
  ~1 s, `docling` 0.4–0.6 s, `marker` slowest (13 s). Total time is
  mapper-dominated (~2–3 s on a 1B Ollama model).
- **Reliable — structural validity:** the 1B mapper *can* emit schema-valid JSON
  82–100% of the time (best with `pymupdf4llm` text, 100%).
- **NOT reliable — nested accuracy:** experience/education/fuzzy F1 are ≈0.000
  across **all** extractors while only `skills` carries signal. This is the key
  limitation: either the 1B mapper produces *valid-but-wrong* nested records, or
  the notebook's nested scorer was mis-wired. Either way it is **not a clean
  result** — it is precisely why the rigorous `eraparse` evaluator (Hungarian
  matching + ANLS + clean metric) was built.

### Takeaway
Parser choice changes **speed** and **text quality for skills**, but the 1B
mapper is the **bottleneck**: schema-valid output does not imply correct nested
records. `pymupdf4llm` is the best speed/quality parser to carry forward.

---

## Trial 2 — `cv-parsing-finetune-main` (Gemma-3-1B QLoRA fine-tune, dataset **v4**)

- **URL:** https://www.kaggle.com/code/anasahmad202202029/cv-parsing-finetune-main
- **Dataset:** EraMatch CV dataset **v4** (current)
- **Goal:** test whether fine-tuning a tiny 1B model fixes the
  "valid-but-wrong / can't-structure" problem from Trial 1.

### Approach
QLoRA fine-tune of **`unsloth/gemma-3-1b-it`** to map raw CV text → a **rich**
`CVSchema` directly (one-shot text→JSON). Rich schema includes `ContactInfo`,
`WorkExperience[]`, `Education[]`, `SkillEntry[]`, `Project[]`, `Certification[]`,
`MiscItem[]`, plus `seniority_level`, `primary_domain`, `years_of_experience`,
`has_github`, `has_linkedin`.

### Config
| Setting | Value |
|---|---|
| Base model | `unsloth/gemma-3-1b-it` (1B) |
| Quantization | 4-bit QLoRA |
| LoRA | r=16, alpha=16, dropout=0, all proj modules |
| max_seq_length | **2048** ⚠️ |
| Learning rate | 2e-4, warmup 0.03 |
| Epochs | 2 |
| Effective batch | 4 (bs=1 × grad-accum=4) |
| Data | ShareGPT format, **610 train / 128 test** |
| Hardware | Kaggle T4 |

### Results
| Metric | Base 1B (zero-shot) | Fine-tuned 1B |
|---|---:|---:|
| Parses as JSON | 60.2% | **not exported** |
| Schema-valid (pydantic) | **31.2%** | **not exported** |
| Median latency | 147 s/CV (unoptimized HF generate) | — |

- **Base 1B zero-shot: only 31.2% schema-valid, 39.8% unparseable.** The slow
  147 s median is an artifact — when the model can't form valid JSON it never
  emits a stop token and rambles to max-tokens (under vLLM a 1B model is
  sub-second).
- **The fine-tuned eval was computed in-notebook (base-vs-FT bar charts exist)
  but the executed outputs were never exported** — only `predictions_base.jsonl`
  was saved. The single most important number (does fine-tuning fix the 31%?)
  must be **re-run** (Phase B1).

### Known issue
`max_seq_length=2048` while the rich-schema system prompt alone is ~4,270 tokens
→ likely heavy input truncation during training/inference. Any re-run must use a
longer context (or compact-schema prompt) and vLLM serving.

---

## How these motivate the thesis

Both trials converge on the same problem statement, from two angles:

1. **Trial 1 (v3):** swapping parsers cannot fix nested-record accuracy — the
   small mapper is the bottleneck, and naive scoring is untrustworthy → need a
   rigorous evaluator (built: `eraparse`).
2. **Trial 2 (v4):** a tiny model zero-shot cannot reliably produce valid nested
   JSON (31% valid) → the core thesis question: *can fine-tuning + constrained
   decoding + architecture make a small model a reliable, faithful,
   schema-conformant nested extractor?*

These become the **"prior work / problem motivation"** section. The rigorous
re-measurement (A/B/C lanes, clean metric, faithfulness) lives in
`RESULTS_AND_PARETO.md`; the forward plan is `THESIS_DESIGN.md` /
`PHASED_TASK_PLAN.md`.

### Local copies of pulled artifacts
Adapters, predictions, and result CSVs pulled to (git-ignored dev area):
`Dev_temp_files/gemma3_finetune_output/` — adapter `gemma3-cv-adapter/`,
`predictions_base.jsonl`, `cv_parsing_2_output/benchmark_results/phase1/*.csv`.

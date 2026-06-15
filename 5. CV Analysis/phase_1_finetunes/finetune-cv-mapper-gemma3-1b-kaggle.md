# Fine-tune CV Mapper: Gemma 3 1B on Kaggle

## TL;DR

> Fine-tune `unsloth/gemma-3-1b-it` with QLoRA on Kaggle T4 (16GB VRAM) using the EraMatch CV dataset (cached extracted text + ground truth JSON), evaluate against the zero-shot base model with exact/fuzzy scoring matching the existing benchmark, export to GGUF Q5_K_M, and publish to Ollama with a correct Gemma 3 chat-template Modelfile.
>
> **Deliverables:**
> - `finetune_cv_mapper_gemma3_1b.ipynb`: End-to-end Kaggle notebook.
> - `gemma3-cv-parser.gguf`: Quantized fine-tuned model (~850MB).
> - `Modelfile`: Ollama recipe with Gemma 3 chat template, system prompt, and structured-output parameters.
> - `comparison_report.csv`: Side-by-side metrics (base vs fine-tuned): per-field F1, schema validation rate, hallucination rate.
>
> **Estimated Effort:** Medium (single Kaggle session, ~4–6 hours)
> **Parallel Execution:** NO — waves are sequential in one notebook, but cells are modular.
> **Critical Path:** Data loading → Training → Inference benchmark → Export → Verification

---

## Context

### Original Request
Create a new Kaggle notebook to fine-tune an LLM (Gemma 3 1B) as a fast CV-to-JSON mapper, using the existing EraMatch benchmark dataset and scoring methodology, with support for loading the fine-tuned model into Ollama and publishing it.

### Interview Summary
**Key Decisions:**
- Model: `unsloth/gemma-3-1b-it` (primary target).
- Notebook scope: **Fine-tuning only**, reusing pre-extracted text from `cv_parsing_benchmark.py` cache.
- Comparison: Evaluate base zero-shot vs. fine-tuned on the same held-out test set.
- Grammar decoding: Optional in evaluation, only if it doesn’t slow down or degrade accuracy.
- Ollama export: Merge adapters, export GGUF Q5_K_M, create Modelfile, publish to Ollama Hub / HuggingFace.
- Dataset: Use `splits/train.json` (7K IDs) stratified by tier for training, `splits/test.json` (1.5K IDs) for evaluation.
- Data format: ShareGPT (`conversations` with system/user/assistant roles).
- Scoring: Exact + fuzzy matching, replicating `cv_parsing_benchmark.py` logic (skill normalization, Levenshtein distance ≤ 2, fuzzy set P/R with threshold 0.8).

**Research Findings:**
- `cv_parsing_benchmark.py` scoring: `score_cv()` returns 40+ metrics per CV. `skill_set_f1()` handles exact/partial/combined F1 + MRR. `fuzzy_string_match()` uses Levenshtein distance ≤ 2. `fuzzy_set_pr()` matches sets with similarity threshold ≥ 0.8.
- Ollama export: Unsloth’s `save_pretrained_gguf()` merges adapters and quantizes in one step. Must use Gemma 3 chat template (`<start_of_turn>` / `<end_of_turn>`) and set `repeat_penalty 1.0` in Modelfile.
- Publishing: `ollama create`, `ollama cp`, `ollama push` to namespace/model-name on Ollama Hub. Alternative: direct HuggingFace `hf.co/username/repo`.
- VRAM constraints: Gemma 3 1B + QLoRA (r=16, 4-bit) fits comfortably on T4 (~8–10GB peak), leaving headroom for batch=2 and seq_len=4096.

### Metis Review
**Identified Gaps (addressed):**
- Cache limit: Benchmark extraction cache likely only covers the 1,000 sampled CVs. For training, we’ll use the raw `ground_truth/` files plus cached extractions where available, falling back to re-running extraction for uncached IDs (using `extract_docling_custom()` for quality).
- Alias fields: Ground truth JSONs have alias fields (`organization`, `title`, `university`, etc.) that must be stripped via `load_ground_truth()` before training.
- Schema canonicalization: CVSchema validators remap enum aliases (e.g., "Sr" → `senior`). Training prompts must supply canonical enum values to prevent validation mismatches.
- Scoring replication: Evaluation callback must replicate `normalize_skill()` byte-for-byte and import/use `FUZZY_MATCH_THRESHOLD = 0.8`.
- Scope creep locked: No multi-model comparison, no ablations, no real-time training visualization, no web UI.
- Test-set contamination guard: Training must strictly use `train.json` IDs; evaluation uses `test.json` IDs only.

---

## Work Objectives

### Core Objective
Create a production-grade Kaggle fine-tuning notebook that takes the EraMatch CV dataset, trains Gemma 3 1B to map extracted CV text to structured CVSchema JSON, proves measurable improvement over zero-shot base, and exports a deployment-ready Ollama model.

### Concrete Deliverables
1. `finetune_cv_mapper_gemma3_1b.ipynb`: Modular Kaggle notebook with install, data prep, training, evaluation, export cells.
2. `gemma3-cv-parser.gguf`: Fine-tuned model exported to Q5_K_M quantized GGUF.
3. `Modelfile`: Ollama recipe tuned for Gemma 3 structured-output inference.
4. `comparison_report.csv` + comparison chart: Side-by-side (base vs fine-tuned) F1 / precision / recall / Pydantic pass rate.

### Definition of Done
- [ ] Notebook runs end-to-end on Kaggle T4 without OOM.
- [ ] Fine-tuned model achieves >10% relative improvement in combined skill F1 over zero-shot base on `test.json`.
- [ ] Fine-tuned model achieves >95% Pydantic schema validation pass rate on `test.json`.
- [ ] Exported GGUF loads and runs in Ollama with correct JSON output structure.

### Must Have
- [ ] Unsloth QLoRA fine-tuning (`load_in_4bit=True`) on Kaggle T4.
- [ ] ShareGPT formatted training data, with alias stripping, canonical enum values in system prompt.
- [ ] Stratified data split by tier (T1–T5) to ensure balanced difficulty coverage.
- [ ] Exact replication of `cv_parsing_benchmark.py` scoring functions (`skill_set_f1`, `fuzzy_set_pr`, `fuzzy_string_match`).
- [ ] Base vs. fine-tuned comparison report with aggregated metrics.
- [ ] GGUF export using `save_pretrained_gguf()` to Q5_K_M.
- [ ] Ollama `Modelfile` with correct Gemma 3 chat template and system prompt.
- [ ] Publishing instructions for both Ollama Hub and HuggingFace.

### Must NOT Have (Guardrails)
- [ ] Must NOT train on `test.json` CV IDs — strict split enforcement.
- [ ] Must NOT use original alias fields in training JSON (strip via `load_ground_truth()`).
- [ ] Must NOT call `merge_and_unload()` before GGUF export (causes silent failures).
- [ ] Must NOT add new scoring metrics beyond what the benchmark uses.
- [ ] Must NOT implement multi-model comparison or ablation studies.
- [ ] Must NOT include real-time training visualization (W&B, Comet, etc.).
- [ ] Must NOT skip Pydantic validation before scoring (invalid JSON must be counted as failure).
- [ ] Must NOT use GGUF quantization below Q5_K_M for this sub-1B model (accuracy degrades disproportionately).
- [ ] Must NOT set `repeat_penalty` to Ollama default 1.1 for Gemma 3 Modelfile (use 1.0).
- [ ] Must NOT train on uncached CVs without first ensuring extraction pipeline runs with `docling_custom` (PyMuPDF raw text quality is too low for training).

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: PARTIAL — `cv_parsing_benchmark.py` has scoring functions but no pytest suite.
- **Automated tests**: NO — but every task includes agent-executed QA scenarios for notebook cell execution.
- **Agent-Executed QA**: YES — Primary verification. The executing agent runs each notebook cell, validates outputs, captures evidence.

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

> This is a single-notebook sequential pipeline, but cells are grouped into independent “waves” for clarity. Not all waves need to be run every time (restart-safe).

```
Wave 1 — Setup & Configuration (restart-safe):
├── Task 1: Install deps (unsloth, trl, peft, etc.) [quick]
├── Task 2: Define CVSchema + scoring functions (copied from benchmark) [quick]
└── Task 3: Define paths, constants, Kaggle config [quick]

Wave 2 — Data Pipeline (depends: Wave 1):
├── Task 4: Load & clean ground truth (alias stripping, canonical enums) [unspecified-high]
├── Task 5: Load / run extraction cache (docling_custom) [unspecified-high]
├── Task 6: Build ShareGPT dataset, stratify by tier, save to CSV/JSON [unspecified-high]
└── Task 7: Sanity checks: dataset length, roles distribution, sample preview [quick]

Wave 3 — Training (depends: Wave 2):
├── Task 8: Load Gemma 3 1B with Unsloth FastLanguageModel + QLoRA [deep]
├── Task 9: Configure SFTTrainer, train, save adapter weights [deep]
└── Task 10: Post-training: merge adapters (memory-safe) [unspecified-high]

Wave 4 — Evaluation (depends: Wave 3):
├── Task 11: Zero-shot baseline inference on test set (base model) [unspecified-high]
├── Task 12: Fine-tuned inference on test set (adapter model) [unspecified-high]
├── Task 13: Run scoring: per-field F1, fuzzy scores, Pydantic pass rate [unspecified-high]
├── Task 14: Build comparison table + chart (base vs fine-tuned) [visual-engineering]
└── Task 15: Optional: grammar-constrained decoding evaluation [unspecified-high]

Wave 5 — Export & Deployment (depends: Wave 3):
├── Task 16: Export to GGUF Q5_K_M with save_pretrained_gguf() [unspecified-high]
├── Task 17: Create Ollama Modelfile (Gemma 3 template, system prompt) [quick]
├── Task 18: Validate Ollama create + test inference locally [quick]
└── Task 19: Publish instructions (Ollama Hub + HuggingFace) [quick]

Wave FINAL — Review (after ALL tasks):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA: run notebook end-to-end on Kaggle (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay before completing.
```

---

## TODOs

- [x] 1. Install Dependencies & Environment Setup

  **What to do:**
  - Install `unsloth`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `transformers`, `datasets`, `torch`, `pydantic>=2.0`, `tqdm`, `pandas`, `matplotlib`, `seaborn`.
  - Verify GPU availability (`nvidia-smi`) and check RAM/VRAM.
  - Set random seeds for reproducibility.
  - Define `POSSIBLE_PATHS` matching Kaggle Input directory structures.

  **Must NOT do:**
  - Do not install heavy packages not needed (e.g., `docling`, `marker-pdf`) — extraction is pre-cached.
  - Do not set `IS_CLOUD` or Modal paths; this is a pure Kaggle notebook.

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: `using-git-worktrees` (if branching needed), none otherwise.
  - Why: This is a straightforward setup cell.

  **Parallelization:**
  - **Can Run In Parallel**: NO (sequential notebook cell)
  - **Blocked By**: None (can start immediately)
  - **Blocks**: Tasks 2, 3, 4, 5, 6, 7

  **References:**
  - `cv_parsing_benchmark.py:34-180` (pip_install, GPU detection, path constants).
  - `Fll_feature_plan.md` §3.4 (Unsloth standard recipe for LoRA hyperparameters).
  - `Fll_feature_plan.md` §3.8 (VRAM requirements on Kaggle T4).

  **Acceptance Criteria:**
  - [ ] All packages import without error: `import unsloth; import trl; import peft; import transformers; import torch`.
  - [ ] `torch.cuda.is_available()` returns `True` and device name prints `Tesla T4`.
  - [ ] `torch.cuda.get_device_properties(0).total_memory` reports ~15.9 GB VRAM.

  **QA Scenarios:**
  ```yaml
  Scenario: Install cell runs without error
    Tool: Bash (kaggle notebook cell)
    Preconditions: Kaggle notebook with GPU on, internet on
    Steps:
      1. Run install cell: `!pip install unsloth trl peft accelerate bitsandbytes`
      2. Verify imports: `python -c "import unsloth; print('✅ Unsloth OK')"`
    Expected Result: All imports succeed, no pip errors, GPU detected.
    Evidence: .sisyphus/evidence/task-1-install-success.png
  ```

  **Evidence to Capture:**
  - [ ] Screenshot of successful cell execution.
  - [ ] `nvidia-smi` output text file.

  **Commit**: NO (notebook cell artifact, not committed).

---

- [x] 2. Define CVSchema Pydantic Models (copied from benchmark)

  **What to do:**
  - Copy the `CVSchema` and all nested models (`WorkExperience`, `Education`, `SkillEntry`, `Project`, `Certification`, etc.) from `cv_parsing_benchmark.py` into the notebook.
  - Ensure `field_validator` for `seniority_level` and `primary_domain` canonicalization is included.
  - Export `CV_JSON_SCHEMA = CVSchema.model_json_schema()` for use in system prompts.
  - Define `load_ground_truth()` to strip alias fields and remap `title` → `job_title`, `organization` → `company`, etc.

  **Must NOT do:**
  - Do NOT simplify or truncate the schema — keep field descriptions so the model learns field nuances.
  - Do NOT omit validators — they define canonical enum forms.

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - Why: This is a copy/paste + minor adaptation task.

  **Parallelization:**
  - **Can Run In Parallel**: NO (sequential)
  - **Blocked By**: Task 1 (install)
  - **Blocks**: Tasks 4, 13 (scoring needs schema)

  **References:**
  - `cv_parsing_benchmark.py:400-984` (`CVSchema` definition with validators).
  - `cv_parsing_benchmark.py:1154-1182` (`load_ground_truth()` alias stripping).

  **Acceptance Criteria:**
  - [ ] `CVSchema.model_validate(gt_json)` succeeds on a representative `ground_truth/cv_XXXXX.json` after alias stripping.
  - [ ] `CV_JSON_SCHEMA` is a valid JSON schema dict with all fields.
  - [ ] Strip alias test: `load_ground_truth(path)` returns only canonical fields, no `organization` / `title` / `skills_flat` keys remain.

  **QA Scenarios:**
  ```yaml
  Scenario: Schema loads and validates a ground truth sample
    Tool: Python (notebook cell)
    Preconditions: cv_parsing_benchmark.py extracted text and ground_truth JSON available
    Steps:
      1. Load `cv_00001.json`, pass through `load_ground_truth()`
      2. Validate with `CVSchema.model_validate(stripped_json)`
      3. Check that `stripped_json` has no `skills_flat`, `organization`, `title` keys
    Expected Result: Pydantic validation passes, alias keys absent.
    Evidence: .sisyphus/evidence/task-2-schema-validation.png
  ```

  **Evidence to Capture:**
  - [ ] Screenshot of cell output showing validation pass.
  - [ ] JSON dump of stripped vs original ground truth for one CV.

  **Commit**: NO.

---

- [x] 3. Define Scoring Functions (exact + fuzzy)

  **What to do:**
  - Copy `score_cv()`, `skill_set_f1()`, `fuzzy_set_pr()`, `fuzzy_seq_match()`, `fuzzy_string_match()`, `normalize_skill()`, and `set_pr()` from `cv_parsing_benchmark.py` into the notebook.
  - Add `FuZZY_MATCH_THRESHOLD = 0.8` constant.
  - Ensure `normalize_prediction()` is included.
  - Provide a lightweight unit-test cell that asserts `skill_set_f1` produces expected scores on a synthetic example.

  **Must NOT do:**
  - Do NOT change thresholds, distance bounds, or normalization logic from the benchmark.
  - Do NOT use `rapidfuzz` or `thefuzz` for now — keep benchmark-identical logic to ensure apples-to-apples comparison.

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - Why: Copy/paste + unit test.

  **Parallelization:**
  - **Can Run In Parallel**: NO
  - **Blocked By**: Task 1
  - **Blocks**: Tasks 13, 14 (evaluation)

  **References:**
  - `cv_parsing_benchmark.py:2786-3160` (scoring functions).
  - `cv_parsing_benchmark.py:2891-2958` (`skill_set_f1()` exact + partial + MRR).

  **Acceptance Criteria:**
  - [ ] `score_cv()` returns a dict with 40+ keys on a valid `(pred, true)` pair.
  - [ ] `skill_set_f1({"Python", "Django"}, {"Python", "Django", "React"})` returns exact_f1 ~0.8, partial_f1 > 0.8, combined_f1 > 0.8.
  - [ ] `fuzzy_string_match("Sr Engineer", "Senior Engineer", max_distance=2)` returns `1.0`.

  **QA Scenarios:**
  ```yaml
  Scenario: Scoring functions produce expected synthetic scores
    Tool: Python (notebook cell)
    Preconditions: scoring functions loaded
    Steps:
      1. Create synthetic pred CVSchema and true CVSchema
      2. Run score_cv(pred, true) and assert key metrics are within expected ranges
      3. Run skill_set_f1 on known skill sets
    Expected Result: exact_f1=1.0 when sets identical, 0.8 when 2/3 match, etc.
    Evidence: .sisyphus/evidence/task-3-scoring-unit-test.png
  ```

  **Evidence to Capture:**
  - [ ] Cell output showing assertion passes.
  - [ ] Text file with `score_cv()` output on one example.

  **Commit**: NO.

---

- [x] 4. Load & Clean Ground Truth Data

  **What to do:**
  - Read `metadata.json` to get full manifest (cv_id, tier, template, domain).
  - Read `splits/train.json`, `splits/test.json` to get train/test IDs.
  - For each train ID: read `ground_truth/cv_XXXXX.json`, strip aliases via `load_ground_truth()`, flatten nested data for training input.
  - Compute tier distribution and report class balance. Stratify sampling if needed.

  **Must NOT do:**
  - Do NOT accidentally load and use `test.json` IDs during training.
  - Do NOT skip alias stripping.

  **Recommended Agent Profile:**
  - **Category**: `deep`
  - Why: Data quality is critical — need to verify alias stripping, canonicalization, and stratification.

  **Parallelization:**
  - **Can Run In Parallel**: NO
  - **Blocked By**: Tasks 2 (schema), 1 (install)
  - **Blocks**: Tasks 5, 6

  **References:**
  - `cv_parsing_benchmark.py:996-1120` (load_manifest, allocate_balanced_counts, attach_file_paths).
  - `cv_parsing_benchmark.py:1154-1182` (load_ground_truth alias stripping).

  **Acceptance Criteria:**
  - [ ] Train set contains >= 5K unique CV IDs from `train.json`.
  - [ ] Test set contains >= 1K unique CV IDs from `test.json`, no overlap with train.
  - [ ] Alias fields (`organization`, `title`, `skills_flat`, etc.) are absent from all loaded ground truth dicts.
  - [ ] Tier distribution is roughly balanced (±10% per tier).
  - [ ] All canonical enum values (e.g., `senior`, `backend`) are used; no alias forms remain.

  **QA Scenarios:**
  ```yaml
  Scenario: Ground truth data is clean and stratified
    Tool: Python (notebook cell)
    Preconditions: metadata.json and splits JSONs loaded
    Steps:
      1. Load train/test manifests, compare IDs for overlap
      2. Check `cv_XXXXX.json` entries for alias key absence
      3. Count tiers across sets
    Expected Result: Zero overlapping IDs, alias-free dicts, balanced tiers.
    Evidence: .sisyphus/evidence/task-4-ground-truth-qa.png
  ```

  **Evidence to Capture:**
  - [ ] Histogram of tier distribution per split.
  - [ ] Text log of alias stripping verification.

  **Commit**: NO.

---

- [x] 5. Load / Generate Extraction Cache (docling_custom)

  **What to do:**
  - Try to load `extractions__{SAMPLE_TAG}.pkl` from the existing benchmark cache.
  - For any CVs in `train.json` / `test.json` without cached extraction, run `extract_docling_custom()` (table structure + sidebar reorder) to extract text.
  - Save a new consolidated extraction cache for the training notebook.

  **Must NOT do:**
  - Do NOT use `extract_pymupdf_raw` or `extract_pymupdf4llm` — docling_custom is the best quality.
  - Do NOT skip cached results if they exist.

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`
  - Why: May involve running Docling extraction (potentially slow), needs deep understanding.

  **Parallelization:**
  - **Can Run In Parallel**: NO
  - **Blocked By**: Task 4 (needs ground truth IDs)
  - **Blocks**: Task 6

  **References:**
  - `cv_parsing_benchmark.py:1240-1480` (extraction functions, docling_custom implementation).
  - `cv_parsing_benchmark.py:379-392` (EXTRACTION_CACHE_PATH naming convention).

  **Acceptance Criteria:**
  - [ ] Every train ID has extracted text available (either from cache or freshly extracted).
  - [ ] Every test ID has extracted text available.
  - [ ] Sample extraction preview looks reasonable (no garbled text, layout sections preserved).

  **QA Scenarios:**
  ```yaml
  Scenario: Extraction cache is complete for all train/test IDs
    Tool: Python (notebook cell)
    Preconditions: train/test IDs loaded, benchmark cache searched
    Steps:
      1. Count IDs with cached extraction vs total
      2. For missing IDs, run extract_docling_custom() inline, verify output
      3. Save to new .pkl, then reload and verify count matches
    Expected Result: 100% coverage on both splits.
    Evidence: .sisyphus/evidence/task-5-extraction-coverage.png
  ```

  **Evidence to Capture:**
  - [ ] Extraction coverage table (cached vs fresh).
  - [ ] Sample extracted text for one T1 and one T3 CV.

  **Commit**: NO.

---

- [x] 6. Build ShareGPT Dataset

  **What to do:**
  - For each (cv_id, extracted_text, ground_truth_json):
    - System message: `"You are a CV extractor. Extract structured data and return valid JSON matching this schema: {cv_json_schema_str}"`
    - User message: Extracted CV text (docling_custom output).
    - Assistant message: Ground truth JSON (stringified, alias-free).
    - Format as ShareGPT `conversations` list.
  - Save dataset to HuggingFace `datasets` Dataset format or JSONL.
  - Stratify train/test split by tier.
  - Show sample entries for spot-check.

  **Must NOT do:**
  - Do NOT use raw ground truth JSON with aliases in the assistant message.
  - Do NOT include test IDs in training split.
  - Do NOT shorten system prompt schema — include full CVSchema definition with field descriptions.

  **Recommended Agent Profile:**
  - **Category**: `deep`
  - Why: Prompt engineering and data formatting are high-importance for fine-tuning success.

  **Parallelization:**
  - **Can Run In Parallel**: NO
  - **Blocked By**: Tasks 4, 5
  - **Blocks**: Tasks 8, 9

  **References:**
  - `cv_parsing_benchmark.py:400-990` (CVSchema field descriptions — use these in system prompt).
  - `Fll_feature_plan.md` §3.5 (Unsloth dataset format: Alpaca, ShareGPT, custom raw text).
  - Unsloth docs: `standardize_data_formats` helper.

  **Acceptance Criteria:**
  - [ ] Dataset length matches train/test split counts.
  - [ ] Every sample has exactly 3 roles: system, user, assistant.
  - [ ] System message includes valid JSON schema string derived from `CVSchema.model_json_schema()`.
  - [ ] Assistant content is valid JSON that passes `json.loads()` and `CVSchema.model_validate()`.
  - [ ] Sample preview shows correct schema definition and clean JSON output.

  **QA Scenarios:**
  ```yaml
  Scenario: Dataset is correctly formatted ShareGPT
    Tool: Python (notebook cell)
    Preconditions: cleaned ground truth and extractions loaded
    Steps:
      1. Build dataset, inspect first 3 entries
      2. Check each entry has 3 conversations with correct roles
      3. Parse assistant content JSON, validate against CVSchema
      4. Assert test IDs are absent from training set
    Expected Result: No schema errors, correct roles, no leaked test IDs.
    Evidence: .sisyphus/evidence/task-6-sharegpt-dataset.png
  ```

  **Evidence to Capture:**
  - [ ] Screenshot/pretty-print of first 3 ShareGPT samples.
  - [ ] JSONL dump of first 100 training samples.
  - [ ] Bar chart of per-tier sample counts.

  **Commit**: NO.

---

- [x] 7. Sanity Checks & Dataset Validation

  **What to do:**
  - Compute text length distribution (user messages) — flag any > 4096 tokens.
  - Check schema validation rate on assistant messages (should be 100%).
  - Verify no test IDs leaked into training.
  - Save dataset to Kaggle working directory (`/kaggle/working/`) for restart safety.

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Acceptance Criteria:**
  - [ ] 95%+ of user messages are < 4096 tokens.
  - [ ] 100% of assistant messages pass `json.loads` and `CVSchema.model_validate()`.
  - [ ] Zero overlap between train and test IDs.

  **QA Scenarios:**
  ```yaml
  Scenario: Dataset passes all sanity checks
    Tool: Python (notebook cell)
    Steps:
      1. Tokenize user messages, plot length histogram
      2. Batch validate assistant JSONs
      3. Assert train ∩ test = ∅
    Expected Result: All checks pass.
    Evidence: .sisyphus/evidence/task-7-sanity-checks.png
  ```

  **Evidence to Capture:**
  - [ ] Text length histogram image.
  - [ ] Token count summary text.

  **Commit**: NO.

---

- [x] 8. Load Gemma 3 1B with Unsloth

  **What to do:**
  - Use `FastLanguageModel.from_pretrained("unsloth/gemma-3-1b-it", max_seq_length=4096, load_in_4bit=True, full_finetuning=False)`.
  - Wrap with `FastLanguageModel.get_peft_model()`:
    - `r=16`, `lora_alpha=16`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
    - `lora_dropout=0`, `bias="none"`, `use_gradient_checkpointing="unsloth"`.
    - `random_state=3407`.
  - Print VRAM usage after loading.

  **Must NOT do:**
  - Do NOT use `load_in_4bit=False` on Kaggle T4 (will OOM).
  - Do NOT set `full_finetuning=True`.

  **Recommended Agent Profile:**
  - **Category**: `deep`
  - Why: VRAM management and Unsloth internals are complex.

  **Parallelization:**
  - **Can Run In Parallel**: NO
  - **Blocked By**: Tasks 1 (install), 7 (sanity checks)
  - **Blocks**: Task 9

  **References:**
  - `Fll_feature_plan.md` §3.4 (Unsloth standard recipe).
  - `Fll_feature_plan.md` §3.8 (Kaggle T4 VRAM requirements).
  - Unsloth docs: `FastLanguageModel` API reference.

  **Acceptance Criteria:**
  - [ ] Model loads successfully on T4 without OOM.
  - [ ] VRAM usage after model load is < 6 GB (leaving room for training).
  - [ ] PEFT adapter config prints correctly with rank=16.
  - [ ] `model.print_trainable_parameters()` reports a small fraction (e.g., < 5%) of total params trainable.

  **QA Scenarios:**
  ```yaml
  Scenario: Model loads within T4 VRAM limits
    Tool: Python (notebook cell)
    Preconditions: Kaggle T4 with 16GB VRAM
    Steps:
      1. Execute model loading cell
      2. Print `torch.cuda.memory_allocated() / 1e9` before and after
      3. Verify `model.print_trainable_parameters()` output
    Expected Result: Memory allocated < 6 GB. No OOM.
    Evidence: .sisyphus/evidence/task-8-model-load-vram.png
  ```

  **Evidence to Capture:**
  - [ ] Screenshot of `nvidia-smi` or `torch.cuda.memory_allocated()` output.
  - [ ] Text of `model.print_trainable_parameters()`.

  **Commit**: NO.

---

- [x] 9. Fine-tune with SFTTrainer

  **What to do:**
  - Configure `SFTTrainer` from `trl`:
    - `per_device_train_batch_size=2` or `1` (depending on VRAM).
    - `gradient_accumulation_steps=4`.
    - `learning_rate=2e-4`.
    - `lr_scheduler_type="linear"`.
    - `warmup_ratio=0.03`.
    - `num_train_epochs=2` (or 3, time permitting).
    - `weight_decay=0.01`.
    - `optim="adamw_8bit"`.
    - `dataset_text_field="text"` or dataset prepared with `conversations` format.
    - `max_seq_length=4096` (or 2048 if memory is tight).
  - Train and save adapter weights to `/kaggle/working/gemma3-cv-adapter`.
  - Log training loss curve.

  **Must NOT do:**
  - Do NOT use `bf16` on T4 (not supported); use `fp16=True`.
  - Do NOT set `save_strategy="steps"` with high frequency (slows training).

  **Recommended Agent Profile:**
  - **Category**: `deep`
  - Why: This is the core training step; hyperparameters need tuning based on VRAM.

  **Parallelization:**
  - **Can Run In Parallel**: NO
  - **Blocked By**: Task 8
  - **Blocks**: Task 10 (merge adapters), Task 16 (export)

  **References:**
  - `Fll_feature_plan.md` §3.4 (training recipe).
  - Unsloth docs: `SFTTrainer` configuration examples for Gemma 3.

  **Acceptance Criteria:**
  - [ ] Training completes without OOM.
  - [ ] Training time < 3 hours for 2 epochs on ~5K samples.
  - [ ] Final training loss is < 0.5 (or lower, depending on data complexity).
  - [ ] Adapter weights saved to disk.
  - [ ] Loss curve is monotonically decreasing (or mostly decreasing).

  **QA Scenarios:**
  ```yaml
  Scenario: Training completes successfully on Kaggle T4
    Tool: Python (notebook cell)
    Preconditions: Model loaded, dataset prepared
    Steps:
      1. Start SFTTrainer training loop
      2. Monitor `torch.cuda.memory_allocated()` every 100 steps
      3. After training, plot loss curve and save adapter weights
    Expected Result: No OOM, loss decreases, adapter weights saved.
    Evidence: .sisyphus/evidence/task-9-training-loss-curve.png
  ```

  **Evidence to Capture:**
  - [ ] Training loss curve plot.
  - [ ] Final loss value text.
  - [ ] Adapter weights directory listing.

  **Commit**: NO.

---

- [x] 10. Post-training Merge (Memory-Safe)
- [x] 11. Zero-Shot Base Inference (Baseline)
- [x] 12. Fine-Tuned Model Inference
- [x] 13. Scoring & Metrics Extraction
- [x] 14. Comparison Report + Chart
- [x] 15. Optional: Grammar-Constrained Decoding Evaluation
- [x] 16. Export Fine-Tuned Model to GGUF
- [x] 17. Create Ollama Modelfile
- [x] 18. Validate Ollama Create + Test Inference
- [x] 19. Publish Instructions (Ollama Hub + HuggingFace)

  **What to do:**
  - After training, save adapter weights.
  - For evaluation, temporarily merge adapters into base model using `model = model.merge_and_unload()` (only if needed for inference).
  - For export, do NOT call `merge_and_unload` — Unsloth’s `save_pretrained_gguf()` handles merging internally.
  - Clear CUDA cache after merge.

  **Must NOT do:**
  - Do NOT call `merge_and_unload()` before `save_pretrained_gguf()` — causes silent failures.

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`
  - Why: Memory management and Unsloth-specific quirks.

  **Acceptance Criteria:**
  - [ ] Merged model loads correctly and produces coherent text.
  - [ ] VRAM does not spike above 12 GB during merge.

  **QA Scenarios:**
  ```yaml
  Scenario: Merge works and inference is stable
    Tool: Python (notebook cell)
    Steps:
      1. Merge adapters, test inference on one sample
      2. Verify output is valid JSON
      3. Check VRAM usage after merge
    Expected Result: JSON output valid, no OOM.
    Evidence: .sisyphus/evidence/task-10-merge-qa.png
  ```

  **Evidence to Capture:**
  - [ ] Sample inference output.
  - [ ] VRAM usage after merge.

  **Commit**: NO.

---

- [x] 11. Zero-Shot Base Inference (Baseline)

  **What to do:**
  - Load the **base** (non-fine-tuned) `unsloth/gemma-3-1b-it` model.
  - For each CV in test set, pass extracted text through the prompt (same system/user format as training, but no fine-tuning).
  - Parse model output as JSON, validate with `CVSchema`.
  - Save predictions to predictions_base.jsonl.

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`
  - Why: Inference on all test CVs (1.5K samples) is time-consuming.

  **Parallelization:**
  - **Can Run In Parallel**: NO (sequential)
  - **Blocked By**: Task 8 (model loading pattern)
  - **Blocks**: Task 13 (scoring)

  **References:**
  - `cv_parsing_benchmark.py:1500-1800` (mapping / inference cell block, prompt construction).
  - `cv_parsing_benchmark.py:3200-3400` (Ollama client or huggingface inference pattern).

  **Acceptance Criteria:**
  - [ ] Base model produces output for every test CV.
  - [ ] Pydantic pass rate is recorded (expected ~18–40% for zero-shot small model).
  - [ ] Predictions saved to disk.

  **QA Scenarios:**
  ```yaml
  Scenario: Base model inference on test set
    Tool: Python (notebook cell)
    Steps:
      1. Load base model, run on first 10 test samples
      2. Parse JSON, count Pydantic validation success
      3. Run full test set (1.5K) if OK
    Expected Result: All samples processed, pass rate computed.
    Evidence: .sisyphus/evidence/task-11-base-inference.png
  ```

  **Evidence to Capture:**
  - [ ] First 5 inference outputs (pretty-printed).
  - [ ] Pydantic pass rate summary.
  - [ ] Histogram of output lengths.

  **Commit**: NO.

---

- [x] 12. Fine-Tuned Model Inference

  **What to do:**
  - Load the fine-tuned model (with merged adapters or adapter weights).
  - Run exact same inference code as Task 11 on the same test set.
  - Parse outputs, validate with Pydantic.
  - Save predictions to predictions_finetuned.jsonl.

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`

  **Acceptance Criteria:**
  - [ ] Fine-tuned model produces output for every test CV.
  - [ ] Pydantic pass rate is recorded (expected > 90%).
  - [ ] Predictions saved to disk.

  **QA Scenarios:**
  ```yaml
  Scenario: Fine-tuned inference on test set
    Tool: Python (notebook cell)
    Steps:
      1. Load fine-tuned model, run on first 10 test samples
      2. Parse JSON, compare to base outputs
      3. Run full test set
    Expected Result: Fine-tuned outputs are more structured, higher pass rate.
    Evidence: .sisyphus/evidence/task-12-finetuned-inference.png
  ```

  **Evidence to Capture:**
  - [ ] First 5 fine-tuned outputs vs base outputs side-by-side.
  - [ ] Pydantic pass rate summary.

  **Commit**: NO.

---

- [x] 13. Scoring & Metrics Extraction

  **What to do:**
  - For each test CV, run `score_cv(pred=prediction, true=ground_truth)` using `cv_parsing_benchmark.py` scoring logic.
  - Aggregate per-field metrics: exact F1, fuzzy F1, precision, recall.
  - Compute macro averages across test set.
  - Compute hallucination rate and missing field rate.
  - Compute combined accuracy metric.

  **Must NOT do:**
  - Do NOT change scoring thresholds.
  - Do NOT skip invalid predictions — count them as zero scores.

  **Recommended Agent Profile:**
  - **Category**: `deep`
  - Why: Scoring logic replication is critical for apples-to-apples comparison.

  **Acceptance Criteria:**
  - [ ] `score_cv()` runs on every test CV pair (base + fine-tuned).
  - [ ] Output dicts contain all 40+ fields.
  - [ ] skill_set_f1 exact/partial/combined are included.
  - [ ] Hallucination rate and missing field rate are calculated.

  **QA Scenarios:**
  ```yaml
  Scenario: Scoring pipeline runs correctly on predictions
    Tool: Python (notebook cell)
    Steps:
      1. Load base + fine-tuned predictions
      2. Run score_cv() on a random sample of 50 CVs
      3. Check that all 40+ metric keys exist
    Expected Result: No KeyErrors, score dicts complete.
    Evidence: .sisyphus/evidence/task-13-scoring-qa.png
  ```

  **Evidence to Capture:**
  - [ ] Sample score dict for one CV.
  - [ ] Aggregate summary tables (CSV text).

  **Commit**: NO.

---

- [x] 14. Comparison Report + Chart

  **What to do:**
  - Build a side-by-side comparison table: base vs fine-tuned for all fields.
  - Create bar charts / radar charts per metric group (skills, work experience, education, flat fields).
  - Highlight Pydantic pass rate delta.
  - Export CSV report `comparison_report.csv`.

  **Recommended Agent Profile:**
  - **Category**: `visual-engineering`
  - Skills: `frontend-design:frontend-design` for chart aesthetics (optional — since this is matplotlib/seaborn within a notebook, a general `unspecified-high` agent with Python/plotting skills is sufficient).
  - Why: Visualization quality matters for the research paper / blog post narrative.

  **Acceptance Criteria:**
  - [ ] CSV report exists with base + fine-tuned per-field metrics.
  - [ ] Charts are readable and show clear improvement in fine-tuned model.
  - [ ] Pydantic pass rate improvement is prominently displayed.

  **QA Scenarios:**
  ```yaml
  Scenario: Comparison report is generated and readable
    Tool: Python (notebook cell)
    Steps:
      1. Generate comparison table and chart
      2. Save CSV and PNG to /kaggle/working/
      3. Open PNG and verify readability
    Expected Result: Charts saved, fields clearly labeled.
    Evidence: .sisyphus/evidence/task-14-comparison-chart.png
  ```

  **Evidence to Capture:**
  - [ ] Screenshot of comparison chart.
  - [ ] CSV file content (first 20 lines).

  **Commit**: NO.

---

- [x] 15. Optional: Grammar-Constrained Decoding Evaluation

  **What to do:**
  - Convert `CV_JSON_SCHEMA` to LlamaGrammar GBNF format using `json_schema_to_grammar.py`.
  - Run a subset of test inference (e.g., 50 CVs) with `grammar=cv_grammar`.
  - Compare pass rate and F1 with/without grammar.
  - If grammar slows down inference significantly (>2×) OR reduces F1, document and skip for full test.

  **Must NOT do:**
  - Do NOT run full 1.5K test with grammar if it’s too slow.
  - Do NOT use grammar if it causes F1 degradation.

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`
  - Why: Grammar conversion and evaluation is a specialist task.

  **Acceptance Criteria:**
  - [ ] Grammar generation succeeds from CVSchema.
  - [ ] 50-sample grammar inference completes within reasonable time.
  - [ ] Pass rate and F1 documented with vs without grammar.

  **QA Scenarios:**
  ```yaml
  Scenario: Grammar evaluation on 50-sample subset
    Tool: Python (notebook cell)
    Steps:
      1. Generate GBNF grammar from CV_JSON_SCHEMA
      2. Run inference on 50 test samples with grammar
      3. Compare metrics to no-grammar baseline
    Expected Result: Metrics reported, decision to enable/disable for full test documented.
    Evidence: .sisyphus/evidence/task-15-grammar-eval.png
  ```

  **Evidence to Capture:**
  - [ ] Grammar file snippet.
  - [ ] Metrics comparison table.

  **Commit**: NO.

---

- [x] 16. Export Fine-Tuned Model to GGUF

  **What to do:**
  - Use `model.save_pretrained_gguf("gemma3-cv-parser", tokenizer, quantization_method="q5_k_m")`.
  - Verify output file exists and size is ~850MB.
  - Do NOT call `merge_and_unload()` first.

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`
  - Why: Unsloth-specific GGUF export with quantization settings.

  **Acceptance Criteria:**
  - [ ] GGUF file generated at `gemma3-cv-parser/unsloth.Q5_K_M.gguf`.
  - [ ] File size is between 800MB and 900MB.
  - [ ] File loads in `llama.cpp` or Ollama without errors.

  **QA Scenarios:**
  ```yaml
  Scenario: GGUF export completes and file is valid
    Tool: Python (notebook cell)
    Steps:
      1. Run save_pretrained_gguf() with q5_k_m
      2. Check file size with `ls -lh`
      3. Attempt to load via `llama_model = GGUFLoader(file)` or `ollama create`
    Expected Result: File size ~850MB, Ollama create succeeds.
    Evidence: .sisyphus/evidence/task-16-gguf-export.png
  ```

  **Evidence to Capture:**
  - [ ] File size screenshot.
  - [ ] Ollama create success log.

  **Commit**: NO.

---

- [x] 17. Create Ollama Modelfile

  **What to do:**
  - Create a `Modelfile` with:
    - `FROM ./gemma3-cv-parser/unsloth.Q5_K_M.gguf`
    - `TEMPLATE` using Gemma 3 `<start_of_turn>` / `<end_of_turn>` blocks (Go template).
    - `SYSTEM`: The structured output system prompt (same as training).
    - `PARAMETER temperature 0.1`, `top_k 64`, `top_p 0.95`, `repeat_penalty 1.0`, `num_ctx 4096`.
    - `PARAMETER stop "<end_of_turn>"`, `PARAMETER stop "<start_of_turn>"`.
  - Save Modelfile to `/kaggle/working/Modelfile`.

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Acceptance Criteria:**
  - [ ] Modelfile is syntactically valid (Ollama parses without error).
  - [ ] Chat template handles system + user + assistant turns correctly.
  - [ ] Stop tokens prevent infinite generation.

  **QA Scenarios:**
  ```yaml
  Scenario: Modelfile is valid and matches Gemma 3 template
    Tool: Bash (Ollama CLI)
    Steps:
      1. Run `ollama create gemma3-cv-parser -f Modelfile`
      2. Verify `ollama show gemma3-cv-parser --modelfile` returns correct content
      3. Run a one-shot inference: `echo '{"messages":[{"role":"system","content":"..."},{"role":"user","content":"...CV text..."}]}' | ollama run gemma3-cv-parser`
    Expected Result: Model created, inference returns JSON.
    Evidence: .sisyphus/evidence/task-17-ollama-modelfile.png
  ```

  **Evidence to Capture:**
  - [ ] Modelfile content text.
  - [ ] Ollama show output.
  - [ ] One-shot inference output.

  **Commit**: NO.

---

- [x] 18. Validate Ollama Create + Test Inference

  **What to do:**
  - In the notebook (or locally/another Kaggle cell), run `ollama create gemma3-cv-parser -f Modelfile`.
  - Send one test CV text via `curl` or Ollama Python client.
  - Verify output JSON passes Pydantic validation.

  **Recommended Agent Profile:**
  - **Category**: `quick`

  **Acceptance Criteria:**
  - [ ] Ollama model loads and responds.
  - [ ] Output JSON is valid and passes CVSchema validation.
  - [ ] Response time < 10 seconds per CV on local CPU/GPU.

  **QA Scenarios:**
  ```yaml
  Scenario: Ollama inference produces valid JSON
    Tool: Bash (curl) or Python (ollama client)
    Steps:
      1. `ollama create gemma3-cv-parser -f Modelfile`
      2. `curl -s http://localhost:11434/api/generate -d '{"model":"gemma3-cv-parser","prompt":"...CV text..."}' | jq .response`
      3. Validate JSON against CVSchema
    Expected Result: JSON validates, no schema errors.
    Evidence: .sisyphus/evidence/task-18-ollama-inference.png
  ```

  **Evidence to Capture:**
  - [ ] Curl output text.
  - [ ] Pydantic validation result.

  **Commit**: NO.

---

- [x] 19. Publish Instructions (Ollama Hub + HuggingFace)

  **What to do:**
  - Write clear markdown instructions in the notebook for:
    - Publishing to Ollama Hub: `ollama signin`, `ollama cp gemma3-cv-parser username/gemma3-cv-parser`, `ollama push username/gemma3-cv-parser`.
    - Publishing to HuggingFace: `push_to_hub_gguf()` command snippet, username/repo setup.
    - Sharing GGUF + Modelfile directly for offline use.

  **Recommended Agent Profile:**
  - **Category**: `writing`

  **Acceptance Criteria:**
  - [ ] Instructions are in the final markdown cell of the notebook.
  - [ ] Commands are copy-paste ready.
  - [ ] Both Ollama Hub and HuggingFace paths are documented.

  **QA Scenarios:**
  ```yaml
  Scenario: Publish instructions are documented
    Tool: Read (notebook markdown cell)
    Steps:
      1. Scroll to final cell
      2. Verify presence of `ollama signin`, `ollama cp`, `ollama push` commands
      3. Verify presence of HuggingFace `push_to_hub_gguf` snippet
    Expected Result: All three sharing methods documented.
    Evidence: .sisyphus/evidence/task-19-publish-instructions.png
  ```

  **Evidence to Capture:**
  - [ ] Screenshot of final markdown cell.

  **Commit**: NO.

---

## Final Verification Wave

> 4 review agents run in PARALLEL after all notebook cells are complete. ALL must APPROVE. Present consolidated results to user before marking done.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the generated notebook end-to-end. For each "Must Have": verify the cell exists, produces the expected output (install cell imports packages, training cell saves adapter, export cell produces GGUF, etc.). For each "Must NOT Have": search the notebook for forbidden patterns (`merge_and_unload` before GGUF export, `test.json` IDs in training set, etc.).
  Output: Must Have [Yes/No per item] | Must NOT Have [Yes/No per item] | Notebook cells [N complete] | VERDICT: APPROVE/REJECT

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run a static check on the notebook (convert to .py if needed): check for `as any` / `@ts-ignore` equivalents, bare `except:` blocks, unused imports, hardcoded paths, Python 3.11+ compatibility. Verify Pydantic v2 usage.
  Output: Lint issues [N found/N total lines] | Code style [PASS/WARN] | VERDICT

- [x] F3. **Real Manual QA** — `unspecified-high`
  Run the notebook end-to-end on Kaggle T4 (or simulate if not available). Execute each cell in order, capture evidence. Verify: training converges, evaluation shows improvement, GGUF is valid, Ollama create works.
  Output: Cells [N/N pass] | Training converged [YES/NO] | Fine-tuned F1 > base F1 [YES/NO] | VERDICT

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual notebook cell content. Verify 1:1 completeness — everything in spec was built, nothing beyond spec was built. Check cross-cell contamination (e.g., evaluation cell touching data-prep files).
  Output: Tasks [N/N compliant] | Scope creep [CLEAN/N issues] | VERDICT

---

## Commit Strategy

- **This plan creates a SINGLE file** (`finetune_cv_mapper_gemma3_1b.ipynb`).
- No git commits during notebook execution. Kaggle notebook outputs are saved as artifacts (GGUF, Modelfile, comparison_report.csv, etc.).
- After notebook is reviewed and approved, user may download GGUF + Modelfile and create a git commit or push to Ollama Hub / HuggingFace as a separate step.

---

## Success Criteria

### Verification Commands

```bash
# 1. Notebook runs cell-by-cell without error
jupyter nbconvert --to notebook --execute finetune_cv_mapper_gemma3_1b.ipynb

# 2. GGUF file exists and size is valid
stat -f%z gemma3-cv-parser/unsloth.Q5_K_M.gguf && echo "OK" || echo "FAIL"

# 3. Ollama create works
ollama create gemma3-cv-parser -f Modelfile && echo "OK" || echo "FAIL"

# 4. Ollama inference returns valid JSON
result=$(curl -s http://localhost:11434/api/generate -d '{"model":"gemma3-cv-parser","prompt":"Extract CV to JSON: Name: John Doe..."}')
python -c "import json; json.loads('$result')" && echo "JSON OK" || echo "JSON FAIL"
```

### Final Checklist
- [ ] All "Must Have" present in notebook cells.
- [ ] All "Must NOT Have" absent from notebook.
- [ ] Training converges without OOM.
- [ ] Fine-tuned F1 > base F1 on test set.
- [ ] Pydantic pass rate > 95% for fine-tuned model.
- [ ] GGUF exported successfully, size ~850MB.
- [ ] Modelfile valid, Ollama inference produces valid JSON.
- [ ] Publish instructions documented.
- [ ] Final verification agents all approve.

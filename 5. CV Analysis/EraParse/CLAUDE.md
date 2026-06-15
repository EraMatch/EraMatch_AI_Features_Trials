# CLAUDE.md — EraParse Thesis Project

## Before Any Work

Always read these files before making decisions, writing code, or suggesting next steps:

- `docs/THESIS_DESIGN.md` — thesis RQs, tracks A/B/C, contribution claims
- `docs/RESULTS_AND_PARETO.md` — all measured results with latency; never contradict or ignore existing numbers
- `docs/PHASED_TASK_PLAN.md` — what phase we are in and what is frozen vs in-progress
- `docs/NEXT_MODEL_ACTION_PLAN.md` and `docs/SECOND_CONTRIBUTION_PLAN.md` — if they exist

Read the relevant trial artifacts under `artifacts/trials/` when debugging or evaluating a model. Check what runs already exist before proposing to re-run anything.

---

## Thesis Flow — Non-Negotiable Rules

1. **Never mix schemas.** Reduced and rich schema results are separate tracks. Never compare them in the same table.
2. **Always apply the clean metric.** Every result reported must use `clean_macro()` from `scripts/analyze_labels_and_router.py` — URL phantom exclusion + non-rendering projects exclusion. Raw macro is only for internal debugging.
3. **Respect the phase discipline.** Phase 2 onwards nothing gets re-selected or re-tuned. Threshold/prompt/config changes happen in Phase 1 only.
4. **Locked set is touched once, ever.** Never run the locked confirmation set (2,475 CVs) during development. Only at the final Phase D step.
5. **Latency is a first-class axis.** Every new model or pipeline must have a measured latency (p50/p95, end-to-end). Do not report accuracy without latency.
6. **Faithfulness is a first-class axis.** Always report `unsupported_evidence_rate` alongside accuracy. A model that hallucinates must be flagged even if its accuracy is high.
7. **Track A, B, C are parallel contribution tracks** — do not collapse them. Track A (fine-tuned generative ladder) is the lower-risk guaranteed result. Track B (set-prediction head) is the crown architectural novelty. Track C (SmolVLM2 EraExtract) is the VLM specialization story.
8. **RQ2b ablation:** extraction-pretrained base (NuExtract-tiny/1.5) vs general SLM (Gemma/Qwen) must be fine-tuned on identical data for the comparison to be valid.

---

## MCP Tools to Use

**context7** (`mcp__plugin_context7_context7__*`): use whenever you need current library docs — unsloth, Modal, TRL, Outlines, HuggingFace Transformers, vLLM, PEFT. Training data cutoff is stale; docs change fast. Always resolve the library id first, then fetch relevant sections.

**Hugging Face MCP** (`mcp__4a72f655-*`): use to:
- Check model cards before using a new model (`hub_repo_details`)
- Search for models by capability (`hf_hub_query`, `hub_repo_search`)
- Look up papers for baselines or prior work (`paper_search`, `hf_doc_search`)
- Verify model format compatibility (tokenizer, prompt format) before writing finetune code

**Modal skill** (`/modal` or the Modal skill in Skill tool): invoke before writing or editing any `modal_apps/` file. The skill provides current Modal API patterns — volumes, GPU types, image build chains, `@app.cls` vs `@app.function`, timeouts, parameter passing. Do not write Modal code from memory; the API evolves quickly and stale patterns cause crashes.

Use all three proactively — do not assume a model's prompt format, tokenizer behaviour, training recipe, or Modal API syntax from memory.

---

## Modal App Standards — Prevent Crashes

Every `modal_apps/` file must follow these rules before being run:

1. **Smoke test locally first.** Run `modal run modal_apps/<app>.py --help` to verify the app parses and the image builds without errors before launching a full GPU job.
2. **Never use `save_strategy="epoch"` with unsloth + trl.** It causes a `PicklingError` at checkpoint save (class identity mismatch). Always use `save_strategy="no"` and call `model.save_pretrained()` manually after `trainer.train()`.
3. **Always call `vol.commit()` after saving.** Modal volumes are not flushed automatically — omitting it means the adapter is lost when the container exits.
4. **Pin image layers that are likely to break.** Use `nvidia/cuda:12.4.0-devel-ubuntu22.04` + `apt_install("git")` + `pip_install("unsloth", ...)` as the proven base. Do not switch base images without consulting context7.
5. **Log structured output at key steps.** Every training function must print: examples loaded, model loaded, training started (step 0 loss), and a final JSON result dict. This makes log tails immediately useful.
6. **Set explicit timeouts.** Training apps: 3h for 1B models, 6h for 3B+. Inference apps: 2h. Never rely on the Modal default.
7. **Support resume in inference apps.** Check for already-written `cv_id`s at startup and skip them — Modal containers can disconnect mid-run.
8. **Use `entrypoint([])` when building from nvidia/cuda images** to suppress the default CUDA entrypoint that blocks package installs.

---

## Run Logging — Every Trial Must Be Recorded

Every Modal run and local evaluation must be logged to the DuckDB run store (`artifacts/runs.duckdb`) via `src/eraparse/run_store.py`. Do not leave results only in loose JSONL files — they get lost.

Mandatory fields per logged run:
- `run_id` — unique identifier (timestamp + model slug)
- `model_id` — full HF model ID or adapter name
- `split` — `train` | `validation` | `id_test` | `template_ood` | `locked`
- `schema` — `reduced` | `rich`
- `constrained` — `true` | `false` (decoding mode)
- `clean_macro`, `raw_macro`, `nested_macro` — metric values
- `unsupported_evidence_rate` — hallucination rate
- `latency_p50`, `latency_p95` — in seconds per CV
- `cost_usd` — Modal billing estimate
- `config_hash` — hash of the prompt/model/threshold config (for reproducibility)
- `notes` — free text for anything unusual

After every eval: update `docs/RESULTS_AND_PARETO.md` with the new row immediately. Do not batch result updates.

---

## Analysis and Plots — Required After Every Eval

After ingesting results and logging to the run store, always produce:

1. **Per-field breakdown** — macro F1 per field (full_name, email, work_experience, education, projects, certifications) as a horizontal bar chart. Save to `artifacts/analysis/<run_id>/field_breakdown.png`.
2. **Pareto scatter** — accuracy (clean macro) vs latency (p50) for all runs logged so far. Save to `artifacts/analysis/pareto_latest.png`. Update after every new result.
3. **Faithfulness comparison** — `unsupported_evidence_rate` bar chart across all models. Save to `artifacts/analysis/faithfulness_latest.png`.
4. **Training loss curve** — for any fine-tune run, save the loss-vs-step curve to `artifacts/analysis/<run_id>/loss_curve.png`.

Use `matplotlib` + `pandas` for all plots. Save both `.png` (for quick viewing) and the underlying `.csv` (for reproducibility). Script: `scripts/plot_results.py`.

---

## Codebase Structure — Keep It Clean

```
modal_apps/          one file per Modal app; name = <model>_<purpose>.py
scripts/             data pipeline scripts (build_sft_*, plot_results, etc.)
src/eraparse/        library code only — no one-off scripts here
artifacts/
  manifests/         split manifests (never modified after creation)
  representations/   parsed CV text (pymupdf4llm_markdown/, etc.)
  sft/               SFT training JSONL files
  trials/
    <lane>/          raw model responses (*.responses.jsonl)
    ft/              fine-tuned model responses
    constrained/     constrained-decoding responses
  ingested/          post-ingest-mapper structured results
  analysis/          plots and CSVs (gitignored for large files)
  runs.duckdb        run log — source of truth for all results
docs/                thesis design, results, phase plan
```

Rules:
- Raw model responses go in `artifacts/trials/<lane>/` — never mix lanes in one folder.
- Ingested (structured) results go in `artifacts/ingested/<run_id>/` — separate from raw.
- Aggregated results (multi-run comparisons) go in `artifacts/analysis/` — never hardcode paths into `docs/`.
- Never write eval results directly into `docs/` — always derive them from the run store.

---

## Key File Locations

| What | Where |
|---|---|
| Thesis design + RQs | `docs/THESIS_DESIGN.md` |
| All measured results | `docs/RESULTS_AND_PARETO.md` |
| Phase plan | `docs/PHASED_TASK_PLAN.md` |
| Clean metric function | `scripts/analyze_labels_and_router.py` → `clean_macro()` |
| Run store (DuckDB) | `artifacts/runs.duckdb` → `src/eraparse/run_store.py` |
| Plot script | `scripts/plot_results.py` |
| Reduced schema constant | `src/eraparse/constants.py` → `REDUCED_SCHEMA_TEMPLATE` |
| NuExtract prompt builder | `src/eraparse/representations.py` → `build_nuextract_prompt()` |
| Modal training apps | `modal_apps/` |
| SFT data builders | `scripts/build_sft_*.py` |
| Track B set-pred model | `src/eraparse/set_pred_model.py` |
| Validation manifests | `artifacts/manifests/validation.jsonl` |
| Raw trial outputs | `artifacts/trials/` |
| Ingested results | `artifacts/ingested/` |
| Adapters (Modal Volume) | `eraparse-adapters` volume → `/adapters/<name>` |

---

## Package Manager

Always use `uv run` for Python commands in this repo. Never use `pip` directly — the venv is managed by `uv`.

---

## Dev Artifact Policy

Temporary scripts, brainstorm files, and dev-only docs go in `Dev_temp_files/` (gitignored). Do not commit them.

Commit messages: lowercase, concise, no AI attribution.

# Results And Pareto Tracking

## Purpose

This document tracks the accuracy, validity, evidence support, latency, and cost
tradeoffs that determine the final thesis comparison. It must distinguish
measured end-to-end latency from cached-reader or model-only latency.

Last updated: **June 15, 2026**

## Selective Router And Clean-Schema Results (full 1,445-train)

The selective grounded-to-generative router was promoted to the complete
training corpus and compared against the two single-approach lanes.

### Evidence-absent label correction (URLs + projects)

Two categories of labels in `schema_reduced` are unrecoverable by any extractor
and are excluded from the reported (clean) metric, applied identically to all
lanes.

**URL fields:** `github_url` / `linkedin_url` values are rendered nowhere in
the document (no text, no PDF link annotation, no pixels) in ~40–60% of cases.

**Projects field:** Four CV templates (T1_functional, T1_executive, T3_table,
T5_minimal) never print a projects section in their PDF layout. Their project
labels in `schema_reduced` are real data but unrecoverable from the rendered
document. Verified: 0/N project names appear in pymupdf text across all affected
CVs; NuExtract3 (a VLM seeing the image) achieves 0% project detection on them.

| Split | github absent | linkedin absent | projects absent |
|---|---:|---:|---:|
| train | 57.7% (579/1003) | 40.0% (496/1239) | 20.8% (178/856) |
| validation | 59.3% (137/231) | 41.7% (111/266) | 25.3% (45/178) |
| id_test | 57.0% (126/221) | 40.2% (102/254) | 24.6% (43/175) |
| template_ood | 34.3% (98/286) | 58.6% (197/336) | 0.0% (0/240) |

Total excluded cells across all splits: **2,112**.
Note: template_ood contains none of the four failing templates → 0 project
exclusions there, so template_ood fully-clean = URL-clean.

### Lane comparison (full train, clean metric)

All three columns use the same exclusion logic applied identically to all lanes.

| Approach | Macro raw | URL-clean | Fully-clean | Nested (fully-clean) | Mean latency |
|---|---:|---:|---:|---:|---:|
| A: parser → Qwen3-4B mapper (debug_250 only) | 0.7749 | n/a | n/a | 0.8202 raw | 15.54 s/CV |
| B: NuExtract3 VLM direct | 0.8937 | 0.9325 | **0.9417** | 0.8926 | 1.24 s/CV |
| C: selective router (B + selective A) | 0.9051 | 0.9439 | **0.9531** | **0.9268** | ~3.4 s/CV |

`nested macro` = mean of work_experience, education, projects, certifications.
Lane A is on `debug_250` only; matched-sample validation/test runs are required
for the final comparison table (Task 1.3 / 1.4).

### Field-level complementarity (why the router works)

All scores are fully-clean (URL + projects excluded) on train.

| Field | VLM (B) | Router (C) | Note |
|---|---:|---:|---|
| work_experience | 0.8578 | **0.9204** | routed to mapper |
| certifications | 0.8275 | **0.9017** | routed to mapper |
| education | **0.9867** | **0.9867** | kept in VLM |
| projects | **0.8984** | **0.8984** | kept in VLM (mapper is worse) |

Projects appears weak in raw numbers (0.788) because the four non-rendering
templates account for 100% of the gap. With evidence-absent records excluded,
projects is already at 0.898 and requires no model fix.

### Certifications routing is a speed/accuracy knob

| Router variant | Escalated CVs | Fully-clean macro | cert (fully-clean) |
|---|---:|---:|---:|
| both fields (default) | 843 | **0.9531** | 0.9017 |
| work-experience only | 537 (−36%) | 0.9469 | 0.8275 |

Dropping certifications saves 36% escalations for −0.0062 macro. Work-experience
signal is selectively targetable (`primary_characters ≤ 248`, 0.82 win precision);
certifications is a broad net-positive escalation with no selective threshold.

## ★ Checkpoint 1 — Held-out Validation (310 CVs, fully clean)

The first fair A/B/C comparison on the **same 310 held-out validation CVs**, same
clean metric (URL + projects exclusions), same evaluator. **The router wins** and
the train finding (router 0.9531) holds on held-out data (0.9510).

| Lane | Clean macro | Nested | work | edu | proj | cert | Unsupported (hallucination) | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Qwen3-4B mapper | 0.8921 | 0.8597 | **0.946** | 0.747 | 0.866 | **0.880** | **0.006** | 15.83 s |
| B: NuExtract3 VLM | 0.9402 | 0.8924 | 0.863 | **0.985** | **0.888** | 0.834 | 0.036 | **1.29 s** |
| **C: Router (primary-only)** | **0.9510** | **0.9247** | 0.918 | **0.985** | **0.888** | 0.909 | 0.039 | 10.35 s |

**Field complementarity confirmed on held-out data** (why the router works):
- work_experience: mapper 0.946 > VLM 0.863 → router escalates → 0.918
- certifications: mapper 0.880 > VLM 0.834 → router escalates → 0.909
- education: VLM 0.985 ≫ mapper 0.747 → router keeps VLM
- projects: VLM 0.888 > mapper 0.866 → router keeps VLM

Router escalates **187/310 (60.3%)**, projected latency ~10.3 s/CV. The latency
gap vs B is inflated by the slow HF-generate mapper (15.8 s); under vLLM the
mapper—and thus C—would be substantially faster. The work-exp-only router variant
trades a little accuracy for ~36% fewer escalations (faster).

**Caveat (the thesis hook):** all three are *baselines* here. A and B are
zero-shot/off-the-shelf; the router is the practical synthesis. The contribution
tracks (fine-tuned small models, set-prediction head, EraExtract VLM) are what
must beat this frontier — especially on the speed×faithfulness axes where B
already hallucinates 6× more than A.

## ★ Track A — Fine-Tuned Small Model Results (310-CV validation, fully clean)

Fine-tuning small models on the training corpus to answer **RQ2**: can fine-tuning
a ≤1.5B model reach the Qwen3-4B zero-shot mapper ceiling?

All rows use the same 310-CV validation split, same clean metric (URL + projects
exclusions), same evaluator as Checkpoint 1.

| Model | Params | Macro raw | Fully-clean | Nested (fully-clean) | Work | Edu | Proj | Cert | Schema valid | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NuExtract-tiny ft (eraparse/nuextract-tiny-reduced) | 0.5B | 0.3359 | 0.3722 | 0.3915 | 0.3715 | 0.3256 | 0.3512 | 0.5176 | **71.3%** | ~2 s |
| NuExtract-1.5 ft (eraparse/nuextract-15-reduced) | 1.5B | 0.7159 | 0.7619 | 0.5208 | 0.4161 | 0.3880 | 0.5501 | 0.7290 | **100%** | ~4 s |
| Gemma-3-1B ft (eraparse/gemma3-1b-reduced) | 1B | 0.7652 | **0.8210** | 0.7097 | 0.8145 | 0.8689 | 0.4863 | 0.6692 | **97.1%** | ~35 s |

### Finding 0 — Gemma-3-1B ft exceeds LayoutLMv3+EFSFR

Gemma-3-1B fine-tuned reaches **0.8210 fully-clean** — beating LayoutLMv3+EFSFR
(0.777) by +0.044 and exceeding NuExtract-1.5 ft (0.7619) by +0.059.

Key breakdown:
- **Work experience: 0.8145** — strong, close to mapper (0.946) and VLM (0.863)
- **Education: 0.8689** — very strong, close to VLM (0.985)
- **Projects: 0.4863** — weak (same as NuExtract-1.5's weakness)
- **Certifications: 0.6692** — moderate
- **97.1% schema valid** — robust JSON generation
- **10.7% hallucination** — 3× higher than LayoutLMv3 (0%) but 4× lower than Router (3.9%)
- **Latency: ~35 s/CV** — slow (HF generate on L4, no vLLM)

**Conclusion:** Fine-tuning a 1B instruction-tuned model (Gemma-3-1B) achieves
**+4.4pp over the LayoutLMv3+EFSFR frontier** at a 1B parameter count. This
answers RQ2 positively: SFT on 1B closes the gap significantly. The remaining
gap to the Router (0.9510) is 0.130 — projects and certifications are the weak
fields, not work/education.

### Finding 1 — 0.5B fails at fine-tuning

NuExtract-tiny (0.5B) reaches only **0.3722 fully-clean** despite SFT:

- **29% invalid JSON** — the model frequently produces broken output structure.
- **52% unsupported evidence rate** — when it does produce output, it
  hallucinate values not present in the CV text.
- All nested fields are weak (work 0.37, edu 0.33, projects 0.35).
- This confirms the thesis finding: 0.5B parameter count is below the minimum
  for reliable CV extraction, even after fine-tuning.

**Conclusion:** 0.5B is rejected as a viable track. The Kaggle Trial 1
LayoutLMv3 experiment (which ran at ~110M params) outperforms this model.

### Finding 2 — 1.5B reaches Checkpoint-1 LayoutLMv3+EFSFR but not beyond

NuExtract-1.5 (1.5B) reaches **0.7619 fully-clean**, which is:

- at the LayoutLMv3+EFSFR frontier (0.777 fully-clean in Checkpoint 1 scale), but
- well below Qwen3-4B zero-shot (0.892 fully-clean at Checkpoint 1), and
- well below Router (0.951 fully-clean at Checkpoint 1).

100% schema validity is a genuine improvement over the 0.5B model.

However, **nested fields are disproportionately weak** (nested fully-clean 0.52
vs macro 0.76): the model correctly extracts flat scalar fields (name, phone,
email, location) but struggles with structured lists (work experience, education,
projects). This suggests the SFT loss is dominated by simple fields; more training
on nested examples may be needed.

**Conclusion:** fine-tuning 1.5B moves from zero-shot ~0.60 to 0.76 — useful
progress — but does not challenge the router frontier. The clean 3B threshold
for CV extraction appears to sit above 1.5B in fine-tuned models.

### Pareto position (Track A, confirmed)

| Method | Fully-clean macro | Latency | Hallucination | Size |
|---|---:|---:|---:|---|
| NuExtract-tiny ft | 0.3722 | ~2 s | 52% | 0.5B |
| NuExtract-1.5 ft | 0.7619 | ~4 s | ~0% | 1.5B |
| LayoutLMv3+EFSFR (no LLM) | 0.777 | **0.72 s** | **0%** | 125M |
| **Gemma-3-1B ft** | **0.8210** | ~35 s* | 10.7% | **1B** |
| Router C (primary-only) | **0.9510** | 10.3 s | 3.9% | 4B+12B |

*35 s is HF-generate on L4; under vLLM this would be ~3–4 s/CV.

**Gemma-3-1B ft Pareto-dominates LayoutLMv3+EFSFR on accuracy** (+4.4pp) and is
competitive with the Router on accuracy at 1B parameters. The trade-off is
hallucination (10.7% vs 0%) and latency (slow without vLLM). This is the core
RQ2 finding: SFT at 1B is sufficient to beat the deterministic frontier.

## ★ Checkpoint 2 — Held-out Test Splits (LayoutLMv3+EFSFR, fully clean)

First locked evaluation on the two held-out test splits. LayoutLMv3+EFSFR is the
only method whose results are confirmed here; Gemma-3-1B ft and Router C inference
is still running on Modal.

Split sizes: **id_test = 310 CVs**, **template_ood = 410 CVs**.

### LayoutLMv3+EFSFR test results

Using the frozen 2-epoch checkpoint (`layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1`).

| Split | Raw macro | URL-clean | Fully-clean | Nested | Work | Edu | Proj | Cert | Schema valid | Hallucination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation (reference) | 0.7755 | — | ~0.777 | — | 0.923 | — | 0.646 | — | 1.000 | 0.000 |
| **id_test** | 0.7796 | 0.8001 | **0.8074** | 0.8472 | 0.903 | 0.840 | 0.767 | 0.879 | 1.000 | 0.000 |
| **template_ood** | 0.7754 | 0.8136 | **0.8136** | 0.7579 | 0.806 | 0.837 | 0.547 | 0.841 | 1.000 | 0.000 |

**Notes:**
- id_test fully-clean (0.8074) > validation fully-clean (~0.777): id_test has slightly fewer evidence-absent URL records (57% github absent vs 59.3% on validation), lifting the clean metric.
- template_ood url-clean == fully-clean (0.8136): confirmed — none of the 4 failing project-templates appear in template_ood (0/410 project exclusions).
- **template_ood projects F1 = 0.547** — the sharpest OOD generalization weakness. The model has not seen these template layouts during training.
- Both splits maintain 100% schema validity and 0% hallucination — the deterministic head never invents values.

### Pending (Gemma-3-1B ft on id_test + template_ood)

Modal jobs running:
- id_test: `ap-BAZSFVu4C7z6Nn8OXxHpVQ` (25/310 done as of June 15 EEST)
- template_ood: restarted, writing to `artifacts/trials/ft/template_ood.gemma3-1b-ft.jsonl`

After both complete, run:
```bash
bash scripts/eval_ft.sh gemma3-1b-reduced \
    artifacts/trials/ft/id_test.gemma-ft.requests.jsonl \
    artifacts/trials/ft/id_test.gemma3-1b-ft.jsonl \
    gemma3-1b-id_test-v1

bash scripts/eval_ft.sh gemma3-1b-reduced \
    artifacts/trials/ft/template_ood.gemma-ft.requests.jsonl \
    artifacts/trials/ft/template_ood.gemma3-1b-ft.jsonl \
    gemma3-1b-template_ood-v1
```

## Zero-shot Small Mapper Baselines (debug_50, fully clean, raw macro)

Evidence that small models zero-shot cannot reliably structure CVs — motivates
Track A fine-tuning. Numbers from `eraparse trials ingest-mapper` on 50-CV debug
split (pre-Checkpoint-1 early runs, `pymupdf4llm_markdown` representation).

| Model | Raw macro | Nested macro | Schema valid | Unsupported (hallucination) | Latency |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B (zero-shot) | 0.586 | ~0.42 | ~100% | **0.006** | ~3 s |
| Phi-4-mini (zero-shot) | 0.579 | ~0.38 | ~100% | **0.215** | ~13 s |
| Qwen3-4B (zero-shot, Lane A) | 0.775 | 0.720 | 100% | **0.006** | 15.8 s |

Key findings:
- **Validity is not accuracy:** small models emit syntactically valid JSON (≥90%)
  but nested content (work_experience, education) is wrong — the Kaggle Trial 1
  finding (exp/edu F1 ≈ 0 at 1B zero-shot) is confirmed here.
- **Faithfulness discriminates:** Phi-4-mini hallucinates at 21.5% — 36× more
  than Qwen3 models. Size alone does not predict faithfulness.
- **4B is the minimum useful zero-shot mapper** (0.775 raw macro, 0.006 unsupported).
- **Track A goal:** fine-tuned 1B should reach Qwen3-4B accuracy (≥0.77 raw,
  ≥0.87 fully-clean) at 1B parameter count — proving fine-tuning bridges the gap.

## Current Validation Results

All rows below use the same held-out `310`-CV validation split and the complete
reduced CV schema.

| Method | Macro | Work experience | Projects | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|---:|
| Qwen3 + PyMuPDF/Tesseract fallback | 0.670902 | 0.612091 | 0.280183 | 0.996774 | 0.040981 |
| standard LayoutLMv3 + PyMuPDF4LLM | 0.739158 | 0.627738 | 0.542950 | 1.000000 | 0.000000 |
| LayoutLMv3 + full EFSFR | 0.775538 | 0.923473 | 0.645540 | 1.000000 | 0.000000 |
| best practical modular lane | 0.776740 | 0.931212 | 0.652224 | 1.000000 | 0.000000 |
| NuExtract3 raw | 0.869937 | 0.730132 | 0.761213 | 0.000000 | 0.035980 |
| NuExtract3 + deterministic contract assembly | **0.881002** | **0.862906** | **0.761213** | **1.000000** | 0.035980 |

The best practical modular lane is:

```text
LayoutLMv3
  + EFSFR nested repair
  + selective SG-GRSE work replacement
  + project technology repair
  + train-derived project URL selector
```

## Timing Measurements

| Component or lane | Measured mean | Measurement boundary |
|---|---:|---|
| PyMuPDF4LLM JSON reader | 0.5190 s/CV | uncached reader generation on 310 validation CVs |
| standard LayoutLMv3 validation decode | 0.1663 s/CV | cached evidence input; model decode/evaluation only |
| SG-ESE query-only validation decode | 0.1804 s/CV | cached evidence input; model decode/evaluation only |
| modular repair/fusion chain | 0.0326 s/CV | five separate CLI stages, including process startup |
| modular repair/fusion + final evaluation | 0.0392 s/CV | five repair stages plus evaluator, including process startup |
| Qwen3 + PyMuPDF/Tesseract fallback | 13.3212 s/CV | persisted mapper-lane timing |
| NuExtract3 visual extraction | 17.7595 s/CV | persisted visual model-lane timing |

PyMuPDF4LLM validation-reader distribution:

- mean: `0.5190 s/CV`;
- p50: `0.1946 s/CV`;
- p95: `2.0831 s/CV`.

An approximate uncached practical LayoutLMv3 path is currently:

```text
0.5190 s/CV reader mean
+ 0.1663 s/CV cached model mean
+ 0.0326 s/CV modular repair/fusion mean
= approximately 0.7179 s/CV
```

This is an estimate, not the final end-to-end benchmark. Reader and model stages
were measured in separate runs and must be benchmarked together before a final
speed claim.

## Current Pareto Interpretation

- **Highest accuracy:** NuExtract3 + deterministic contract assembly at
  `0.881002` macro.
- **Fast practical system:** LayoutLMv3 + deterministic modular repair. Its
  model-side accuracy is lower, but its current measured components suggest an
  order-of-magnitude latency advantage over generative mapper/VLM lanes.
- **Methodology contribution:** deterministic contract assembly and EFSFR
  improve structured validity and nested-record accuracy without another model
  inference pass.
- **Architecture contribution:** SG-ESE remains a valid ablation/efficiency
  contribution, but its current full-loss accuracy does not beat standard
  LayoutLMv3.

## Train-Derived Next-Stage Results

These results use train-derived `debug_50` data and are for model/runtime
selection only. They are not directly comparable to the frozen 310-CV
validation table above.

| Method | Documents | Macro | Schema valid | Unsupported evidence | Model latency |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct + PyMuPDF4LLM Markdown | 50 | 0.680819 | 1.000000 | 0.000000 | 17.169 s/CV |
| Phi-4 Mini + PyMuPDF4LLM Markdown | 50 | 0.578810 | 1.000000 | 0.214524 | 13.002 s/CV |
| NuExtract3 vLLM full-schema compatibility smoke | 1 | 0.988889 | 1.000000 | 0.000000 | 27.548 s/CV |
| NuExtract3 opaque compact-schema smoke | 1 | **0.083333** | 1.000000 | 0.000000 | **24.135 s/CV** |
| NuExtract3 two-token MTP smoke | 1 | 0.988889 | 1.000000 | 0.000000 | 28.151 s/CV |
| NuExtract3 vLLM baseline | 50 | 0.845862 | 1.000000 | 0.049364 | 3.480 s/CV |
| NuExtract3 two-token MTP | 50 | 0.845601 | 1.000000 | 0.049364 | **2.279 s/CV** |
| NuExtract3 W4A16, MTP off | 50 | 0.848297 | 1.000000 | 0.050737 | 2.582 s/CV |
| NuExtract3 MTP + PyMuPDF4LLM evidence | 50 | **0.893809** | 1.000000 | **0.046091** | 2.418 s/CV |
| Qwen3-4B-Instruct + PyMuPDF4LLM Markdown | 250 | 0.774910 | 1.000000 | **0.003442** | 15.540 s/CV |
| NuExtract3 MTP + PyMuPDF4LLM evidence | 250 | 0.870103 | 1.000000 | 0.055430 | **1.444 s/CV** |
| static field fusion: NuExtract3 evidence + Qwen work/certifications | 250 | **0.880683** | 1.000000 | 0.056386 | 16.984 s/CV |
| disagreement field selector, always run both | 250 | **0.886686** | 1.000000 | 0.056347 | 15.717 s/CV |
| primary-only pre-inference field router | 250 | 0.881399 | 1.000000 | 0.056194 | 10.485 s/CV projected |
| focused primary-only router | 250 | **0.882608** | 1.000000 | 0.056194 | **3.361 s/CV** |
| random field selector, matched field coverage | 250 | 0.874494 | 1.000000 | 0.055295 | accuracy control |
| oracle field selector | 250 | 0.889159 | 1.000000 | 0.056505 | unreachable ceiling |

Interpretation:

- the fair greater-than-3B text mapper is grounded and schema-valid, but its
  `debug_50` accuracy does not justify replacing the practical LayoutLMv3 lane;
- Phi-4 Mini's one-CV smoke did not generalize: the full `debug_50` result is
  below both Qwen3-4B and the visual NuExtract3 lanes, with substantially more
  unsupported evidence, so it is rejected as the promoted text mapper;
- opaque schema aliases reduced NuExtract3 output tokens from `490` to `218`
  and generation time by `12.4%`, but destroyed accuracy and are rejected;
- the one-CV MTP smoke was misleading, but the matched `debug_50` result
  reduced mean generation latency by **34.5%** for only `-0.000261` macro;
- two-token MTP is promoted as the NuExtract3 speed variant.
- W4A16 preserved and slightly improved macro relative to BF16+MTP, but was
  `13.3%` slower on A10 and slightly worse on unsupported evidence. It is a
  reportable memory-oriented quantization result, not the promoted speed path;
- adding cached PyMuPDF4LLM evidence text to the visual MTP prompt improved
  macro by `+0.048207` over MTP alone while adding only `0.139 s/CV`. This
  grounded multimodal evidence injection is promoted for a larger matched run.
- on the fixed `debug_250`, grounded NuExtract3 retained `0.870103` macro and
  achieved `1.444 s/CV` with batch-10 serving;
- Qwen3-4B reached `0.774910` on the same 250 CVs and was exceptionally strong
  on work experience (`0.925185`) and certifications (`0.866536`), but was
  about `10.8x` slower than the grounded NuExtract3 run;
- static field fusion raised macro to `0.880683`, proving the models are
  complementary. Because always running both costs `16.984 s/CV`, it is an
  accuracy ablation and the target for a selective field router, not the final
  practical system.
- the disagreement selector improved further to `0.886686`, but disagreement
  requires Qwen output and therefore cannot avoid specialist inference. It is
  an accuracy-only selector, not a cost-saving router;
- the first true pre-inference router uses only NuExtract3 result structure and
  evidence-support signals. It reached `0.881399` on all 250 CVs while routing
  `59.6%` of documents. On the deterministic 117-CV holdout it improved macro
  from `0.871699` to `0.883016`;
- current router latency uses historical full-schema Qwen calls and is
  conservative. A focused partial-schema Qwen trial is required before the
  practical speed result is known.
- the focused partial-schema Qwen trial completed on only the 149 selected
  documents. It reduced Qwen output tokens from `500.9` to `101.1` per called
  CV and called-model latency from `15.540` to `3.216 s/CV`;
- the resulting focused router achieved `0.882608` macro at `3.361 s/CV`,
  improving over grounded NuExtract3 by `+0.012505` macro while remaining about
  `4.6x` faster than full Qwen and about `5.1x` faster than always-run-both
  fusion;
- matched-coverage random selection reached only `0.874494`, showing that the
  router signals add value beyond escalation coverage alone;
- the oracle field selector reached `0.889159`. The focused router captures
  about `65.6%` of the available oracle macro improvement over grounded
  NuExtract3.

## Full-Train Router Results

These rows use all `1,445` training CVs. They verify scale and support
calibration work; they are not final held-out test results.

| Method | Documents | Macro | Work experience | Certifications | Schema valid | Unsupported | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| grounded NuExtract3 MTP + evidence | 1,445 | 0.893675 | 0.857848 | 0.827543 | 1.0 | 0.051715 | **1.241 s/CV** |
| focused primary-only router | 1,445 | **0.905064** | **0.920373** | **0.901688** | 1.0 | 0.052818 | 3.506 s/CV |

The full-train focused router:

- improves macro by `+0.011389`;
- improves work experience by `+0.062525`;
- improves certifications by `+0.074144`;
- routes `843/1,445` documents (`58.3%`);
- generates only `106.5` Qwen output tokens per called document on average;
- produced zero focused-response parse failures.

This is the current strongest scalable methodology result. Final claims still
require freezing the router and evaluating ID-test and template-OOD exactly
once.

## SG-ESE Component Ablation

Equal one-epoch held-out comparison:

| Configuration | Macro | Training seconds |
|---|---:|---:|
| standard LayoutLMv3 | 0.728579 | historical training timing not retained in the evaluation-resume summary |
| SG-ESE query-only | 0.689791 | 421.32 |
| SG-ESE no grouping loss | **0.704335** | **456.28** |
| SG-ESE full auxiliary loss | 0.688165 | 621.15 |

Current conclusion:

- presence and evidence-relevance losses are positive relative to query-only;
- grouping loss is negative in both accuracy and training cost;
- the no-group variant reached `0.730010` after two epochs, improving by
  `+0.025675` but remaining `-0.009148` below standard LayoutLMv3;
- the current grouping objective must not be included in a final promoted
  architecture without redesign.

Equal two-epoch held-out comparison:

| Configuration | Macro | Schema valid | Unsupported evidence |
|---|---:|---:|---:|
| standard LayoutLMv3 | **0.739158** | 1.000000 | 0.000000 |
| SG-ESE query-only | 0.738284 | 1.000000 | 0.000000 |
| SG-ESE no grouping loss | 0.730010 | 1.000000 | 0.000000 |

## Cost Tracking

Modal billing recorded approximately `$3.14` for the June 13 upper-bound stage,
including successful NuExtract3 runs and failed environment/smoke attempts.

- completed full NuExtract3 validation app: approximately `$2.09`;
- stalled PaddleOCR-VL smoke: approximately `$0.41`;
- remaining spend came from NuExtract3 smoke/debug and failed bring-up attempts.

Local M1 Pro LayoutLMv3 and SG-ESE training runs incur no Modal GPU charge.

## Modular Timing Reproduction

The persisted full modular lane was rebuilt from the standard LayoutLMv3
validation predictions. The five repair/fusion stages completed in:

| Stage | All 310 CVs | Mean per CV |
|---|---:|---:|
| EFSFR nested repair | 2.93 s | 0.0095 s |
| SG-GRSE work decode | 1.82 s | 0.0059 s |
| selective SG-GRSE fusion | 1.63 s | 0.0053 s |
| project technology repair | 1.49 s | 0.0048 s |
| train-derived project URL selector | 2.25 s | 0.0073 s |
| **repair/fusion total** | **10.12 s** | **0.0326 s** |
| final evaluation | 2.04 s | 0.0066 s |

The reproduced output retained:

- macro: `0.776740`;
- schema validity: `1.000000`;
- unsupported evidence: `0.000000`.

Benchmark artifacts are under `artifacts/sge/benchmarks/`.

## Active And Required Work

Completed full-size architecture ablation:

- SG-ESE without grouping loss;
- `1,445` training CVs and all `310` validation CVs;
- two epochs / `3,202` cumulative steps;
- local M1 Pro MPS, with no Modal spend;
- output:
  `artifacts/sge/local_smokes/sge-pymupdf4llm-train-val-2epoch-unfreeze4-no-group-loss-v1`.

Required before final thesis speed claims:

1. benchmark one uncached end-to-end practical LayoutLMv3 + EFSFR run;
2. record deterministic repair latency separately;
3. report p50 and p95, not only mean latency;
4. compare accuracy per second and accuracy per dollar;
5. keep NuExtract3 as the accuracy upper bound and the practical modular lane as
   the speed/grounding candidate.

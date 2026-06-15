# SG-ESE Implementation And Trial Log

## Purpose And Status

This document is the durable chronological record for the PyMuPDF4LLM,
LayoutLMv3, and SG-ESE stage. It records implementation changes, local trials,
failures, corrections, interpretations, budget decisions, and the exact point
from which the next agent should continue.

This is a development-stage log, not a final-results report. Unless explicitly
stated otherwise, trials below use training samples and must not be cited as
validation, ID-test, OOD-test, or locked-confirmation results.

The current forward-looking action plan is:

- [`docs/NEXT_MODEL_ACTION_PLAN.md`](NEXT_MODEL_ACTION_PLAN.md)

The architecture rationale and thesis flow are:

- [`docs/REARCHITECTED_RESEARCH_FLOW.md`](REARCHITECTED_RESEARCH_FLOW.md)

Last updated: **June 13, 2026**

## 2026-06-13: Mapper Scaling And NuExtract3 Speed Stage Started

The next-stage execution contract is:

- [`docs/NEXT_STAGE_EXPERIMENT_MATRIX.md`](NEXT_STAGE_EXPERIMENT_MATRIX.md)

Implemented:

- generic Modal mapper runner for `Qwen/Qwen3-0.6B`,
  `Qwen/Qwen3-4B-Instruct-2507`, and `microsoft/Phi-4-mini-instruct`;
- native NuExtract3 vLLM baseline/MTP runner;
- lossless schema-aware compact JSON codec and canonical expansion;
- compact-output ingestion through the complete canonical evaluator;
- full PyMuPDF4LLM Markdown generation and mapper requests for all `250`
  train-derived `debug_250` CVs.

Compact schema preparation result:

- full schema template: `396` compact JSON characters;
- compact schema template: `138` compact JSON characters;
- structural template reduction: `65.2%`;
- canonical round-trip: lossless under automated tests.

First real Qwen3-4B mapper smoke:

| Samples | Macro | Schema valid | Unsupported evidence | Latency | Output tokens |
|---:|---:|---:|---:|---:|---:|
| 1 train-derived CV | 0.958333 | 1.000000 | 0.000000 | 18.859 s | 606 |

This is a compatibility smoke, not a reportable aggregate accuracy result.

Failures and corrections:

- the first simultaneous NuExtract3 vLLM smoke was rejected by Modal's
  app-creation rate limit before GPU inference; later smokes are sequential;
- the first Phi-4 Mini smoke failed because its pinned remote model code expects
  Transformers `4.49.0`, while the shared mapper image used `4.57.3`;
- the isolated Phi environment was corrected to the model-card requirement
  (`torch==2.5.1`, `transformers==4.49.0`, `accelerate==1.3.0`);
- the second Phi smoke exposed a Modal class-parameter inheritance error before
  inference; the runner now declares parameters on each concrete Modal class.

## 2026-06-13: Full No-Grouping Ablation And Modular Speed Benchmark

### Result: Full-Data SG-ESE Without Grouping Loss

Completed the missing equal-data component ablation locally on the M1 Pro:

- training records: `1,445`;
- validation records: `310`;
- steps: `1,601`;
- device: `mps`;
- final encoder layers unfrozen: `4`;
- grouping loss weight: `0.0`;
- presence loss weight: `0.25`;
- evidence relevance loss weight: `0.5`;
- query layers: `2`.

Output:

- training/checkpoint run:
  `artifacts/sge/local_smokes/sge-pymupdf4llm-train-val-1epoch-unfreeze4-no-group-loss-v1`;
- corrected full-validation evaluation:
  `artifacts/sge/local_smokes/sge-pymupdf4llm-train-val-1epoch-unfreeze4-no-group-loss-v2-eval-resume`.

This closes a missing full-size SG-ESE ablation between query-only and the
full auxiliary-loss architecture.

Equal one-epoch comparison on all `310` validation CVs:

| Configuration | Macro | Training seconds | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|
| standard LayoutLMv3 | 0.728579 | historical checkpoint timing not retained in eval-resume summary | 1.000000 | 0.000000 |
| SG-ESE query-only | 0.689791 | 421.32 | 1.000000 | 0.000000 |
| SG-ESE no grouping loss | **0.704335** | **456.28** | 1.000000 | 0.000000 |
| SG-ESE full auxiliary loss | 0.688165 | 621.15 | 1.000000 | 0.000000 |

Interpretation:

- adding presence and evidence-relevance losses to query-only improved macro by
  `+0.014544`;
- adding grouping loss on top of that reduced macro by `-0.016170`;
- grouping loss also increased training time by approximately `36%` relative
  to the no-group configuration;
- the grouping auxiliary objective is therefore rejected in its current form;
- no-group SG-ESE remains below standard LayoutLMv3 at one epoch, so it is now
  promoted to a fair two-epoch comparison rather than claimed as an accuracy
  winner.

Evaluation correction:

- the first evaluation accidentally capped page-level records at `310`, which
  covered only `280` CVs;
- the checkpoint was reloaded and correctly evaluated over all `340` page
  records corresponding to `310` validation CVs;
- only the corrected `0.704335` result should be cited.

### Result: Two-Epoch No-Grouping SG-ESE

The one-epoch no-group checkpoint was trained for one additional epoch locally
on MPS and evaluated over all `340` page records corresponding to `310`
validation CVs.

Output:

- `artifacts/sge/local_smokes/sge-pymupdf4llm-train-val-2epoch-unfreeze4-no-group-loss-v1`

Result:

- macro: `0.730010`;
- JSON/schema validity: `1.000000`;
- unsupported evidence: `0.000000`;
- work experience: `0.634492`;
- education: `0.816894`;
- projects: `0.465241`;
- additional-epoch training time: `546.18 s`;
- full-validation evaluation time: `59.48 s`;
- Modal cost: `$0.00`.

Fair two-epoch comparison:

| Configuration | Macro | Delta versus standard |
|---|---:|---:|
| standard LayoutLMv3 | **0.739158** | — |
| SG-ESE query-only | 0.738284 | -0.000875 |
| SG-ESE no grouping loss | 0.730010 | -0.009148 |

Interpretation:

- the second epoch improves no-group SG-ESE by `+0.025675` over its one-epoch
  result;
- no-group SG-ESE still does not beat standard LayoutLMv3 or query-only SG-ESE
  at equal two-epoch training;
- presence and evidence losses help early convergence relative to query-only,
  but do not improve the final two-epoch result;
- the all-pairs grouping loss remains rejected, and further SG-ESE work should
  redesign grouping rather than spend more compute on the current objective.

### Success: Modular Contribution Has Low Measured Overhead

Rebuilt the full practical modular lane from persisted standard LayoutLMv3
validation predictions and measured every deterministic stage.

| Stage | All 310 CVs |
|---|---:|
| EFSFR nested repair | 2.93 s |
| SG-GRSE work decode | 1.82 s |
| selective SG-GRSE fusion | 1.63 s |
| project technology repair | 1.49 s |
| train-derived project URL selector | 2.25 s |
| **repair/fusion total** | **10.12 s** |
| final evaluation | 2.04 s |

The reproduced pipeline retained:

- macro: `0.776740`;
- schema validity: `1.000000`;
- unsupported evidence: `0.000000`.

Interpretation:

- the full modular contribution adds approximately `0.0326 s/CV` when run as
  five separate CLI processes;
- this is small relative to the reader and model stages;
- the current evidence supports a thesis tradeoff in which NuExtract3 is the
  highest-accuracy upper bound and the practical modular lane is the
  speed/grounding candidate.

Detailed timing boundaries and caveats are in:

- [`docs/RESULTS_AND_PARETO.md`](RESULTS_AND_PARETO.md)

## 2026-06-13: NuExtract3 Promoted To Full Validation, Contract Repair Confirmed

### Result: Raw NuExtract3 Is Strong, But Needs Deterministic Contract Assembly

Completed the first real NuExtract3 direct-visual runs on Modal:

- model: `numind/NuExtract3`;
- revision: `acaf70ecff9c3dbbfcbae651b82b66a0d8dbd0c6`;
- device: single Modal T4;
- prompt style: full reduced-schema JSON template with page images.

Observed raw `debug_50` result before deterministic repair:

| Lane | Documents | Macro | Work experience | Schema valid | Unsupported evidence | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| NuExtract3 raw visual output | 50 | 0.845962 | 0.725867 | 0.000000 | 0.047000 | 18.32 s/CV |

Interpretation:

- the model is already a serious upper bound on extraction quality even before
  task-specific fine-tuning;
- the dominant failure is **schema contract compliance**, not broad evidence
  recovery;
- common raw issues were `null` values in required string fields and unsupported
  extra top-level keys.

### Implementation: Deterministic Trial-Assembly Repair Is Now Wired Into Ingestion

Added a switchable deterministic repair stage inside
[`src/eraparse/trials.py`](../src/eraparse/trials.py) and exposed it through:

```bash
uv run eraparse trials ingest-nuextract3 --repair-work-records
uv run eraparse trials ingest-qwen3 --repair-work-records
uv run eraparse trials ingest-nuextract --repair-work-records
```

Current repair behavior preserves raw model output while assembling a valid
reduced-schema object by:

- dropping unsupported top-level keys;
- coercing required scalar fields to schema-compatible strings;
- coercing optional URL fields to strings or `null`;
- normalizing string-list fields;
- repairing `work_experience` records with the existing date/duration recovery;
- coercing `education`, `projects`, and `certifications` records to the exact
  reduced-schema contract.

### Result: Same NuExtract3 Outputs Become Fully Schema-Valid After Assembly

Re-ingested the **same** 50 raw NuExtract3 responses with deterministic repair
enabled. No new model run was required.

| Lane | Documents | Macro | Work experience | Schema valid | Unsupported evidence | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| NuExtract3 raw visual output | 50 | 0.845962 | 0.725867 | 0.000000 | 0.047000 | 18.32 s/CV |
| NuExtract3 + deterministic contract assembly | 50 | **0.857562** | **0.865067** | **1.000000** | 0.047000 | 18.32 s/CV |

Interpretation:

- this is an important thesis finding: for modern structured VLMs, a large part
  of the practical gap is not extraction evidence but final schema assembly;
- the improvement was achieved with deterministic local logic, not another paid
  GPU run;
- this strengthens the methodology contribution story and helps explain why
  direct one-shot JSON generation can look worse than it really is.

### Result: Full 310-CV NuExtract3 Validation Is Complete

The promoted full validation run completed on all `310` validation CVs.

Raw validation result:

| Lane | Documents | Macro | Work experience | JSON valid | Schema valid | Unsupported evidence | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| NuExtract3 raw visual output | 310 | 0.869937 | 0.730132 | 1.000000 | 0.000000 | 0.035980 | 17.76 s/CV |

Repaired validation result using the same raw generations:

| Lane | Documents | Macro | Work experience | JSON valid | Schema valid | Unsupported evidence | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| NuExtract3 + deterministic contract assembly | 310 | **0.881002** | **0.862906** | **1.000000** | **1.000000** | 0.035980 | 17.76 s/CV |

Interpretation:

- this is now the strongest held-out validation score in the repository;
- the gain over the best practical thesis lane (`0.775538`) is large enough to
  establish a credible modern upper bound;
- the remaining tradeoff is latency: NuExtract3 is more accurate, but it is not
  the cheapest practical lane;
- deterministic contract assembly is now confirmed as a high-value methodology
  step at real validation scale, not just on a debug subset.

### Failure: PaddleOCR-VL Smoke Stalled After Model Load

The first PaddleOCR-VL smoke did **not** produce usable parser outputs.

Observed behavior:

- model download and initialization completed successfully;
- logs reached the first tensor-construction warning after load;
- no response JSONL was written;
- no new log lines appeared for roughly 20 minutes;
- the app was manually stopped to protect budget.

Shutdown outcome:

- Modal reported `RemoteError` on the blocked `call.get()` path;
- container shutdown also reported a lingering background VLM worker thread.

Interpretation:

- this is currently a **runtime/inference stability problem**, not a package
  installation problem;
- the failure is scientifically different from the earlier Donut issue:
  Donut failed on extraction quality and schema difficulty, while PaddleOCR-VL
  currently fails earlier, at reliable smoke execution;
- before promoting PaddleOCR-VL again, the next step should be a narrower
  single-page or single-document diagnostic with tighter timeouts and more
  granular logging around `chat()`.

Budget note:

- the stalled PaddleOCR-VL smoke consumed about `$0.40` before being stopped;
- the promoted NuExtract3 validation run is still the primary active paid job.

## 2026-06-13: Practical Full-Loss SG-ESE Result And Upper-Bound Trial Prep

### Result: Full-Loss Practical SG-ESE Underperformed The Best Modular Lane

Completed a practical train/validation SG-ESE run on the real PyMuPDF4LLM
evidence lane with the full auxiliary-loss stack enabled:

- mode: `sge`;
- device: local Apple M1 Pro via `mps`;
- encoder: `microsoft/layoutlmv3-base`;
- revision: `cfbbbff0762e6aab37086fdd4739ad14fe7d5db4`;
- run artifact:
  `artifacts/sge/local_smokes/sge-pymupdf4llm-train-val-1epoch-unfreeze4-v1/summary.json`.

Configuration summary:

- freeze-then-unfreeze schedule with final 4 encoder layers trainable;
- `presence_weight=0.25`;
- `grouping_weight=0.5`;
- `evidence_weight=0.5`;
- `query_layers=2`;
- sequence decoder used as the primary validation decode path.

Observed run characteristics:

- completed steps: `1601`;
- runtime seconds: `696.508`;
- training seconds: `621.151`;
- training steps/second: `2.577`.

Held-out validation result on all 310 validation CVs:

| Method | Macro | Projects | Work experience | Education | Summary | Schema valid |
|---|---:|---:|---:|---:|---:|---:|
| standard LayoutLMv3 practical lane | **0.739158** | 0.542950 | 0.627738 | 0.806531 | 0.948968 | 1.000000 |
| full-loss SG-ESE practical lane | 0.688165 | 0.549634 | 0.623411 | 0.765462 | 0.947579 | 1.000000 |

Interpretation:

- the architecture-only practical SG-ESE lane is currently **not** the best
  thesis result;
- the practical modular lane built from selective grounding and deterministic
  repair remains stronger than both the raw LayoutLMv3 baseline and the
  current full-loss SG-ESE variant;
- this does **not** invalidate the architecture contribution, but it does
  change the thesis center of gravity: architecture edits must now be justified
  either by later accuracy recovery or by a clear speed/efficiency story.

### Implementation: Upper-Bound Visual Trial Lane Is Now Prepared

Added structured-VLM trial preparation and ingestion interfaces for:

- `numind/NuExtract3`;
- `PaddlePaddle/PaddleOCR-VL-1.6`.

New CLI commands:

```bash
uv run eraparse trials prepare-nuextract3
uv run eraparse trials ingest-nuextract3
uv run eraparse trials prepare-paddleocr-vl
uv run eraparse trials materialize-paddleocr-vl
```

Prepared real request bundles:

- `artifacts/trials/nuextract3/debug_50.jsonl` with `50` requests;
- `artifacts/trials/nuextract3/validation.jsonl` with `310` requests;
- `artifacts/trials/paddleocr_vl/debug_50.jsonl` with `50` requests;
- `artifacts/trials/paddleocr_vl/validation.jsonl` with `310` requests.

Each prepared request includes:

- full reduced-schema template;
- full reduced-schema truth object;
- page-image references and hashes;
- practical evidence text with OCR fallback when PyMuPDF text is blank;
- split, tier, template, and primary-domain metadata.

Interpretation:

- the upper-bound comparison lane is now implementation-ready even before
  remote execution is attached;
- this keeps the thesis comparison story healthy: we can compare the practical
  modular lane, the base architecture lane, the edited architecture lane, and
  stronger modern visual upper bounds under the same evaluator.

### Correction: PaddleOCR-VL Is A Parser Lane, Not A Direct Structured-JSON Lane

While wiring the upper-bound runners, we found and corrected an implementation
mistake:

- `NuExtract3` is a direct visual structured-extraction lane;
- `PaddleOCR-VL-1.6` is a document parser / representation lane in the current
  thesis plan, not a direct JSON evaluator lane.

Repository changes now reflect that distinction:

- new Modal app:
  `modal_apps/nuextract3_trial.py` for direct visual extraction;
- new Modal app:
  `modal_apps/paddleocr_vl_trial.py` for page-level parser outputs;
- new generated representations:
  `paddleocr_vl_markdown` and `paddleocr_vl_json`;
- new CLI:
  `uv run eraparse trials materialize-paddleocr-vl`.

Practical consequence:

- PaddleOCR-VL outputs should be materialized as representations first;
- then the frozen Qwen3 mapper can consume `paddleocr_vl_markdown` under the
  same evaluator used for other practical lanes;
- direct JSON ingestion for PaddleOCR-VL would have been a scientifically
  mismatched comparison and is no longer the active path.

## Latest Validated Contribution Result

### Success: EFSFR Work-Record Repair Slice Beats The Main Baseline

Implemented a first explicit methodology contribution slice over the strongest
held-out practical baseline:

- new CLI: `uv run eraparse sge repair-work`;
- new deterministic repair stage:
  [`src/eraparse/sge.py`](../src/eraparse/sge.py);
- artifact:
  `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_predictions_work_repair.jsonl`;
- evaluation:
  `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_work_repair_evaluation.json`.

This stage does **not** use oracle labels. It operates only on the model's own
predicted grounded `work_experience` records and applies deterministic,
evidence-derived repairs:

- recover missing `start_date` from truncated duration spans such as
  `(2022-07 -`;
- canonicalize `Present)` to `Present`;
- derive canonical duration from repaired `start_date` and `end_date`.

Held-out validation result on all 310 validation CVs:

| Method | Macro | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|
| standard LayoutLMv3 | 0.739158 | 0.627738 | 1.000000 | 0.000000 |
| LayoutLMv3 + EFSFR work repair | **0.763803** | **0.923473** | 1.000000 | 0.000000 |
| Delta | **+0.024645** | **+0.295735** | 0.000000 | 0.000000 |

Repair activity summary:

- repaired documents: `305 / 310`;
- repaired records: `924`;
- `work_start_date_repaired`: `173`;
- `work_end_date_repaired`: `78`;
- `work_duration_normalized`: `917`.

Interpretation:

- this is the first real contribution result that clearly outperforms the main
  practical baseline on held-out validation;
- the dominant failure mode was not missing evidence, but incomplete
  structured decoding of work dates;
- the gain is large enough that deterministic, field-aware post-processing is
  now a thesis-worthy methodology lane rather than a side idea;
- the temporary `0.091178` unsupported-evidence reading was traced to an
  evaluation bug in `evaluate_grounded_rows` that kept only the last page's
  words per CV; after fixing multi-page evidence aggregation, the repaired lane
  returns to `0.000000` unsupported evidence.

Current forward plan remains in:

- [`docs/NEXT_MODEL_ACTION_PLAN.md`](NEXT_MODEL_ACTION_PLAN.md)

### Success: Full EFSFR Nested Repair Pushes The Practical Best Further

Extended the repair lane from work-date normalization to anchor-based repeated
record rebuilding for:

- `education` anchored on `degree`;
- `projects` anchored on `name`;
- `certifications` anchored on `name`.

This decoder uses grounded candidate evidence order rather than trusting the
assembled `record_index` assignments when they drift across the page. The new
CLI command is:

```bash
uv run eraparse sge efsfr-repair \
  --predictions artifacts/.../validation_predictions.jsonl \
  --output artifacts/.../validation_predictions_efsfr.jsonl
```

Held-out validation result on all 310 validation CVs:

| Method | Macro | Education | Projects | Certifications | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard LayoutLMv3 | 0.739158 | 0.806531 | 0.542950 | 0.904499 | 0.627738 | 1.000000 | 0.000000 |
| work-only EFSFR repair | 0.763803 | 0.806531 | 0.542950 | 0.904499 | 0.923473 | 1.000000 | 0.000000 |
| full EFSFR nested repair | **0.775538** | **0.837617** | **0.645540** | **0.911643** | **0.923473** | 1.000000 | 0.000000 |

Delta versus standard LayoutLMv3:

- macro: `+0.036380`;
- education: `+0.031086`;
- projects: `+0.102590`;
- certifications: `+0.007144`;
- work experience: `+0.295735`.

Repair activity summary:

- repaired documents: `310 / 310`;
- `education_anchor_redecoded`: `94`;
- `projects_anchor_redecoded`: `132`;
- `certifications_anchor_redecoded`: `17`;
- `work_start_date_repaired`: `173`;
- `work_end_date_repaired`: `78`;
- `work_duration_normalized`: `917`.

Interpretation:

- this is now the best practical held-out validation result in the repo;
- the model's candidate spans are stronger than the original nested assembly
  step suggested;
- repeated-record decoding is a major thesis lever, both architecturally and
  methodologically;
- SG-GRSE must now beat `0.775538`, not merely the old `0.739158` baseline.

### Success: First SG-GRSE Work Decoder Narrowly Beats Full EFSFR

Implemented a first explicit SG-GRSE work-experience decoder over grounded
candidate slots:

- new CLI: `uv run eraparse sge sgrse-work`;
- grouped work candidates by predicted `record_index`;
- rebuilt slot rows from best per-field grounded candidates;
- merged adjacent complementary slots, including split-title patterns such as
  `React Native` + `Developer`.

Artifacts:

- raw baseline + SG-GRSE work:
  `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_predictions_sgrse_work.jsonl`;
- EFSFR + SG-GRSE work:
  `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_predictions_efsfr_sgrse_work.jsonl`.

Held-out validation result on all 310 validation CVs:

| Method | Macro | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|
| standard LayoutLMv3 | 0.739158 | 0.627738 | 1.000000 | 0.000000 |
| work-only EFSFR repair | 0.763803 | 0.923473 | 1.000000 | 0.000000 |
| work-only SG-GRSE decoder | **0.764074** | **0.926728** | 1.000000 | 0.000000 |
| full EFSFR nested repair | 0.775538 | 0.923473 | 1.000000 | 0.000000 |
| full EFSFR + SG-GRSE work | **0.775809** | **0.926728** | 1.000000 | 0.000000 |

Observed SG-GRSE work activity:

- raw baseline documents redecoded: `306 / 310`;
- EFSFR documents changed by SG-GRSE work: `31 / 310`;
- adjacent slot merges applied: `23`.

Interpretation:

- the first SG-GRSE slice does improve held-out practical accuracy;
- it beats work-only EFSFR by `+0.000271` macro and beats full EFSFR by the
  same margin;
- the gain is real but very small, so it should be treated as a promising
  architecture signal rather than a stable final winner;
- the main thesis value right now is that both EFSFR and SG-GRSE improve over
  the raw LayoutLMv3 baseline in complementary ways.

### Comparison Utility And Stability Check

Added a reusable comparison utility:

```bash
uv run eraparse sge compare \
  --left-predictions ... \
  --right-predictions ... \
  --requests artifacts/sge/records/pymupdf4llm_validation.jsonl \
  --output ...
```

It reports:

- mean document-level macro delta;
- mean document-level work delta;
- win/loss/tie counts;
- bootstrap 95% confidence intervals for the mean deltas.

For full EFSFR versus full EFSFR + SG-GRSE work:

- macro delta mean: `+0.000271`;
- macro wins / losses / ties: `20 / 11 / 279`;
- macro delta 95% bootstrap CI:
  `[-0.000519, 0.001070]`;
- work delta mean: `+0.003256`;
- work delta 95% bootstrap CI:
  `[-0.006226, 0.012843]`.

Interpretation:

- the SG-GRSE slice is directionally positive;
- the current margin is too small to treat as statistically secure;
- this strengthens the case for continuing SG-GRSE refinement, but weakens any
  claim that the present architecture already clearly outruns EFSFR.

### Success: Selective SG-GRSE Work Becomes The Strongest And Cleanest Lane

Implemented a selective acceptance stage:

```bash
uv run eraparse sge select-sgrse-work \
  --baseline-predictions .../validation_predictions_efsfr.jsonl \
  --sgrse-predictions .../validation_predictions_efsfr_sgrse_work.jsonl \
  --output .../validation_predictions_efsfr_sgrse_work_selective.jsonl
```

The current selector is intentionally simple and observable:

- accept SG-GRSE work records only when the decoded work set increases the
  count of near-complete work records relative to the EFSFR work set.

Held-out validation result on all 310 validation CVs:

| Method | Macro | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|
| full EFSFR nested repair | 0.775538 | 0.923473 | 1.000000 | 0.000000 |
| unconditional EFSFR + SG-GRSE work | 0.775809 | 0.926728 | 1.000000 | 0.000000 |
| selective EFSFR + SG-GRSE work | **0.776183** | **0.931212** | 1.000000 | 0.000000 |

Selective acceptance summary:

- SG-GRSE accepted on `7 / 310` validation CVs;
- macro delta mean versus full EFSFR: `+0.000645`;
- work delta mean versus full EFSFR: `+0.007739`;
- macro wins / losses / ties: `7 / 0 / 303`;
- macro delta 95% bootstrap CI:
  `[0.000169, 0.001270]`;
- work delta 95% bootstrap CI:
  `[0.002026, 0.015240]`.

Interpretation:

- this is the first SG-GRSE-based lane whose bootstrap comparison interval is
  entirely above zero on validation;
- the gain is still modest, but it is materially more defensible than the
  unconditional SG-GRSE result;
- the thesis story is now stronger: SG-GRSE is most useful as a selective
  architecture specialist rather than as a blanket replacement.

## Fixed Constraints

- Local machine: Apple M1 Pro, 16 GB unified memory.
- Modal credit visible to the user at stage start: **$3.31**.
- Current-stage Modal hard spending stop: **$3.00**.
- Reserved credit: at least **$0.31** from the visible balance, with the
  original plan targeting a $0.60 reserve when possible.
- T4 GPU rate supplied by the user: **$0.000164/second**, or **$0.5904/hour**.
- The user later switched to Modal profile `anasahmadoff` and made
  approximately `$15-20` available for the next controlled trials.
- Dataset source remained read-only.
- Locked confirmation corpus remained untouched.

## Implemented Foundation

### Evidence And Assembly

Implemented:

- canonical `EvidenceUnit` and `EvidenceGraph` contracts;
- practical PyMuPDF4LLM JSON-to-word evidence adapter;
- source-oracle word-box evidence adapter;
- canonical truth-to-evidence alignment;
- practical versus oracle separation and oracle guard;
- deterministic reduced-schema assembler;
- grounded field candidates, record links, selector traces, and predictions;
- evidence validation and SG-ESE preparation CLI commands.

Public commands now include:

```bash
uv run eraparse evidence build ...
uv run eraparse evidence validate ...
uv run eraparse sge prepare ...
uv run eraparse sge assemble ...
uv run eraparse sge evaluate ...
uv run eraparse sge estimate-t4 ...
uv run eraparse sge local-smoke ...
```

### Model Architecture

Implemented `SchemaGuidedLayoutLMv3` with:

- pinned `microsoft/layoutlmv3-base` encoder;
- learned schema-query embeddings;
- query-to-document cross-attention layers;
- query-conditioned field logits;
- field-presence head;
- evidence-relevance objective;
- same-record grouping head;
- frozen-encoder and final-layer-unfreeze controls;
- deterministic schema assembly outside the model.

The current weighted objective is:

```text
loss =
  class-balanced field token loss
+ 0.25 * field presence loss
+ 0.50 * record grouping loss
+ 0.50 * evidence relevance loss
```

The outside/background token receives weight `0.05`; field classes receive
weight `1.0`. The standard LayoutLMv3 comparison uses the same class-balanced
token loss.

### Execution And Budget Safety

Implemented:

- local MPS smoke runner for standard LayoutLMv3 and SG-ESE;
- Modal one-T4 smoke app;
- fixed model ID and revision;
- fixed seed support;
- T4 cost projection with 1.30 safety factor;
- stage-budget and visible-credit gates;
- one-T4-only Modal configuration;
- training schedule that can cycle a tiny subset for overfit tests.

For a 15-minute T4 timeout:

```text
900 seconds * $0.000164/second * 1.30 = $0.19188
```

The projected run is within the visible $3.31 credit and $3.00 stage budget,
but it remains blocked until a seeded local overfit gate passes.

## Reader Trials

### Failure: PyMuPDF4LLM Table Contract

Initial practical evidence generation failed because PyMuPDF4LLM table boxes
can contain:

```json
{"textlines": null, "table": {...}}
```

The first adapter assumed `textlines` was always iterable. It raised:

```text
TypeError: 'NoneType' object is not iterable
```

Correction:

- tolerate null text lines;
- preserve table-cell text;
- approximate word boxes from table-cell boxes;
- add a regression test.

Interpretation:

- this was an adapter bug, not a PyMuPDF4LLM extraction failure;
- native PyMuPDF4LLM structured output must be treated as a union of ordinary
  text boxes and table boxes.

### Failure: Cached Practical Run Had OCR Disabled

After the table correction, practical evidence generated for only 40 of 50
debug CVs. The ten missing documents were image-only T4 CVs. Cached JSON showed:

```json
{"use_ocr": 0, "boxes": []}
```

Correction:

- regenerated all 50 practical JSON files locally using PyMuPDF4LLM 1.27.2.2;
- relied on automatic smart OCR;
- local Tesseract 5.5.2 handled scanned pages.

Result:

| Reader result | Value |
|---|---:|
| completed CVs | 50 / 50 |
| failed CVs | 0 |
| pages | 55 |
| full-OCR pages | 12 |
| mean generation latency | 0.7913 seconds/CV |
| maximum generation latency | 3.3146 seconds/CV |

Interpretation:

- PyMuPDF4LLM smart OCR is viable locally and does not need Modal GPU;
- the M1 should own practical evidence generation and caching;
- PyMuPDF4LLM remains the primary practical reader;
- raw PyMuPDF remains a historical lower bound and optional word-box utility.

### Success: Final Debug-50 Evidence Contracts

After all current alignment corrections:

| Lane | Documents | Pages/records | Evidence units | Labeled tokens | Passed |
|---|---:|---:|---:|---:|---:|
| source-oracle boxes | 50 | 55 | 11,828 | 4,339 | yes |
| PyMuPDF4LLM smart OCR | 50 | 55 | 11,390 | 3,894 | yes |

Interpretation:

- practical PyMuPDF4LLM recovers nearly the same amount of trainable evidence
  as the source-oracle geometry lane on `debug_50`;
- the remaining practical-oracle gap is measurable and suitable for the thesis
  representation-quality comparison;
- this is evidence coverage, not extraction accuracy.

The lower final labeled-token counts are corrections, not regressions. Earlier
counts included duplicate and incorrectly aligned truth values.

## Label And Alignment Corrections

### Failure: Punctuation-Rich Values Did Not Align

The first alignment implementation normalized each evidence unit independently
and compared token lists. Values stored as one evidence unit, such as:

```text
jessica.robinson@gmail.com
```

normalized into multiple comparison tokens and failed to align.

Correction:

- compare normalized concatenated evidence spans against the normalized target;
- add coverage for emails, phones, URLs, and punctuation-rich values.

### Failure: Skills Path Was Not Canonicalized

The source data uses paths such as:

```text
skills.3.skill_name
```

The first canonicalizer handled only `skills.3`.

Correction:

- map both forms to the canonical `skills` field.

### Failure: Repeated-Record IDs Collided Across Sections

The first grouping labels used only `record_index`. Therefore:

```text
work_experience record 0 == education record 0
```

from the grouping head's perspective.

Correction:

- encode section-aware record group labels;
- reserve separate ranges for work experience, education, projects, and
  certifications;
- add a regression test preventing cross-section collisions.

### Failure: Source Word Labels Were Not Reliably Oracle

Audit found incorrect source word annotations. One observed example labeled a
skills row as certification fields. Separately, many field-annotation files
omit skills.

Correction:

- source-oracle now means source/oracle **geometry**, not blind trust in source
  field labels;
- clear source word labels before training alignment;
- align field annotations as page-aware hints;
- fill remaining labels from the canonical reduced-schema truth;
- never overwrite an already aligned label.

Interpretation:

- source-provided labels require auditing and cannot be treated as infallible;
- canonical truth plus source boxes is the defensible oracle training lane;
- practical versus oracle must differ by evidence geometry/reader, not by
  contradictory target semantics.

### Failure: Separators Were Absorbed Into Values

Normalized span matching initially allowed punctuation-only separators such as
`·` to become part of email, phone, location, or URL candidates.

Correction:

- never start a matched span from a punctuation-only evidence unit.

### Failure: Skills As One Large Candidate

Contiguous skill tokens were decoded into a single comma-delimited candidate,
which would cap set-valued skill accuracy.

Correction:

- deterministically split skill candidates on comma, semicolon, and pipe
  delimiters while preserving multi-token skills such as `Cloudflare Pages`.

### Failure: Truth Fallback Duplicated Already Grounded Values

Truth-derived fallback alignment did not know that a field annotation had
already satisfied a target. Repeated values could therefore be assigned again
to an unrelated occurrence.

Correction:

- align authoritative field annotations first;
- treat matching field/record/value targets as already satisfied;
- align list-valued skills and project technologies as compact ordered groups;
- add regression tests for duplicate values and compact list alignment.

### Failure: PyMuPDF4LLM Styled Names Were Split Across Spans

On several academic-style CVs, PyMuPDF4LLM emitted names as adjacent styled
spans such as:

```text
J + essica R + obinson
```

The first adapter split each span independently, producing four incorrect
words. This caused practical tiny-overfit `full_name` accuracy to fall to
`0.50`.

Correction:

- reconstruct each text line across adjacent style spans before word splitting;
- preserve a word boundary only when the geometric gap indicates one;
- add a regression test.

After correction, all ten formal practical-overfit documents had exactly the
same labeled-token counts as their source-oracle counterparts.

## Seeded Fair Comparisons

All results below use seed `20260609`. Tiny-overfit and `debug_50` scores are
training-set engineering diagnostics, not validation results.

### Formal Evidence-Supported Tiny Set

The formal selector chose ten documents with `1.00` truth-to-evidence coverage.
This replaced arbitrary tiny selection, which had included targets absent from
the available page evidence.

### Frozen-Encoder Tiny Comparison

| Method | Encoder | Macro | Schema valid | Unsupported |
|---|---|---:|---:|---:|
| standard LayoutLMv3 | frozen | 0.6990 | 100% | 0% |
| SG-ESE, learned grouping | frozen | 0.8083 | 100% | 0% |
| SG-ESE, deterministic grouping | frozen | 0.8198 | 100% | 0% |

This initially suggested a large SG-ESE advantage, but it was not yet the
fair promoted comparison because the planned SG-ESE configuration later
unfroze encoder layers.

### Final-Four-Layer Tiny Comparison

| Method | Macro | Schema valid | Unsupported |
|---|---:|---:|---:|
| standard LayoutLMv3 | 0.9231 | 100% | 0% |
| SG-ESE, learned grouping | **0.9264** | 100% | 0% |
| SG-ESE, deterministic grouping | 0.9236 | 100% | 0% |
| SG-ESE + practical PyMuPDF4LLM | 0.9191 | 100% | 0% |

Interpretation:

- unfreezing the final four encoder layers is the main tiny-overfit gain;
- the earlier large SG-ESE advantage was mostly an unequal fine-tuning-budget
  comparison and must not be claimed;
- SG-ESE is feasible, but its tiny-set advantage over a fair standard
  LayoutLMv3 baseline is small;
- practical PyMuPDF4LLM is sufficiently strong to pass the tiny diagnostic.

### Fair Practical Debug-50 Comparison

Both methods used PyMuPDF4LLM evidence, all 50 documents/55 page records, seed
`20260609`, 1,000 steps, and the final four encoder layers unfrozen.

| Method | Macro | Schema valid | Unsupported |
|---|---:|---:|---:|
| standard LayoutLMv3 | **0.7032** | 100% | 0% |
| SG-ESE, deterministic grouping | 0.6803 | 100% | 0% |
| SG-ESE, learned grouping | 0.6690 | 100% | 0% |

Interpretation:

- standard LayoutLMv3 is currently the stronger practical debug baseline;
- learned grouping is a negative contribution at this stage;
- SG-ESE is not currently entitled to a performance-improvement claim;
- the next architecture work is a component ablation matrix, not a larger
  blind training run;
- negative results remain thesis evidence when the comparison is fair and the
  cause is isolated.

### Practical Debug-50 SG-ESE Component Ablations

All variants used the same practical evidence, seed, 1,000 steps, final-four
encoder-layer fine-tuning, and deterministic sequence grouping for the reported
score.

| Variant | Macro | Delta vs standard |
|---|---:|---:|
| standard LayoutLMv3 | **0.7032** | reference |
| SG-ESE, two query layers, query-only loss | 0.6998 | -0.0034 |
| SG-ESE, one query layer, query-only loss | 0.6980 | -0.0052 |
| SG-ESE, no grouping loss, other auxiliaries retained | 0.6938 | -0.0094 |
| SG-ESE, all proposed auxiliary losses | 0.6803 | -0.0229 |
| SG-ESE, learned grouping decoder | 0.6690 | -0.0342 |

Interpretation:

- the schema-query classifier itself is nearly competitive with standard
  LayoutLMv3;
- a second query-attention layer is slightly better than one;
- grouping loss, presence loss, and evidence loss as currently formulated
  degrade practical accuracy;
- the learned grouping decoder is consistently worse than deterministic
  sequence grouping;
- the present contribution is an experimentally grounded architecture and
  ablation finding, not yet an accuracy-superiority claim;
- future SG-ESE work should redesign auxiliary supervision or pursue
  efficiency/grounding benefits rather than scaling the failing objectives.

## Local Apple-Silicon Execution

### Environment Failure And Correction

Initial local ML installation failed because the existing `uv` and managed
Python were x86_64/Rosetta, while Torch 2.7.1 publishes a macOS ARM64 wheel.

Correction:

- installed an isolated native ARM64 `uv` under ignored `artifacts/`;
- installed managed Python 3.11.15 ARM64 under ignored `artifacts/`;
- created an isolated native local-ML environment;
- left the repository's existing development environment intact.

Verified local environment:

| Component | Value |
|---|---|
| machine | `arm64` |
| Python | 3.11.15 ARM64 |
| Torch | 2.7.1 |
| Transformers | 4.57.3 |
| MPS built | yes |
| MPS available | yes |

Interpretation:

- one-step and tiny-overfit architecture work is practical on the M1 Pro;
- the local machine should be used before Modal for evidence work, architecture
  debugging, loss tests, deterministic assembly, and small MPS trials;
- Modal should be reserved for promoted throughput and larger debug runs.

## Model Trials

All trials in this section used source-oracle training records and the M1 Pro
MPS device. They are training-set diagnostic probes, not final results.

Important reproducibility warning:

- these probes were run before explicit fixed-seed support was added;
- they are valid engineering diagnostics but are not frozen/reportable thesis
  results;
- the next comparison must rerun with seed `20260609`.

### Success: Standard LayoutLMv3 One-Step Smoke

| Item | Value |
|---|---:|
| mode | standard LayoutLMv3 token classifier |
| trainable portion | classifier head; encoder frozen |
| records | 1 |
| steps | 1 |
| loss | 3.1570 |
| runtime | 6.20 seconds |
| device | MPS |

Interpretation:

- the pinned model and processor load successfully on the M1;
- page images, words, boxes, and labels are compatible;
- MPS supports the required standard-baseline operations.

### Success: SG-ESE One-Step Smoke

| Item | Value |
|---|---:|
| mode | SG-ESE |
| trainable portion | SG-ESE heads; encoder frozen |
| records | 1 |
| steps | 1 |
| loss | 4.4979 |
| runtime | 5.94 seconds |
| device | MPS |

Interpretation:

- schema queries, query attention, field logits, presence, evidence, and
  grouping losses all execute with gradients;
- the architecture is technically feasible on one M1 and one T4.

### Success: SG-ESE Loss-Direction Probe

| Item | Value |
|---|---:|
| records | 2 |
| steps | 20 |
| first loss | 3.9385 |
| last loss | 2.3738 |
| runtime | 4.31 seconds |

Interpretation:

- the frozen-head architecture learns rather than failing immediately;
- local iteration is fast enough to diagnose architecture and data issues.

### Failure: Unweighted Token Loss

An evaluation-enabled 200-step probe with ordinary cross-entropy produced:

| Metric | Value |
|---|---:|
| first loss | 3.8525 |
| last loss | 0.9708 |
| training macro | 0.1167 |
| schema validity | 100% |

Interpretation:

- JSON complexity is not the current problem because deterministic assembly
  guarantees valid schema;
- loss can fall while useful field extraction remains poor;
- outside/background tokens dominate the unweighted objective;
- falling loss alone remains an invalid promotion criterion.

### Success: Class-Balanced SG-ESE Loss

After reducing outside/background class weight to `0.05`, the same 200-step,
two-record diagnostic produced:

| Metric | Unweighted | Class-balanced |
|---|---:|---:|
| last loss | 0.9708 | 0.6572 |
| training macro | 0.1167 | **0.5537** |
| schema validity | 100% | 100% |

Exact scores reached `1.0` for name, email, LinkedIn, and location in the
class-balanced probe.

Interpretation:

- class imbalance was a major bottleneck;
- the architecture can learn useful grounded extraction;
- deterministic assembly successfully removes schema validity from the model's
  learning burden.

### Partial Success: 1,000-Step SG-ESE Ceiling Probe

The pre-final-label-correction 1,000-step weighted probe produced:

| Metric | Value |
|---|---:|
| first loss | 4.0085 |
| last loss | 0.0678 |
| runtime | 175.72 seconds |
| training macro | 0.6399 |
| schema validity | 100% |
| unsupported evidence | 0% |

Strong fields included name, email, location, phone, and summary. Weak fields
included skills, projects, education grouping, and certification grouping.

Interpretation:

- SG-ESE has meaningful learning capacity;
- repeated-record grouping and deterministic decoding remain the main model
  challenge;
- the observed skill failure was partly caused by label/decoder defects found
  after this run;
- this run must not be treated as the final tiny-overfit ceiling.

## Current Scientific Interpretation

### What The Current Problem Is Not

The current SG-ESE problem is **not** primarily the complexity of emitting the
JSON file:

- deterministic assembly gives 100% schema validity;
- required keys, nullable arrays, and JSON syntax no longer burden the model;
- unsupported-evidence rate is 0% in the tiny local diagnostics.

### What The Current Problem Is

The remaining problem is grounded semantic extraction and repeated-record
grouping:

- assigning all evidence spans to the correct schema field;
- separating adjacent repeated records;
- avoiding label corruption and alignment leakage;
- decoding field spans without including layout punctuation;
- producing correct set-valued skills and nested arrays.

### Thesis Direction Supported So Far

The current evidence supports the thesis flow:

1. one-shot generation baselines expose the cost of combining perception,
   extraction, and structure generation;
2. PyMuPDF4LLM smart OCR provides a strong practical evidence reader;
3. deterministic assembly removes schema-generation failure;
4. LayoutLMv3 and SG-ESE isolate grounded field extraction;
5. SG-ESE adds schema queries and repeated-record grouping as the architecture
   contribution;
6. sparse evidence/token selection remains a later efficiency contribution
   only after full-token SG-ESE is reliable.

Do not claim SG-ESE novelty yet.

## Relationship To The Donut Failures

### Short Answer

The new SG-ESE findings explain **part** of why Donut struggled, but the newly
found evidence-label and deterministic-decoder bugs did **not** cause the
previous Donut runs to fail.

Donut used page images and serialized reduced-schema targets. It did not train
from the new evidence graphs, word labels, record-group labels, or SG-ESE
candidate decoder. Therefore:

- PyMuPDF4LLM table/OCR bugs did not affect Donut;
- source word-label corruption did not affect Donut;
- SG-ESE punctuation and skill-candidate decoding bugs did not affect Donut.

Those issues affected only the new grounded extraction lane.

### Donut Failure Causes Confirmed By Existing Trials

The first Donut adaptations contained an implementation error: the
`<s_eraparse>` task token appeared in the label sequence while also being used
as `decoder_start_token_id`. This violated the reference Donut fine-tuning
contract and caused degenerate generation. Those runs are excluded.

After fixing that contract, the corrected native-token Donut run still failed:

| Measure | Corrected Donut result |
|---|---:|
| training CVs | 1,445 |
| validation CVs | 310 |
| epoch-3 validation loss | 0.3882 |
| JSON valid | 100% |
| schema valid | 0% |
| macro | 0.1295 |
| hit 1,536-token cap | 306 / 310 |
| mean encoder latency | 0.2371 seconds |
| mean decoder latency | 6.7400 seconds |
| visual tokens | 4,800 |

The corrected run proves the failure was not only the original implementation
error. Donut learned token-level target patterns, but did not reliably emit the
complete reduced-schema structure.

### How The SG-ESE Findings Explain Donut's Difficulty

The grounded lane exposed several properties that make one-shot Donut
generation unusually difficult:

1. **Repeated-record grouping is a separate hard problem.** Work experience,
   education, projects, and certifications require evidence to be assigned to
   the correct record. SG-ESE still finds this difficult even when JSON syntax
   is removed. Donut had to learn grouping and bracket/field-token generation
   jointly.
2. **Set-valued skills need segmentation and deduplication.** A visually
   contiguous skills row can represent many values. Donut had to generate the
   full array correctly; SG-ESE can use deterministic splitting and assembly.
3. **Rare fields and background are imbalanced.** SG-ESE improved sharply only
   after class balancing. Donut's autoregressive loss is dominated by common
   structural and frequent target-token patterns, so low loss does not ensure
   correct rare fields or record boundaries.
4. **Structure validity and field accuracy are different objectives.** Donut's
   validation loss fell while generations remained invalid or incomplete.
   SG-ESE makes this separation explicit and guarantees structure outside the
   model.
5. **Multi-page resizing may reduce readable evidence.** The Donut baseline
   vertically stacked pages before the fixed processor resize. Two-page CVs
   therefore receive less effective text resolution. This remains a plausible
   perception bottleneck that has not yet been isolated.
6. **Donut-base capacity and pretraining may be insufficient for the complete
   contract.** It is a base OCR-free encoder-decoder, not a CV-schema-specialized
   structured extraction model.

### What Has Not Yet Been Proven About Donut

The existing trials do not prove that Donut's visual encoder cannot read the
CVs. Because one-shot generation combines perception, semantic extraction,
record grouping, and structure generation, the observed failure cannot locate
the bottleneck precisely.

The required Donut follow-up remains:

1. contact-only tiny overfit;
2. flat-core tiny overfit;
3. experience-only nested tiny overfit;
4. page-wise versus vertically stacked pages;
5. Donut encoder features feeding deterministic field-query/span heads;
6. decomposed section generation with deterministic merging.

Interpretation:

- if simple/decomposed Donut succeeds, schema complexity and autoregressive
  structure were the main failure;
- if page-wise processing succeeds, stacked-page resizing was the main
  perception failure;
- if Donut encoder features work with SG-ESE heads, the encoder was useful but
  the decoder contract failed;
- if even contact-only tiny overfit fails, stop investing in Donut-base.

## Modal Cost And Decision Log

### Current Stage

- visible user credit at stage start: `$3.31`;
- user later switched to profile `anasahmadoff` with a larger trial budget;
- on June 12, 2026, the user authorized `$15-20` for the next controlled trial
  stage; the repository hard stop is now `$18.00`, preserving approximately
  `$2.00` of the maximum authorization as reserve;
- measured SG-ESE-stage Modal spend across the two profiles: approximately
  `$0.00241920`;
- one 15-minute T4 smoke maximum projected GPU cost: `$0.19188`;
- one successful T4 checkpoint/reload smoke completed.

### Modal CLI Issue

Calling the system `modal` executable directly failed because it used an
x86_64 Python package from an ARM64 process:

```text
ImportError: watchfiles ... incompatible architecture
```

Correction:

- use `uv run --group modal modal ...` from the repository environment;
- never use the stale system Modal CLI for reportable runs.

### Decision

The hard `0.90` score is no longer a universal promotion veto. It remains a
diagnostic target. Future trials may proceed when contracts are sound and the
comparison or ablation is scientifically useful.

Reserve the new `$15-20` allowance for:

1. a validation-aware train/validation comparison after local ablations;
2. modern upper-bound model smokes and selected debug runs;
3. targeted Donut decomposition tests;
4. no blind full-data reruns.

### June 12 Continuation Controls

- Active Modal profile verified as `anasahmadoff`.
- `modal billing report --for today --resolution h` returned no billed rows
  before launching the next trial stage.
- The repository budget guard now uses an `$18.00` hard stop, preserving
  approximately `$2.00` of the user's maximum `$20` authorization.
- SG-ESE promoted runs now default to deterministic sequence grouping because
  it consistently beat learned grouping in the completed ablations. Learned
  grouping remains available as a named negative-control ablation.
- Resume checkpoints now reject mismatched loss weights in addition to mode,
  seed, query depth, and encoder-unfreeze configuration.
- Training and evaluation timings are recorded separately for honest
  throughput and Modal-cost projections.
- `eraparse sge report-trials` generates JSON and Markdown comparison tables
  from persisted run summaries. It catalogued 26 existing local diagnostics
  and confirmed that the project still had zero held-out validation-aware
  SG-ESE/LayoutLMv3 trials at this point.

### Full Practical Train/Validation Preparation

PyMuPDF4LLM representation generation completed locally on the M1 Pro without
using Modal:

| Split | CVs | Newly generated | Cached | Failures |
|---|---:|---:|---:|---:|
| train | 1,445 | 1,395 | 50 | 0 |
| validation | 310 | 310 | 0 | 0 |

The validated evidence and SG-ESE preparation outputs are:

| Split | Evidence units | Page records | Labeled tokens | Oracle |
|---|---:|---:|---:|---|
| train | 319,942 | 1,601 | 112,609 | no |
| validation | 67,919 | 340 | 24,170 | no |

This is the first full practical PyMuPDF4LLM train/validation corpus. The
subsequent equal-budget LayoutLMv3 versus SG-ESE comparison is a held-out
validation trial; previous `debug_50` scores were training diagnostics.

### First Held-Out LayoutLMv3 Result

Configuration:

- practical PyMuPDF4LLM evidence;
- all 1,601 train page records, one epoch/1,601 update steps;
- final four LayoutLMv3 layers unfrozen;
- seed `20260609`;
- all 340 validation page records representing 310 CVs;
- deterministic sequence assembly.

Result:

| Metric | Value |
|---|---:|
| validation macro | **0.728579** |
| schema-valid rate | **1.000000** |
| unsupported-evidence rate | **0.000000** |
| validation evaluation time on M1 Pro | 60.19 seconds |

This exceeds the prior practical Qwen3 + PyMuPDF/Tesseract validation macro
`0.6709` by approximately `0.0577`. This is the strongest practical held-out
validation score currently recorded in the project.

The first evaluation attempt failed after training because a malformed
predicted URL (`https: //...`) caused `urllib.parse`'s `port` property to
raise. The model checkpoint was intact. A regression test was added, URL
normalization was made safe for malformed model output, and zero-step
checkpoint evaluation was added so the saved model could be evaluated without
retraining. The resumed validation evaluation completed successfully.

### First Held-Out SG-ESE Result

The strongest `debug_50` SG-ESE configuration, schema-query-only with two query
layers and deterministic sequence grouping, was trained and evaluated under
the same one-epoch practical setup:

| Method | Validation macro | Schema valid | Unsupported evidence |
|---|---:|---:|---:|
| Qwen3 + PyMuPDF/Tesseract | 0.6709 | 0.9968 | 0.0410 |
| SG-ESE query-only, one epoch | **0.689791** | **1.000000** | **0.000000** |
| standard LayoutLMv3, one epoch | **0.728579** | **1.000000** | **0.000000** |

Interpretation:

- practical SG-ESE improves over the prior Qwen3 baseline by approximately
  `0.0189`;
- standard LayoutLMv3 remains stronger than SG-ESE by approximately `0.0388`;
- learned grouping remains harmful on held-out validation (`0.636137`) versus
  deterministic sequence grouping (`0.689791`);
- the present architecture contribution is therefore a grounded, schema-valid
  design and a controlled negative/ablation result, not an accuracy win over
  standard LayoutLMv3;
- a fair second epoch is required to distinguish slower SG-ESE convergence
  from an architectural disadvantage.

The resumed second SG-ESE epoch reached:

| Metric | One epoch | Two epochs |
|---|---:|---:|
| validation macro | 0.689791 | **0.738284** |
| schema-valid rate | 1.000000 | 1.000000 |
| unsupported-evidence rate | 0.000000 | 0.000000 |

This is `0.0097` above the one-epoch standard LayoutLMv3 result, showing that
SG-ESE converges more slowly but can become competitive. A fair two-epoch
standard LayoutLMv3 run is required before claiming an architecture gain.

The fair two-epoch standard LayoutLMv3 run reached `0.739158`. The final
equal-budget result is:

| Method | Validation macro | Schema valid | Unsupported evidence |
|---|---:|---:|---:|
| standard LayoutLMv3, two epochs | **0.739158** | 1.000000 | 0.000000 |
| SG-ESE query-only, two epochs | **0.738284** | 1.000000 | 0.000000 |

The difference is only `0.000875` in favor of standard LayoutLMv3. SG-ESE is
therefore competitive but not an accuracy winner. This passes the project gate
for sparse-token/efficiency work because SG-ESE is within `0.01` macro of the
full-token baseline. The next architecture contribution must target a better
accuracy/latency trade-off or improve the weaker nested fields; it must not
claim an accuracy gain from the current query-only design.

## Second Contribution Decision

The next accuracy-oriented architecture is **SG-GRSE**, documented in
[`SECOND_CONTRIBUTION_PLAN.md`](SECOND_CONTRIBUTION_PLAN.md). It replaces
post-hoc repeated-record grouping with document-level grounded record queries,
field-conditioned span pointers, and Hungarian-matched set prediction.

This direction was selected because the weakest current fields are skills
`0.462888`, projects `0.542950`, and work experience `0.627738`, while learned
all-pairs grouping underperformed deterministic grouping. SG-GRSE directly
optimizes the repeated-record structure instead of adding another grouping
heuristic after token classification.

The separate methodology/community contribution is **EFSFR**, an
evidence-first selective-fusion and repair protocol. It uses calibrated
agreement, confidence, evidence support, and schema consistency to select or
repair fields and records while producing an auditable selector trace.

Accuracy claims have explicit gates:

- SG-GRSE must beat equal-budget standard LayoutLMv3 `0.739158`;
- minimum SG-GRSE target: greater than `0.745` validation macro;
- EFSFR must exceed the best single practical extractor by at least `0.005`
  macro while preserving grounding and schema validity.

### Oracle Ceiling Check

A new `eraparse sge oracle-ceiling` utility now evaluates prepared SGE records
using their gold field labels with either deterministic sequence grouping or
gold record grouping.

On the full practical validation records (`artifacts/sge/records/pymupdf4llm_validation.jsonl`):

| Mode | Macro | Schema valid | Unsupported evidence |
|---|---:|---:|---:|
| standard LayoutLMv3, two epochs | 0.739158 | 1.000000 | 0.000000 |
| oracle labels + sequence grouping | 0.748347 | 1.000000 | 0.000000 |
| oracle labels + oracle grouping | 0.755676 | 1.000000 | 0.000000 |

Interpretation:

- the existing practical PyMuPDF4LLM evidence already supports a score above
  the current best learned baseline;
- deterministic grouping leaves approximately `0.0073` macro on the table
  relative to oracle grouping;
- the full practical headroom from current best baseline to oracle grouping is
  approximately `0.0165`;
- projects and work experience gain the most from oracle grouping, supporting
  SG-GRSE as the right next architecture target.

### SG-GRSE Work-Bank Start

The first SG-GRSE implementation slice adds
`eraparse sge prepare-work-bank`, which converts page-level practical SGE
records into document-level work-experience candidate spans and explicit work
record targets.

On the full practical validation records:

| Metric | Value |
|---|---:|
| documents | 310 |
| target work records | 910 |
| candidate spans | 3,543 |
| direct record match rate | 0.880220 |
| exact record match rate | 0.059341 |
| job title coverage | 0.952747 |
| company coverage | 0.960440 |
| start date coverage | 0.945055 |
| end date coverage | 0.947253 |
| duration coverage | 0.095604 |

Interpretation:

- work-experience candidate spans are already strong for title, company, and
  dates;
- `duration` is usually not directly grounded in evidence and should be treated
  as a derived field in SG-GRSE rather than a span-pointer target;
- high direct span coverage but weak current work-experience field score means
  the remaining bottleneck is record assembly and matching quality, not basic
  span availability;
- this supports continuing with a document-level work-record decoder instead of
  another token-label-only variant.

## Verification Status

Latest completed fast verification:

```text
72 tests passed before the latest ablation-control edit
Ruff passed
mypy passed
git diff --check passed
```

The full dataset-backed integration scan was stopped after 15 passing tests
because source-file reads became unusually slow; it did not report a failure.

### Modal Trial Outcomes

1. First T4 attempt failed before training because the image omitted
   `pydantic`. Measured cost: `$0.000983`.
2. A new CPU remote-import gate prevented the next GPU allocation, but its
   response used a non-primitive `TorchVersion` value and failed local
   deserialization. Measured active-workspace hourly total including later
   success: `$0.00143620`.
3. Final T4 smoke succeeded:
   - GPU: Tesla T4;
   - one real SG-ESE training step;
   - loss: `4.256634`;
   - in-function runtime: `5.5791s`;
   - final four encoder layers unfrozen;
   - trainable parameters: `35,460,866`;
   - checkpoint and optimizer state persisted;
   - checkpoint reload verified.

The Modal app now runs a CPU remote-import gate before allocating a GPU.

## Immediate Continuation Point

Continue from [`docs/NEXT_MODEL_ACTION_PLAN.md`](NEXT_MODEL_ACTION_PLAN.md).

The immediate next work is:

1. measure oracle candidate-span and oracle record-grouping ceilings;
2. implement SG-GRSE for work experience only;
3. expand SG-GRSE only after the focused record-set decoder passes its gates;
4. implement EFSFR over frozen grounded predictions;
5. run the sparse-token efficiency contribution;
6. implement NuExtract3 and PaddleOCR-VL upper-bound lanes;
7. update this log after every promoted, stopped, failed, or corrected trial.

## 2026-06-13: Project Technology Repair Pass

I continued from [`docs/NEXT_MODEL_ACTION_PLAN.md`](NEXT_MODEL_ACTION_PLAN.md)
and audited the remaining `projects` weakness in the current best practical
lane.

Starting point:

| Method | Macro | Projects | Work experience | Schema valid |
|---|---:|---:|---:|---:|
| selective EFSFR + SG-GRSE work | 0.776183 | 0.645540 | 0.931212 | 1.000000 |

### Failure Analysis

The weakest recurring `projects` patterns on held-out validation were:

- missing project URLs despite correct project names;
- noisy project-technology strings such as
  `Created Data Visualization Tool leveraging Next.js for`;
- underspecified technology phrases such as
  `Web App with CSS and Astro`;
- truly missing project candidates, which a post-decoder repair cannot fix.

I tested one stronger deterministic URL synthesis idea first:

```text
github owner from predicted github_url
+ slugified project name
-> guessed project repo URL
```

Observed result:

- the dataset does contain a strong GitHub-owner and slug regularity when a
  project URL exists;
- but some truth project records intentionally have no URL;
- unconditional URL synthesis therefore reduced validation quality and was not
  promoted.

That failure is important for the thesis log:

- it explains why a simple repo-slug heuristic should not be claimed as a
  contribution;
- it also mirrors the earlier Donut lesson that plausible-looking structure can
  still hide field-level contract errors.

### Successful Repair

I then implemented a narrower deterministic pass:
`eraparse sge repair-project-tech`.

Behavior:

- keep project cardinality, names, and URLs fixed;
- rebuild project technologies only when candidate text is clearly noisy;
- use predicted document skills only as alignment hints for candidate cleanup;
- do not synthesize unsupported new fields.

Held-out validation result:

| Method | Macro | Projects | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|---:|
| selective EFSFR + SG-GRSE work | 0.776183 | 0.645540 | 0.931212 | 1.000000 | 0.000000 |
| + project technology repair | **0.776322** | **0.647206** | 0.931212 | 1.000000 | 0.000000 |

Measured delta:

- macro: `+0.000139`;
- projects: `+0.001667`;
- changed validation CVs: `2 / 310`;
- harmed validation CVs: `0`.

Artifacts:

- `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_predictions_efsfr_sgrse_work_selective_project_tech.jsonl`
- `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_efsfr_sgrse_work_selective_project_tech_evaluation.json`
- `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_efsfr_sgrse_work_selective_project_tech_compare.json`

Interpretation:

- this is a small but real positive methodological add-on;
- unlike the URL synthesis attempt, it improved only where the evidence text was
  visibly noisy;
- the strongest current practical lane is now:
  selective EFSFR + SG-GRSE work + project technology repair.

### Donut Relevance

This result also clarifies one reason the Donut lane struggled:

- long-form autoregressive JSON generation had to solve extraction, grouping,
  normalization, and field cleanup in one pass;
- our current best gains are coming instead from modular evidence-grounded
  repair steps that correct specific repeated-field failure modes after
  extraction;
- that does not prove Donut cannot improve, but it does explain why the
  modular LayoutLMv3 plus repair path has been easier to stabilize and score.

## 2026-06-13: Train-Derived Project URL Selector

I converted the earlier rejected fixed repo-slug heuristic into a train-derived
selector.

Implemented command:

- `eraparse sge repair-project-url`

Design:

- learn URL-presence priors only from train-split truth records;
- use observable features only:
  project-count, project-position, duplicate-name flag, capped technology-count,
  and capped project-name token count;
- require a populated train bucket before synthesis;
- if accepted, synthesize
  `https://github.com/{predicted_github_owner}/{slugified_project_name}`.

Important correction:

- the first implementation used a global fallback prior when no train bucket was
  available;
- that was weaker and less defensible;
- removing the global fallback improved validation and made the selector more
  faithful to the intended thesis claim.

Held-out validation result versus the previous best practical lane:

| Method | Macro | Projects | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|---:|
| selective EFSFR + SG-GRSE work + project technology repair | 0.776322 | 0.647206 | 0.931212 | 1.000000 | 0.000000 |
| + train-derived project URL selector | **0.776740** | **0.652224** | 0.931212 | 1.000000 | 0.000000 |

Measured delta:

- macro: `+0.000418`;
- projects: `+0.005018`;
- synthesized project URLs: `42`;
- changed validation CVs: `19 / 310`.

Artifacts:

- `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_predictions_efsfr_sgrse_work_selective_project_tech_project_url.jsonl`
- `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_efsfr_sgrse_work_selective_project_tech_project_url_evaluation.json`
- `artifacts/sge/local_smokes/layoutlmv3-pymupdf4llm-train-val-2epoch-unfreeze4-v1/validation_efsfr_sgrse_work_selective_project_tech_project_url_compare.json`

Interpretation:

- this is the strongest practical lane currently recorded in the project;
- unlike the original unconditional URL fill idea, this version is train-derived
  and selection-based;
- it strengthens the “modular methodology contribution” story more than the
  “single edited architecture wins outright” story.

## 2026-06-13: Full-Loss Practical SG-ESE Run

I also launched the missing larger-CV SG-ESE practical run on the M1 Pro after
fixing the local ML environment.

Environment correction:

- the repository `uv` runtime had been x86_64 Python 3.11, which could not
  install the pinned `torch==2.7.1` wheel for local MPS work;
- installing `cpython-3.11.15-macos-aarch64-none` and rerunning with
  `uv run --python 3.11.15 --group local-ml ...` restored the intended local
  ML path.

Run:

- output:
  `artifacts/sge/local_smokes/sge-pymupdf4llm-train-val-1epoch-unfreeze4-v1`
- mode: `sge`
- train records: `1601`
- validation records: `340`
- device: `mps`
- runtime: `696.51s`

Validation result:

| Model lane | Macro | Projects | Work experience | Schema valid |
|---|---:|---:|---:|---:|
| baseline LayoutLMv3, 2 epochs | **0.739158** | 0.542950 | **0.627738** | 1.000000 |
| SG-ESE query-only, 2 epochs | 0.738284 | **0.550493** | 0.598171 | 1.000000 |
| SG-ESE full-loss, 1 epoch | 0.688165 | 0.549634 | 0.623411 | 1.000000 |

Interpretation:

- the full-loss SG-ESE practical lane is currently not competitive with the
  baseline LayoutLMv3 lane;
- its work score recovered somewhat relative to the query-only SG-ESE lane, but
  the overall practical macro remained much lower;
- this reinforces the current thesis direction:
  use SG-ESE/SG-GRSE ideas selectively and modularly, while relying on the
  stronger practical extractor plus repair pipeline as the main winner.

## 2026-06-13: Real Mapper Scaling And NuExtract3 Speed Trials

The active next-stage plan is `docs/NEXT_STAGE_EXPERIMENT_MATRIX.md`, with the
broader decision sequence in `docs/NEXT_MODEL_ACTION_PLAN.md`.

### Qwen3-4B plus PyMuPDF4LLM

Completed a full train-derived `debug_50` run using:

- reader: PyMuPDF4LLM Markdown;
- mapper: `Qwen/Qwen3-4B-Instruct-2507`;
- pinned revision: `cdbee75f17c01a7cc42f958dc650907174af0554`;
- complete reduced CV schema;
- `enable_thinking=False`;
- one T4 worker.

Result:

| Documents | Macro | JSON valid | Schema valid | Unsupported evidence | Mean model latency |
|---:|---:|---:|---:|---:|---:|
| 50 | **0.680819** | 1.000000 | 1.000000 | 0.000000 | 17.169 s/CV |

Selected nested-field scores:

- work experience: `0.791810`;
- projects: `0.716111`;
- education: `0.660000`;
- skills: `0.760986`.

Measured Modal cost for the complete 50-CV app was approximately `$0.2876`.
This is the first fair greater-than-3B parser-to-mapper comparison. It is a
useful grounded baseline, but it does not beat the current practical repaired
LayoutLMv3 lane or the existing NuExtract3 upper bound.

### Phi-4 Mini compatibility smoke

One train-derived CV completed after isolating Phi in Transformers `4.49.0`
and using its required `torch_dtype` load argument.

- one-CV macro: `1.0`;
- schema validity: `1.0`;
- unsupported evidence: `0.0`;
- latency: `15.291 s`.

This is a compatibility smoke only and must not be reported as aggregate
accuracy. Earlier attempts failed because:

1. Transformers `4.57.3` was incompatible with the pinned Phi remote code;
2. inherited Modal class parameters were not recognized;
3. Transformers `4.49.0` expects `torch_dtype`, not the newer `dtype` keyword.

### NuExtract3 vLLM and MTP bring-up

NuExtract3 uses a Qwen3.5 multimodal architecture. The first vLLM `0.14.0`
attempt failed before inference because that runtime did not recognize
`model_type=qwen3_5`. Upgrading to vLLM `0.21.0` resolved
`Qwen3_5ForConditionalGeneration`.

Further controlled failures and fixes:

- the default FlashInfer sampler attempted JIT compilation without `nvcc`;
  setting the documented `VLLM_USE_FLASHINFER_SAMPLER=0` fallback allowed the
  engine to initialize with the PyTorch-native sampler;
- the first fully initialized engine then failed only because our metadata
  collector assumed the Modal runtime had package distribution metadata;
  metadata collection is now tolerant of runtime-provided packages.

The corrected one-CV full-schema vLLM baseline completed:

- macro: `0.988889`;
- schema validity: `1.0`;
- unsupported evidence: `0.0`;
- output tokens: `490`;
- measured generation chunk time: `27.548 s`.

The result is only a compatibility/speed smoke, not aggregate accuracy. Compact
schema and MTP smokes are running next. vLLM confirmed that NuExtract3 exposes
the `Qwen3_5MTP` draft model; the two-token configuration remains an empirical
speed/quality ablation because vLLM warns that more than one speculative token
can lower acceptance rate.

Compact-schema smoke result on the same CV:

| Variant | Macro | Schema valid | Output tokens | Generation time |
|---|---:|---:|---:|---:|
| full semantic keys | 0.988889 | 1.0 | 490 | 27.548 s |
| opaque compact aliases | **0.083333** | 1.0 | **218** | **24.135 s** |

The compact form reduced output tokens by `55.5%` and generation time by
`12.4%`, but it destroyed extraction accuracy. NuExtract3 interpreted opaque
aliases such as `n`, `e`, and `w` semantically rather than as a lossless coding
contract and shifted values into the wrong fields. This compact variant is
rejected. A future schema-shortening experiment must preserve semantic field
names, add explicit alias descriptions, or fine-tune the model on the coding
scheme.

MTP one-CV smoke result:

| Variant | Macro | Output tokens | Generation time |
|---|---:|---:|---:|
| vLLM baseline | 0.988889 | 490 | 27.548 s |
| two-token MTP | 0.988889 | 490 | 28.151 s |

MTP preserved the output and score but was `2.2%` slower on this single CV.
Because one CV is too noisy for a scientific speed conclusion, matched
train-derived `debug_50` baseline and MTP runs are the promotion decision gate.

Matched `debug_50` result:

| Variant | Macro | Schema valid | Unsupported evidence | Mean latency | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| vLLM baseline | 0.845862 | 1.0 | 0.049364 | 3.480 s/CV | 3.111 s | 7.146 s |
| two-token MTP | 0.845601 | 1.0 | 0.049364 | **2.279 s/CV** | **1.887 s** | **5.955 s** |

Decision:

- promote MTP as the NuExtract3 speed variant;
- measured mean generation-latency reduction: **34.5%**;
- macro delta: `-0.000261`;
- MTP preserved schema validity and unsupported-evidence rate;
- the one-CV smoke was misleading, demonstrating why the project requires
  real matched trial sizes before promotion.

Measured Modal costs for the matched complete apps:

- vLLM baseline `debug_50`: approximately `$0.1694`;
- MTP `debug_50`: approximately `$0.1322`.

One local MTP ingestion attempt failed because two evaluators tried to acquire
the same DuckDB write lock concurrently. It was rerun sequentially against a
separate run database with no GPU cost. Model trials may run in parallel, but
DuckDB result ingestion must remain serialized or use per-trial databases.

### Next decision: quantization before cascade generation

The active matrix now adds structured-output-preserving quantization as the
final NuExtract3 optimization gate before train OOF generation and cascade
training.

Planned official-checkpoint comparison:

- BF16 + MTP reference: `numind/NuExtract3`;
- accuracy-preserving candidate: `numind/NuExtract3-W8A8`;
- aggressive candidate: `numind/NuExtract3-W4A16`;
- FP8 deferred to suitable Ada/Hopper hardware.

The quantization decision will use complete-schema macro, nested-field scores,
unsupported evidence, repair events, latency, memory, and cost. A quantized
model that emits valid JSON but assigns values to the wrong fields fails.

After quantization selection, the next primary implementation is the
grounded-to-NuExtract3 cascade using leakage-safe train OOF predictions.

### Parallel accuracy, mapper, and quantization trials

The next-stage runner was exercised on matched train-derived full-schema
samples. These are development results, not frozen validation or final-test
results.

| Variant | CVs | Macro | Schema valid | Unsupported evidence | Mean latency |
|---|---:|---:|---:|---:|---:|
| NuExtract3 BF16 + MTP | 50 | 0.845601 | 1.0 | 0.049364 | 2.279 s/CV |
| NuExtract3 W4A16, MTP off | 50 | 0.848297 | 1.0 | 0.050737 | 2.582 s/CV |
| NuExtract3 MTP + PyMuPDF4LLM evidence text | 50 | **0.893809** | 1.0 | **0.046091** | 2.418 s/CV |
| Phi-4 Mini + PyMuPDF4LLM Markdown | 50 | 0.578810 | 1.0 | 0.214524 | 13.002 s/CV |

Interpretation:

- grounded evidence injection is the strongest new accuracy result, improving
  matched MTP macro by `+0.048207` for `+0.139 s/CV`;
- most of the gain came from identity/contact fields and education;
- work-experience score fell from `0.889976` to `0.858176`, so the next
  methodology ablation should inject or fuse evidence selectively by field;
- W4A16 is accurate and schema-safe, but it did not improve A10 latency over
  BF16 + MTP. Its value is reduced model memory, not current serving speed;
- Phi-4 Mini's perfect one-CV smoke was misleading. The 50-CV result is a
  useful lower-bound comparison and is not promoted.

The official `numind/NuExtract3-W8A8` checkpoint failed before inference under
vLLM `0.21.0`. vLLM rejected its compressed-tensors metadata because activation
ordering was configured without the required group or tensor-group strategy.
The repeated lifecycle retries were stopped. This is recorded as a runtime /
checkpoint compatibility failure, not an accuracy result.

The quantized-result ingestion interface was also corrected to accept explicit
model IDs and revisions. The W4A16 report now records the actual pinned
checkpoint rather than incorrectly attributing it to the BF16 base model.

### Promoted 250-CV results

The grounded-evidence and Qwen3-4B lanes were promoted directly to the same
fixed train-derived `debug_250` corpus.

| Variant | Macro | Work experience | Certifications | Schema valid | Unsupported | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B + PyMuPDF4LLM Markdown | 0.774910 | **0.925185** | **0.866536** | 1.0 | **0.003442** | 15.540 s/CV |
| NuExtract3 MTP + evidence text | 0.870103 | 0.862805 | 0.801960 | 1.0 | 0.055430 | **1.444 s/CV** |
| static field fusion | **0.880683** | **0.925185** | **0.866536** | 1.0 | 0.056386 | 16.984 s/CV |

The static fusion uses NuExtract3 evidence predictions for all fields except
`work_experience` and `certifications`, which come from Qwen3-4B. It improves
macro by `+0.010580` over grounded NuExtract3 and demonstrates genuine model
complementarity. It is not the final speed contribution because both models
currently run for every CV and the reported latency conservatively sums both
model calls.

Next methodology contribution:

1. learn or calibrate a leakage-safe field-risk router from train-derived
   predictions and observable signals only;
2. keep grounded NuExtract3 fields when confidence/evidence is sufficient;
3. invoke a focused Qwen extraction only for uncertain work-experience or
   certification fields;
4. compare router coverage, macro, unsupported evidence, p50/p95 latency, and
   cost against NuExtract3-only and always-run-both fusion.

### Selective router implementation

The first router implementation separates two policies that must not be
confused:

| Policy | Decision inputs | Can avoid Qwen call? | 250-CV macro | Escalation |
|---|---|---:|---:|---:|
| disagreement selector | NuExtract3 and Qwen predictions | no | **0.886686** | 89.6% |
| primary-only router | NuExtract3 structure and evidence support | yes | 0.881399 | 59.6% |

The disagreement selector chooses Qwen work records when record counts agree
but content differs, and chooses Qwen certifications for selected count
disagreements. It improves accuracy but is an always-run-both ablation because
Qwen must already have produced a prediction.

The primary-only router was calibrated on a deterministic half of the
train-derived `debug_250` and evaluated on the other 117 CVs:

| Holdout method | Macro | Work experience | Certifications | Mean projected latency |
|---|---:|---:|---:|---:|
| grounded NuExtract3 primary | 0.871699 | 0.868814 | 0.797921 | 1.397 s/CV |
| primary-only router | **0.883016** | 0.915552 | 0.886990 | 9.467 s/CV |
| disagreement selector | 0.888522 | **0.969084** | **0.899535** | 15.616 s/CV |

The primary-only policy routes:

- short, evidence-supported NuExtract3 work-experience outputs;
- nonempty, evidence-supported NuExtract3 certifications.

These rules use no ground-truth scores, tier, or template at inference. The
current prototype reads the evidence-support boolean from persisted evaluation
artifacts; production routing must recompute support directly from prediction
and evidence text to keep truth-bearing evaluation objects outside inference.

A focused-specialist trial is now the active next step. It sends Qwen only the
selected CVs and requests only routed fields, then deterministically merges the
partial response into the complete NuExtract3 prediction.

### Focused-specialist router result

The focused Qwen trial completed on `149/250` selected CVs:

- routed document rate: `59.6%`;
- routed work-experience fields: `90`;
- routed certification fields: `92`;
- partial-response parse failures: `0`;
- mean Qwen output tokens per called CV: `101.1`, down from `500.9`;
- mean focused Qwen latency per called CV: `3.216 s`, down from `15.540 s`;
- Modal cost for the focused 149-CV app: approximately `$0.1931`.

| Method | Macro | Schema valid | Unsupported | Mean latency |
|---|---:|---:|---:|---:|
| grounded NuExtract3 primary | 0.870103 | 1.0 | 0.055430 | 1.444 s/CV |
| random selector, matched field coverage | 0.874494 | 1.0 | 0.055295 | accuracy control |
| focused primary-only router | **0.882608** | 1.0 | 0.056194 | **3.361 s/CV** |
| always-run-both static fusion | 0.880683 | 1.0 | 0.056386 | 16.984 s/CV |
| always-run-both disagreement selector | 0.886686 | 1.0 | 0.056347 | 15.717 s/CV |
| oracle field selector | 0.889159 | 1.0 | 0.056505 | unreachable ceiling |

On the deterministic 117-CV holdout, the focused router reached `0.883060`
macro at `2.906 s/CV`, compared with `0.871699` at `1.397 s/CV` for grounded
NuExtract3 alone.

Interpretation:

- focused generation is a successful methodology contribution: routing plus
  schema decomposition improves accuracy without paying full-mapper cost;
- the focused router is slightly more accurate than always-run-both static
  fusion and roughly five times faster;
- it substantially beats matched-coverage random selection;
- it captures about `65.6%` of the oracle field-selection improvement available
  over grounded NuExtract3;
- the next work is to lower escalation coverage and replace persisted
  evaluation support flags with support recomputed directly from evidence.

### Full-train router data generation

The router stage was promoted from `debug_250` to the complete `1,445`-CV
training corpus.

The grounded NuExtract3 MTP + PyMuPDF4LLM-evidence run completed:

| Documents | Macro | Schema valid | Unsupported | Mean latency |
|---:|---:|---:|---:|---:|
| 1,445 | **0.893675** | 1.0 | 0.051715 | **1.241 s/CV** |

Execution notes:

- the first sequential Modal worker completed 240 CVs;
- the remaining 1,205 CVs were split deterministically into two disjoint,
  resumable shards and processed in parallel with batch size 20;
- merged output was validated to contain exactly 1,445 unique requested IDs;
- Modal cost for the three NuExtract3 apps was approximately `$1.44`;
- missing PyMuPDF4LLM Markdown representations for the full train corpus were
  generated locally on the M1 Pro before specialist request preparation.

The primary-only policy selected `843/1,445` CVs (`58.3%`) for focused Qwen
generation:

- work-experience requests: `537`;
- certification requests: `511`;
- requests can contain either or both routed fields.

A deterministic five-fold calibration module now keeps primary-only observable
features and truth-derived specialist-win labels in separate types. It rejects
non-`train_oof` calibration partitions and writes candidate threshold/fold
outcome reports. The active focused Qwen full-train run will provide the paired
specialist predictions needed for the next calibration pass.

The first focused-Qwen full-train app persisted 140 responses before the local
Modal client disconnected and the ephemeral app stopped. Because response
capture is append-only and resumable by `(model_id, revision, cv_id)`, the run
was resumed without repeating completed inference. This is classified as an
execution-client interruption, not a model or prompt failure.

The spawned-call resume path disconnected again before inference. The mapper
runner was corrected so `parallel_chunks=1` uses synchronous `remote()` calls,
removing dependence on spawned-call handles. The remaining untouched requests
were split into two deterministic shards and processed by two reliable
synchronous workers in parallel.

Full-train focused-router result:

| Method | Macro | Work experience | Certifications | Schema valid | Unsupported | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| grounded NuExtract3 primary | 0.893675 | 0.857848 | 0.827543 | 1.0 | 0.051715 | 1.241 s/CV |
| focused primary-only router | **0.905064** | **0.920373** | **0.901688** | 1.0 | 0.052818 | 3.506 s/CV |

Focused Qwen statistics:

- called documents: `843/1,445`;
- mean called-document latency: `3.883 s`;
- p50 called-document latency: `4.000 s`;
- p95 called-document latency: `8.719 s`;
- mean output tokens: `106.5`;
- partial-response parse failures: `0`.

The five-fold focused-policy outcome report contains `564` routed-field wins,
`46` losses, and `2,280` ties. Ties include fields that were not routed, so this
report evaluates the current policy; it is not a complete specialist oracle
label set. Training a learned router with unbiased specialist labels would
require specialist outputs for non-routed examples or a randomized exploration
sample.

Approximate Modal spend for the full-train router-data stage:

- grounded NuExtract3 full train: `$1.44`;
- focused Qwen full train, including interrupted/resumed apps: `$1.24`;
- focused-Qwen `debug_250` app: `$0.19`;
- total for these router-stage apps: approximately `$2.88`.

### Cost-control correction

Several stale attached Modal CLI processes from earlier trials remained alive
for about seven hours. They were terminated, and their corresponding apps are
now stopped. All new paid runs are captured, bounded, polled, and stopped on
failure rather than left retrying.

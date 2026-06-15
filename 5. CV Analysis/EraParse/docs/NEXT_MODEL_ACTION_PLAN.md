# Next Model Action Plan

## June 13 Next-Stage Decision

The decision-complete next-stage execution contract is now
[`NEXT_STAGE_EXPERIMENT_MATRIX.md`](NEXT_STAGE_EXPERIMENT_MATRIX.md).

It supersedes older immediate-action priorities in this document where they
conflict. In particular:

- freeze the current validation-selected deterministic repair stack;
- stop pursuing marginal repair gains on the same 310-CV validation set;
- compare `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B-Instruct-2507`, and
  `microsoft/Phi-4-mini-instruct` fairly using identical PyMuPDF4LLM evidence;
- optimize NuExtract3 through native vLLM/MTP and schema-aware compact
  serialization, then test structured-output-preserving quantization;
- build the selective grounded-to-NuExtract3 cascade from leakage-safe train
  OOF predictions;
- freeze all configurations before one-time ID/OOD evaluation.

The primary next contribution is the cascade and its
accuracy-grounding-latency-cost Pareto frontier. SG-ESE/SG-GRSE remain honest
architecture ablations, including negative results, rather than the next GPU
spending priority.

## Current Continuation Point

The SG-ESE evidence, architecture, deterministic assembly, local MPS runner,
and budget-safe Modal smoke foundation are implemented. The complete
implementation/trial record is in
[`SG_ESE_STAGE_LOG.md`](SG_ESE_STAGE_LOG.md).

The first promoted NuExtract3 upper-bound run is now real, not just planned:

- raw `debug_50`: `0.845962` macro, `0.725867` work-experience score,
  `0.000000` schema validity, `0.047000` unsupported evidence;
- raw + deterministic contract assembly on the exact same 50 responses:
  `0.857562` macro, `0.865067` work-experience score, `1.000000` schema
  validity, `0.047000` unsupported evidence;
- raw full validation: `0.869937` macro, `0.730132` work-experience score,
  `1.000000` JSON validity, `0.000000` schema validity, `0.035980`
  unsupported evidence, `17.76 s/CV`;
- repaired full validation on the same 310 responses: `0.881002` macro,
  `0.862906` work-experience score, `1.000000` JSON validity,
  `1.000000` schema validity, `0.035980` unsupported evidence,
  `17.76 s/CV`.

This changes the comparison story in an important way: a modern structured VLM
is already competitive on accuracy, but it needs deterministic local assembly
to satisfy the thesis schema contract reliably.

The practical full-loss SG-ESE lane has now also been run on real
train/validation data and did **not** beat the current modular practical lane:

- full-loss practical SG-ESE: `0.688165` validation macro;
- current best practical modular lane:
  `0.776740` validation macro with 100% schema validity and 0% unsupported
  evidence.

This means the thesis now has two active contribution tracks:

1. architecture edits that still need either accuracy recovery or a strong
   efficiency case;
2. modular grounded decoding and deterministic repair that already produce the
   current strongest held-out practical result.

The fair two-epoch practical comparison is complete: standard LayoutLMv3
reached `0.739158` validation macro and SG-ESE query-only reached `0.738284`,
both with 100% schema validity and 0% unsupported evidence. The `0.000875`
difference makes SG-ESE competitive enough to begin sparse-token efficiency
work without claiming an accuracy win. The user has made approximately
`$15-20` available for selected promoted Modal trials.

The next accuracy-oriented architecture is SG-GRSE, and the next independent
methodology contribution is EFSFR. Their decision-complete plan is in
[`SECOND_CONTRIBUTION_PLAN.md`](SECOND_CONTRIBUTION_PLAN.md).

The first practical oracle-ceiling measurement is complete:

- oracle labels + sequence grouping: `0.748347` validation macro;
- oracle labels + oracle grouping: `0.755676` validation macro;
- current best learned baseline: `0.739158`.

This confirms that practical evidence quality is sufficient for another
accuracy gain and that repeated-record decoding/grouping is a meaningful
bottleneck.

The first explicit EFSFR methodology slice is also now validated on held-out
validation:

- standard LayoutLMv3: `0.739158` macro, `0.627738` work experience;
- LayoutLMv3 + deterministic work-record repair: `0.763803` macro,
  `0.923473` work experience;
- schema validity stayed at `1.000000`;
- unsupported evidence is `0.000000` after fixing the multi-page evaluation
  aggregation bug in `evaluate_grounded_rows`.

The next broader EFSFR slice is now also validated on held-out validation:

- full EFSFR nested repair: `0.775538` macro;
- `education`: `0.837617`;
- `projects`: `0.645540`;
- `certifications`: `0.911643`;
- `work_experience`: `0.923473`;
- schema validity: `1.000000`;
- unsupported evidence: `0.000000`.

This means the current best practical accuracy result is no longer the raw
LayoutLMv3 baseline or the work-only repair lane. SG-GRSE now has to beat the
`0.775538` EFSFR lane to become the primary thesis contribution.

The first SG-GRSE work-only decoder is now also validated:

- work-only SG-GRSE: `0.764074` macro, `0.926728` work experience;
- full EFSFR + SG-GRSE work: `0.775809` macro;
- delta over full EFSFR alone: `+0.000271`;
- schema validity: `1.000000`;
- unsupported evidence: `0.000000`.

This is a real win, but only a narrow one. The next SG-GRSE work should focus
on making the gain more robust and easier to defend, not merely preserving a
tiny edge.

The first stability check is now also complete:

- EFSFR vs EFSFR + SG-GRSE work mean macro delta: `+0.000271`;
- document-level wins / losses / ties: `20 / 11 / 279`;
- 95% bootstrap CI for the mean macro delta:
  `[-0.000519, 0.001070]`.

So the current SG-GRSE gain is not yet statistically secure. The next step is
not just “more score”; it is to widen the margin enough that the confidence
interval stops straddling zero.

That next step is now partially complete: a selective SG-GRSE work gate is
validated on held-out data.

- selective EFSFR + SG-GRSE work: `0.776183` macro;
- work-experience score: `0.931212`;
- SG-GRSE accepted on `7 / 310` CVs;
- mean macro delta vs full EFSFR: `+0.000645`;
- macro wins / losses / ties: `7 / 0 / 303`;
- macro 95% bootstrap CI: `[0.000169, 0.001270]`.

This is the first SG-GRSE-based lane with a validation bootstrap interval that
stays above zero. The gain is still small, but the result is much more
defensible than the unconditional SG-GRSE replacement.

Persist every SG-ESE/LayoutLMv3 result with a `summary.json`, then refresh the
comparison artifacts using:

```bash
uv run eraparse sge report-trials \
  --root artifacts/sge/local_smokes \
  --output artifacts/sge/trial_comparison.json
```

Training-only scores remain diagnostics. The first priority is a held-out
validation score under equal data, update steps, seed, unfreeze depth, and
decoder-selection rules.

Current immediate order:

1. freeze the current repair stack and historical validation results;
2. treat BF16 NuExtract3 + two-token MTP as the promoted serving baseline;
3. compare official NuExtract3 W8A8 and W4A16 checkpoints on complete-schema
   `debug_50`, then promote at most one to `debug_250`;
4. freeze the selected NuExtract3 precision/runtime configuration;
5. generate leakage-safe train OOF predictions and build the selective
   grounded-to-NuExtract3 cascade;
6. freeze selected configurations before one-time ID/OOD evaluation.

PaddleOCR-VL, Donut decomposition, and further SG-GRSE work are deferred until
the cascade has a frozen ID/OOD result. They remain useful comparisons, but
opening them now would dilute the central thesis flow and spend budget before
the highest-value unbuilt system is evaluated.

## Revised Research Question

The next stage asks three questions before visual-token compression:

1. can each visual reader recover grounded evidence;
2. can an extractor assign that evidence to scalar and repeated CV fields;
3. which assembly/decoding strategy reliably creates the final schema?

SG-VTC remains conditional on a successful full-token visual baseline.
The main architecture direction is now SG-ESE, described in
`REARCHITECTED_RESEARCH_FLOW.md`.

## Priority Model Lanes

### 1. NuExtract3 Structured Visual Upper Bound

Use `numind/NuExtract3` first. It is a 4.54B structured-extraction VLM based on
Qwen3.5-4B and directly accepts an extraction template.

- revision checked June 11, 2026:
  `acaf70ecff9c3dbbfcbae651b82b66a0d8dbd0c6`;
- role: strongest direct test of whether a modern structured VLM can solve the
  current schema without project-specific fine-tuning;
- prepared request bundles already exist:
  `artifacts/trials/nuextract3/debug_50.jsonl` and
  `artifacts/trials/nuextract3/validation.jsonl`;
- run sequence: 10-record smoke, `debug_50`, then all 310 validation CVs only
  if schema validity and generation-length gates pass.

Source: <https://huggingface.co/numind/NuExtract3>

### 2. PaddleOCR-VL 1.6 Representation Upper Bound

Use `PaddlePaddle/PaddleOCR-VL-1.6` as a page-level document parser, then map
its Markdown/structured output with the existing frozen Qwen3 mapper.

- revision checked June 11, 2026:
  `66317acc4c9fc17bd154591ce650735cd2855f3e`;
- role: separates modern visual parsing/OCR quality from schema-generation
  quality;
- prepared request bundles already exist:
  `artifacts/trials/paddleocr_vl/debug_50.jsonl` and
  `artifacts/trials/paddleocr_vl/validation.jsonl`;
- compare directly with PyMuPDF/Tesseract fallback using the same mapper.

Source: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6>

### 3. Donut Complexity Ablations

Do not run another full Donut fine-tune yet. First require:

1. overfit 10-20 training CVs and generate nearly perfect predictions on those
   same CVs;
2. contact-only target;
3. flat-core target: contact, summary, and skills;
4. experience-only nested target;
5. task-decomposed generation with deterministic merging;
6. page-wise processing compared with vertically stacked pages.

This identifies whether the failure is implementation, nested-schema
complexity, multi-page resizing, or Donut-base capacity.

### 4. Non-Generative Layout Baseline

Implement `microsoft/layoutlmv3-base` source-oracle token classification before
additional Donut full runs. It avoids long autoregressive JSON generation and
tests whether explicit words and boxes can recover fields.

Revision checked June 11, 2026:
`cfbbbff0762e6aab37086fdd4739ad14fe7d5db4`.

Source: <https://huggingface.co/microsoft/layoutlmv3-base>

### 5. Cheap Optional Visual Parser

`ibm-granite/granite-docling-258M` is a useful low-cost page-parser comparator,
not the primary structured CV extractor.

Revision checked June 11, 2026:
`982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe`.

Source: <https://huggingface.co/ibm-granite/granite-docling-258M>

## Promotion Gates

Every model must pass:

1. local serialization and evaluator tests;
2. remote model-load smoke;
3. 10-record generation test;
4. `debug_50` with at least 95% schema validity and fewer than 5% token-cap
   hits;
5. full 310 validation only after the debug gate;
6. protected ID/OOD only after configuration freezing.

Fine-tuned models additionally require a successful tiny-set overfit test.
Near-`0.90` tiny-overfit is a diagnostic target, not a universal hard stop.
Promotion may proceed below it when contracts are sound and the trial is a
fair, informative comparison or ablation.

## GPU Strategy

Do not use `T4:2` as the default replacement for one A10. Two T4s cost slightly
more than one A10 at current Modal prices, each has only 16 GB VRAM, and the
current code would use only GPU 0. Their memory is not automatically combined.

- Qwen3, NuExtract Tiny, PaddleOCR-VL, and Granite-Docling inference: smoke on
  one L4 first, then one A10 only if compatibility or performance requires it.
- NuExtract3 inference: smoke the quantized `numind/NuExtract3-W4A16` on one L4
  or A10; use one L40S for the unquantized upper bound if 24 GB is insufficient.
- Donut inference: one L4 or A10 after a memory smoke.
- Donut training: test one L40S before A100-40GB. Use multiple GPUs only after
  implementing and validating DDP; batch-size-one training does not benefit
  automatically.
- Parallel document inference: scale independent one-GPU Modal workers instead
  of requesting a multi-GPU container.

Current Modal base GPU prices checked June 11, 2026:

| GPU | VRAM | Approximate hourly GPU cost |
|---|---:|---:|
| T4 | 16 GB | $0.59 |
| L4 | 24 GB | $0.80 |
| A10 | 24 GB | $1.10 |
| L40S | 48 GB | $1.95 |
| A100-40GB | 40 GB | $2.10 |

CPU and memory charges are additional.

## Immediate Action Order

Current strongest practical held-out lane:

| Method | Macro | Projects | Work experience | Schema valid |
|---|---:|---:|---:|---:|
| selective EFSFR + SG-GRSE work + project technology repair + train-derived project URL selector | **0.776740** | **0.652224** | **0.931212** | **1.000000** |

Notes:

- the project-technology repair was positive on `2 / 310` validation CVs and
  harmful on `0`;
- the train-derived project-URL selector improved macro by `+0.000418` and
  projects by `+0.005018`;
- unconditional project-URL synthesis was tested and rejected because some
  ground-truth project records intentionally have no URL;
- the first full-loss practical SG-ESE run on train/validation reached
  `0.688165` macro, so the current model-side architecture lane is still behind
  the practical extractor-plus-repair lane.

1. Freeze this current practical lane; do not add more validation-selected
   repair rules.
2. Execute official NuExtract3 W8A8/W4A16 quantization smokes and matched
   complete-schema `debug_50` trials against BF16 + MTP.
3. Promote at most one quantized configuration to `debug_250`, or retain BF16
   + MTP if quantization fails the semantic-accuracy gates.
4. Select one optimized NuExtract3 configuration and generate leakage-safe
   train OOF predictions within the `$15` execution cap.
5. Build the selective grounded-to-generative cascade and compare it with
   random escalation and always-NuExtract3.
6. Freeze selected methods before protected ID and template-OOD evaluation.
7. Defer additional model families and architecture recovery campaigns until
   the cascade has a frozen result.

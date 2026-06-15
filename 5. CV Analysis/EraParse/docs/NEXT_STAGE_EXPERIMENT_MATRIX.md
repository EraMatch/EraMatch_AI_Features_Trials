# Next-Stage Experiment Matrix

## Decision

The next thesis stage freezes the current deterministic repair stack and
concentrates on three connected approach families:

1. **fair parser-to-mapper scaling** using the same PyMuPDF4LLM evidence;
2. **NuExtract3 inference acceleration** using native serving, multi-token
   prediction (MTP), structured-output-preserving quantization, and
   serialization ablations;
3. **a selective grounded-to-generative cascade** that invokes NuExtract3 only
   when the fast grounded lane is uncertain.

This stage does not add more validation-selected deterministic repairs. Existing
SG-ESE and SG-GRSE results remain reportable architecture ablations, including
their negative and near-tie outcomes, but they are not the next spending
priority.

Current promotion decision after the matched `debug_250` runs:

- promote NuExtract3 MTP + PyMuPDF4LLM evidence as the accurate/fast
  generative lane (`0.870103`, `1.444 s/CV`);
- retain Qwen3-4B as a complementary grounded specialist for work experience
  and certifications (`0.925185` and `0.866536`);
- use the always-run-both field fusion (`0.880683`) as the accuracy target for
  the selective field router;
- do not promote Phi-4 Mini, opaque compact schemas, W8A8 under vLLM `0.21.0`,
  or W4A16 as an A10 speed optimization.
- promote the focused primary-only field router as the current practical
  methodology contribution: `0.882608` macro at `3.361 s/CV` on the fixed
  train-derived `debug_250`, with 100% schema validity.
- full-train scaling confirmed the contribution on all 1,445 training CVs:
  focused routing improved grounded NuExtract3 from `0.893675` to `0.905064`
  macro at `3.506 s/CV`, with 100% schema validity.

## Central Thesis Question

> How should a CV extraction system trade model capacity, visual grounding,
> structured-output reliability, latency, and GPU cost while preserving the
> complete target schema?

The experiment flow is:

```text
Approach A: PyMuPDF4LLM -> text mapper -> deterministic assembly
Approach B: CV images -> NuExtract3 -> deterministic assembly
Approach C: fast grounded lane -> risk router -> selective NuExtract3

Within Approach B:
baseline serving -> native vLLM -> vLLM + MTP
BF16 -> W8A8 -> W4A16
full semantic JSON -> experimental compact schema -> deterministic expansion
```

## Frozen Historical References

The following validation results are historical references. They must not be
used for further threshold, prompt, repair-rule, or router selection:

| Method | Validation macro | Schema valid | Unsupported evidence | Mean latency |
|---|---:|---:|---:|---:|
| Qwen3-0.6B + PyMuPDF/Tesseract | 0.670902 | 0.996774 | 0.040981 | 13.3212 s/CV |
| standard LayoutLMv3 + PyMuPDF4LLM | 0.739158 | 1.000000 | 0.000000 | model-only 0.1663 s/CV |
| frozen practical modular lane | 0.776740 | 1.000000 | 0.000000 | estimated uncached 0.7179 s/CV |
| NuExtract3 + deterministic assembly | **0.881002** | **1.000000** | 0.035980 | 17.7595 s/CV |

The current repair stack is frozen at `0.776740`. Its small validation-selected
gains must be described as exploratory until they survive frozen ID/OOD
evaluation.

## Leakage-Safe Data Flow

### Development Pools

| Pool | Size | Permitted use |
|---|---:|---|
| `debug_50` | 50 train CVs | model-load, contract, and generation smoke |
| `debug_250` | 250 train CVs | prompt, serving, serialization, and mapper selection |
| train | 1,445 CVs | final fitting and out-of-fold router feature generation |
| historical validation | 310 CVs | frozen historical comparison only; no new selection |

`debug_250` must be the same fixed train-derived sample for every next-stage
model. A model may not receive a more favorable subset.

### Router Training

Generate out-of-fold (OOF) predictions on all 1,445 training CVs:

1. use five deterministic folds;
2. fit any confidence calibrator or learned router on four folds;
3. predict only the held-out fold;
4. concatenate held-out predictions and train the final router from those OOF
   records;
5. prohibit ground-truth tier, template, field score, or correctness at
   inference time.

If NuExtract3 predictions for all 1,445 training CVs exceed the approved budget,
start with a deterministic stratified OOF subset and report its coverage. Do not
substitute validation predictions.

### Frozen Evaluation

After all prompts, revisions, thresholds, compact schemas, and router parameters
are frozen:

1. run all 310 ID-test CVs once;
2. run all 410 template-OOD CVs once;
3. report the two pools separately and combined;
4. leave `locked_confirmation` inaccessible until the final thesis method is
   selected.

No ID/OOD result may trigger a configuration change. Any post-ID/OOD change
creates a new research iteration and invalidates those pools as untouched final
tests.

## Approach A: Fair PyMuPDF4LLM Mapper Scaling

### Purpose

The current parser-to-mapper comparison uses a roughly 0.8B mapper against a
4.54B structured VLM. This experiment controls for mapper capacity by using the
same reader output, complete target schema, prompt, generation policy,
deterministic assembly, and evaluator across three mapper sizes/architectures.

### Required Models

| Model ID | Approximate role | Required mode |
|---|---|---|
| `Qwen/Qwen3-0.6B` | frozen small-mapper reference | `enable_thinking=False` |
| `Qwen/Qwen3-4B-Instruct-2507` | primary size-matched mapper | non-thinking deterministic extraction |
| `microsoft/Phi-4-mini-instruct` | independent approximately 3.8B mapper | deterministic instruct generation |

Pin a Hugging Face revision before each reportable run. If an exact model ID or
runtime is unavailable at execution time, log the failure and select no silent
substitute.

Sources:

- <https://huggingface.co/Qwen/Qwen3-0.6B>
- <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>
- <https://huggingface.co/microsoft/Phi-4-mini-instruct>

### Controlled Input And Output

- reader: cached PyMuPDF4LLM Markdown from identical source hashes;
- smart-OCR configuration: identical and pinned;
- target: complete reduced CV schema, including every scalar and nested array;
- prompt: semantically identical, with only model-required chat formatting
  differences;
- decoding: greedy or temperature zero, recorded token cap, no thinking;
- output: raw response plus the same deterministic contract assembly;
- evaluator: the canonical full-schema evaluator.

### Mapper Trial Sequence

| Gate | Samples | Decision |
|---|---:|---|
| load/one-record smoke | 1 | confirm revision, memory, and valid response capture |
| contract smoke | `debug_50` | reject repeated crashes, truncation, or unusable outputs |
| fair selection run | `debug_250` | select at most one 3-4B mapper for promotion |
| frozen final run | 310 ID + 410 OOD | run only the selected mapper and frozen 0.6B reference if needed |

Promotion from `debug_250` requires either:

- at least `+0.03` macro over Qwen3-0.6B at no more than `2x` its measured
  latency/cost; or
- at least `+0.05` macro with a defensible accuracy-latency Pareto tradeoff.

Phi-4 Mini and Qwen3-4B are comparisons, not both mandatory full-test models.
Promote the stronger one after the fixed `debug_250` comparison.

### Mapper Scientific Questions

1. Does a 3-4B mapper close the accuracy gap to a similarly sized visual
   structured extractor?
2. Which errors remain attributable to reader evidence rather than mapper
   capacity?
3. Does visual input still provide an advantage after controlling approximate
   model size?

## Approach B: NuExtract3 Speed And Serialization

### Frozen Baseline

The baseline is the existing `numind/NuExtract3` Transformers visual run plus
deterministic contract assembly:

- validation macro: `0.881002`;
- schema validity after assembly: `1.000000`;
- unsupported-evidence rate: `0.035980`;
- mean latency: `17.7595 s/CV`.

This baseline remains unchanged. New experiments must use identical images,
schema semantics, decoding determinism, and evaluation.

Source: <https://huggingface.co/numind/NuExtract3>

### Optimization 1: Native vLLM And Multi-Token Prediction

MTP proposes multiple future tokens and verifies them with the main model,
reducing sequential autoregressive decode steps when proposals are accepted.
For this thesis, MTP is an inference-system optimization, not a claimed new
architecture.

Required ablation:

| Variant | Isolated question |
|---|---|
| existing Transformers baseline | frozen reference |
| native NuExtract3 template through vLLM, MTP off | effect of serving engine/native interface |
| same vLLM configuration, MTP on | incremental effect of MTP |

Keep model revision, precision, input images, compact/full schema, token cap,
and batching policy fixed between the MTP-off and MTP-on rows. Record speculative
acceptance statistics when the runtime exposes them.

MTP succeeds when it:

- preserves macro within `0.005` and all field scores within reported
  confidence intervals;
- preserves 100% post-assembly schema validity;
- reduces p50 latency by at least 15% or cost/CV by at least 15%.

### Optimization 2: Schema-Aware Compact Serialization

Define a lossless compact JSON target with short keys and positional nested
records, then deterministically expand it to the canonical schema.

Example:

```json
{"n":"A Name","e":"a@example.com","w":[["Engineer","Company","2021","2024"]]}
```

The compact representation must:

- represent every canonical field, including nullable arrays and repeated
  records;
- distinguish absent, null, and empty where the canonical contract requires it;
- round-trip canonical target -> compact target -> canonical target without
  semantic loss;
- preserve raw output and every expansion/repair event;
- use the same semantic extraction instructions as the full-schema baseline.

Required ablation:

| Serving | Serialization | Purpose |
|---|---|---|
| vLLM, MTP off | full schema | engine reference |
| vLLM, MTP off | compact schema | compact-schema effect |
| vLLM, MTP on | full schema | MTP effect |
| vLLM, MTP on | compact schema | combined speed configuration |

Compact serialization succeeds when it:

- reduces generated tokens by at least 25%;
- preserves macro within `0.005`;
- preserves 100% post-expansion schema validity;
- reduces mean or p50 latency by at least 15%.

The first opaque-alias compact-schema smoke is now rejected: it reduced output
tokens by `55.5%` and generation time by `12.4%`, but macro collapsed to
`0.083333` because the model reassigned values across semantically opaque keys.
Any future compact schema must preserve semantic names or use a trained,
explicitly described coding contract.

### Optimization 3: Structured-Output-Preserving Quantization

Quantization is evaluated as a complete-schema extraction intervention, not
only a memory benchmark. Lower precision can preserve JSON syntax while
changing field assignment, repeated-record boundaries, dates, or abstention
behavior. Therefore JSON validity alone is not a sufficient success criterion.

Start with official NuMind checkpoints to avoid confounding quantization
quality with a project-created calibration pipeline:

| Variant | Pinned revision | Role |
|---|---|---|
| `numind/NuExtract3` BF16 + MTP | `acaf70ecff9c3dbbfcbae651b82b66a0d8dbd0c6` | promoted accuracy/speed reference |
| `numind/NuExtract3-W8A8` + MTP | `e9ffaea6c5cbf2bed066dcc6b193fb608b8bdcf7` | accuracy-preserving quantization candidate |
| `numind/NuExtract3-W4A16` + MTP | `b5028670152c8130a3f362b66981eee16612b7f6` | aggressive memory/speed candidate |
| `numind/NuExtract3-FP8` | `d88964bad5ba47333cb721b351e19045ee6a6fc0` | deferred Ada/Hopper comparison |

The initial A10/T4-class lane evaluates W8A8 and W4A16. FP8 is deferred because
its strongest hardware acceleration is intended for Ada/Hopper-class GPUs; it
must not be compared on unsuitable hardware and called a quantization failure.

Required matched measurements:

- raw JSON validity and post-assembly schema validity;
- full-schema macro and every field score;
- nested work/education/project/certification scores;
- unsupported-evidence rate and abstention behavior;
- missing/extra keys, repair events, and token-cap rate;
- generated-token count, mean/p50/p95 latency, throughput, peak GPU memory,
  startup time, and cost/CV.

Quantization succeeds only when it:

- preserves macro within `0.005` of BF16 + MTP;
- preserves every nested-field score within `0.01`;
- preserves 100% post-assembly schema validity;
- does not increase unsupported-evidence rate by more than `0.005`;
- reduces warm latency, cost/CV, or required GPU memory by at least `15%`.

Trial sequence:

1. one-record model-load and complete-schema smoke for W8A8 and W4A16;
2. matched `debug_50` comparison against BF16 + MTP;
3. promote at most one quantized configuration to `debug_250`;
4. if both fail accuracy gates, retain BF16 + MTP for the cascade;
5. only after official checkpoints are measured, consider project-created
   mixed-precision quantization that preserves vision encoder, embeddings,
   normalization, and output head at higher precision.

The project-created mixed-precision variant is a possible later methodological
contribution, but it requires calibration and an ablation against the official
checkpoints. It is not the immediate next paid run.

### NuExtract3 Trial Sequence

1. prove canonical/compact round-trip locally without GPU;
2. run one-record vLLM load and native-template smoke;
3. run MTP, serialization, and official quantization cells on identical
   `debug_50`;
4. promote viable MTP/quantization cells to identical `debug_250`;
5. select one optimized NuExtract3 configuration;
6. use the selected configuration for OOF generation and the cascade;
7. run the frozen selected configuration on ID/OOD.

Do not rerun all 310 historical validation CVs for every speed variant. The
existing validation row is the frozen historical reference; next-stage
selection happens on train-derived `debug_250`.

## Approach C: Selective Grounded-To-Generative Cascade

### Purpose

Combine the current fast grounded lane with the optimized NuExtract3 lane:

```text
PyMuPDF4LLM + grounded extractor
  -> observable risk/confidence router
  -> accept grounded fields when reliable
  -> escalate uncertain CVs or bundled uncertain fields to NuExtract3
  -> verify generated values against grounded evidence
  -> deterministic canonical assembly
```

The cascade is the primary systems contribution. It should approach NuExtract3
accuracy while lowering average latency/cost and bounding unsupported evidence.

### Router Inputs

Permitted observable features include:

- field-presence probabilities, margins, and entropy;
- evidence-span support and token coverage;
- disagreement between grounded heads;
- schema consistency and missing required fields;
- parser/OCR confidence, text density, page count, and truncation indicators;
- model response validity and token-cap events.

Forbidden features include ground-truth tier, template, true field values,
per-document field F1, and any test-set-derived threshold.

Bundle uncertain fields into one NuExtract3 request per CV. A separate expensive
call for each field is prohibited unless an ablation demonstrates its value.

### Cascade Comparisons

| Variant | Purpose |
|---|---|
| fast grounded lane only | speed/faithfulness anchor |
| optimized NuExtract3 always | accuracy anchor |
| random escalation at matched rate | router negative control |
| rule-based observable-risk router | interpretable baseline |
| learned OOF router | proposed selector |
| learned router + grounded verification | hallucination-control contribution |

Required risk/coverage points: approximately 10%, 20%, 30%, 50%, and 100%
NuExtract3 escalation. Report the achieved rates, not only requested thresholds.

Primary target:

- macro at least `0.85`;
- NuExtract3 invoked for no more than 30% of CVs;
- 100% schema validity;
- unsupported-evidence rate below always-NuExtract3;
- materially lower mean latency and cost than always-NuExtract3.

## Reporting Contract

Every reportable trial must persist:

- run ID, timestamp, code revision, model ID and pinned revision;
- source manifest hash and exact CV IDs;
- resolved prompt/template, serialization, decoding, and assembly configuration;
- hardware, runtime, precision, batch/concurrency, cold/warm-start boundary;
- raw responses, expanded predictions, validation/repair events, and errors;
- complete-schema aggregate and per-field scores;
- JSON validity, schema validity, unsupported-evidence rate, ANLS, and nested
  Hungarian-matching scores;
- input tokens, generated tokens, token-cap rate, and tokens/second;
- latency mean, p50, p95, and total wall time;
- GPU/CPU/memory measurements where available;
- projected cost, actual Modal cost, and cost/CV;
- comparison baseline and absolute/relative deltas;
- promotion, rejection, or failure decision with reason.

Speed measurements must distinguish:

1. cached reader/model-only latency;
2. warm end-to-end latency;
3. cold-start/container latency;
4. batch throughput.

All positive next-stage claims require document-level bootstrap confidence
intervals on frozen ID/OOD. Small deltas selected on historical validation are
not promoted as thesis contributions.

## Cost And Execution Budget

The user authorized approximately `$15-20` for the next promoted Modal trials.
Use a `$15` execution cap and preserve `$5` as reserve.

Observed NuExtract3 validation cost was approximately `$2.09 / 310`, or about
`$0.00674/CV`. This is a planning reference, not a guaranteed future price.

| Activity | Initial cap |
|---|---:|
| 3-4B mapper smokes and identical `debug_250` comparison | $3.00 |
| NuExtract3 vLLM/MTP/quantization `debug_50` and selected `debug_250` | $4.00 |
| stratified or full train OOF NuExtract3 generation | $6.00 |
| cascade smoke and frozen reporting support | $2.00 |
| **Execution cap** | **$15.00** |
| **Preserved reserve** | **$5.00** |

At the observed baseline rate, always-NuExtract3 inference over 1,445 train CVs
would cost approximately `$9.75`, before safety factor. Therefore full OOF
generation is not launched until an optimized configuration has a measured
cost and the remaining stage budget allows:

```text
projected cost =
  measured cost per CV
  * planned CV count
  * 1.30 safety factor
```

Stop before launch when projected cumulative stage spend exceeds `$15`.
Prefer local M1 Pro work for schemas, round-trip tests, cached evidence,
evaluation, router fitting, and report generation. Use Modal GPU only for model
inference that is not practical locally.

## Decision-Complete Execution Order

1. Freeze the current repair stack and historical validation table.
2. Implement and locally verify the compact-schema round trip.
3. Prepare one identical PyMuPDF4LLM `debug_250` mapper bundle.
4. Run Qwen3-0.6B, Qwen3-4B-Instruct-2507, and Phi-4 Mini under the fair mapper
   contract; select at most one larger mapper.
5. Treat BF16 + two-token MTP as the promoted serving baseline.
6. Run official W8A8 and W4A16 complete-schema smokes and matched `debug_50`;
   promote at most one quantized cell to `debug_250`.
7. Freeze the optimized NuExtract3 serving/precision configuration.
8. Generate leakage-safe train OOF predictions within the approved cost cap.
9. Train and compare random, rule-based, learned, and grounded-verification
   cascades on OOF data.
10. Freeze one mapper lane, one optimized NuExtract3 lane, and one cascade.
11. Run frozen ID and template-OOD evaluation once.
12. Use locked confirmation only after final thesis-method selection.

## Stop Rules

- Stop a mapper after `debug_50` if it repeatedly fails contracts or truncates
  more than 10% of outputs.
- Stop a mapper after `debug_250` if it provides no meaningful accuracy or
  Pareto improvement over Qwen3-0.6B.
- Stop MTP if it changes semantic output beyond the allowed accuracy tolerance
  or does not improve latency/cost by at least 15%.
- Stop compact serialization if round-trip fidelity is not exact or generated
  tokens fall by less than 25%.
- Stop a quantized lane if macro falls by more than `0.005`, any nested field
  falls by more than `0.01`, or unsupported evidence rises by more than
  `0.005`.
- Stop full OOF generation when projected stage spending exceeds the cap.
- Do not open another broad model-training campaign until the cascade has a
  frozen ID/OOD result.

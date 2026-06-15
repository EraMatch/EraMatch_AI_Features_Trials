# Trial Runbook

## Promotion Policy

Every new parser, model, architecture change, or router starts on a fixed
training-only debug subset:

- `debug_50`: 10 samples per tier from train.
- `debug_250`: optional, 50 samples per tier from train.

A method may move to full validation only when it:

- completes without unhandled errors;
- records reproducibility metadata and resource metrics;
- produces parseable, schema-validated outputs;
- passes subsystem-specific tests and stop gates.

Promotion is comparative, not controlled by one universal accuracy cutoff.
Tiny-set overfit near `0.90` remains a useful implementation diagnostic, but a
method may continue when it produces scientifically useful evidence such as:

- a fair architecture comparison with identical data and fine-tuning budget;
- a clear accuracy, grounding, validity, latency, memory, or token-count
  advantage;
- an informative ablation or negative control that isolates a claimed
  contribution;
- a failure whose cause is localized and documented.

Do not promote broken contracts, invalid outputs, leakage, or unbounded cost.
Do not suppress a sound comparison merely because it misses an arbitrary
single score.

ID test, template-OOD test, and locked confirmation are not used for selection.

These small subsets are engineering and promotion gates, not the final
experiment:

- `debug_50` catches broken prompts, invalid JSON, dependency failures, runaway
  generation, and obvious tier-specific failures cheaply.
- `debug_250` checks whether the result is stable enough to justify a larger
  run and exposes rarer failures.
- validation uses all 310 validation samples for method selection.
- after freezing methods, final working-corpus evaluation uses all 310 ID-test
  and all 410 template-OOD samples.
- the complementary 2,475-sample locked corpus is used once for final
  confirmation.

The complete 2,475-sample working corpus is used across training, selection,
and final testing. It is intentionally not treated as one interchangeable test
set: 1,445 samples are training data, 310 are validation, and 720 are protected
final tests.

## Trial 0: Dataset Audit

Generate and validate the manifest and splits. Abort all downstream trials if
counts, artifacts, hashes, or leakage checks fail.

## Trial 1: Parser Representation Study

Mandatory representations:

1. precomputed PyMuPDF text;
2. precomputed pdfminer text;
3. PyMuPDF4LLM Markdown;
4. PyMuPDF4LLM JSON/layout output;
5. Docling default Markdown;
6. Docling OCR configuration for T4;
7. Docling table configuration for T3;
8. Docling structured JSON/dict export;
9. `text_ground_truth` as an oracle input.

Optional only after mandatory coverage: pdfplumber for T3, Unstructured, and
Granite-Docling.

Select the best two practical inputs on validation. Do not run a full
parser-by-mapper cross-product.

## Trial 2: Text Mapper Ladder

### 2A: Input-Format Ablation

Run `numind/NuExtract-1.5-tiny` across mandatory parser representations and the
oracle text input. Select practical representations using field score, support,
validity, latency, and failure rate.

### 2B: Mapper Comparison

On the best two practical inputs, compare:

- `numind/NuExtract-1.5-tiny`;
- `Qwen/Qwen3-0.6B` in non-thinking mode;
- `microsoft/Phi-4-mini-instruct`, conditional on a validated quantized lane;
- `gemma3:12b-cloud` on selected subsets only.

### 2C: Section Decomposition

For Qwen3 and conditional Phi-4 Mini, compare one-shot extraction with:

- contact;
- experience;
- education and certifications;
- skills and projects;
- deterministic merge into the reduced schema.

## Trial 2D: ATS Compatibility And Screening Baselines

Follow `ATS_BASELINE.md`.

Compare:

- deterministic Boolean keyword filtering over raw/precomputed text;
- deterministic BM25 ranking over raw/precomputed text;
- pinned OpenCATS ingestion and full-text search as a legacy system baseline;
- the same deterministic eligibility and ranking logic over EraParse canonical
  and predicted structured fields;
- optional OpenResume parser output as an ATS-readability comparator.

Use debug pools only to validate ingestion, queries, and metric contracts.
Select screening/query rules on train and validation only. After freezing,
report ID and template-OOD candidate pools separately, then use locked
confirmation once.

Do not claim that OpenCATS represents all commercial ATS products. Do not use
name, email, phone, LinkedIn, GitHub, or other identity/contact fields as
ranking signals.

## Trial 3: Direct VLMs

- Run the PyMuPDF-with-Tesseract-fallback Qwen3 lane first as the mandatory OCR
  recovery control.
- Fine-tune `naver-clova-ix/donut-base` for the reduced schema.
- Evaluate `numind/NuExtract3` and `PaddlePaddle/PaddleOCR-VL-1.6` as modern
  upper bounds.
- Keep PP-DocBee and Granite-Docling optional.

Profile encoder, decoder, postprocess, total latency, peak memory, input size,
and visual-token counts before SG-VTC.

## Trial 4: SG-VTC

First run a compatibility spike:

1. obtain Donut encoder outputs;
2. shorten `last_hidden_state`;
3. wrap it in a compatible `BaseModelOutput`;
4. pass the matching attention mask to decoder/generation;
5. verify decoding, gradients where relevant, and timing instrumentation.

Start at `keep_ratio=0.50`:

- full tokens;
- `random_50`, negative control;
- `norm_50`;
- `oracle_50`, true field boxes;
- `global_schema_prior_50`, deployable;
- `predicted_class_prior_50`, practical when a router predicts class.

True template/tier priors are oracle-only. Test 0.75/0.25 ratios or a learned
selector only when the 50% results justify them.

Decision rules:

- If oracle 50% pruning loses unacceptable accuracy, reject reliable
  compression for this task.
- If oracle works but practical schema priors fail, region selection is the
  bottleneck.
- If practical schema priors work, SG-VTC becomes the primary architecture
  hypothesis.
- Interpret latency through `decoder_ms / total_ms`; post-encoder compression
  does not reduce encoder cost.

## Trial 5: LayoutLMv3

- Mandatory: source-oracle lane using existing source-derived boxes.
- Conditional: OCR-realistic lane after OCR boxes and label alignment exist.
- Chunk inputs over 512 tokens and reassemble predictions deterministically.
- Assemble JSON only from predicted evidence spans.
- Compare standard LayoutLMv3 and SG-ESE with identical reader, records, seed,
  trainable encoder layers, optimization steps, and decoder alternatives.
- Required SG-ESE ablations: frozen versus final-four-layer fine-tuning,
  schema queries, learned versus deterministic record grouping, evidence loss,
  presence/abstention loss, and practical versus oracle evidence.
- Report both wins and negative contributions. A component is retained only
  when its measured benefit or scientific explanatory value justifies it.

## Trial 6: Hybrid Router

Compare:

- an oracle router as an upper bound;
- a practical router using only observable parse/OCR confidence, text density,
  JSON/schema validity, evidence support, model confidence/logprobs, repair
  flags, latency, and document signals;
- conditional cloud repair for invalid or low-confidence outputs.

## Final Evaluation

Freeze selected methods and hyperparameters before:

1. ID test evaluation;
2. template-OOD evaluation;
3. locked-half confirmation.

No post-test tuning. Report failed runs, retries, exclusions, confidence
intervals, cost, and reproducibility metadata.

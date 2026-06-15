# Evaluation Protocol

## Canonical Targets

Use the reduced schema for generative extraction comparisons. Preserve the full
ground truth and field annotations for alias resolution, evidence checks, and
analysis.

Validate output in two stages:

1. JSON parse validity;
2. canonical schema validity after explicit normalization.

Never silently discard extra keys or repair invalid values without recording
the action.

## Field-Specific Matching

| Field type | Matching rule |
|---|---|
| email | lowercase and exact normalized match |
| phone | digit/extension normalized exact match |
| URL | normalized scheme, host, path, and trailing-slash match |
| date | normalized date or date-range match with explicit partial-date policy |
| boolean/numeric | typed exact match after allowed normalization |
| skills and other unordered lists | item matching plus list F1 and Jaccard |
| company, title, university, degree | normalized fuzzy match and ANLS |
| summary and descriptions | token F1 and ANLS |
| nested repeated objects | maximum-weight Hungarian matching |

The evaluator must report per-field precision, recall, F1, micro/macro
aggregates, and missing/extra keys. A single generic fuzzy threshold is not
acceptable.

## Nested Matching

For education, experience, projects, and other repeated objects:

1. compute a pairwise similarity matrix from configured component fields;
2. use Hungarian matching to maximize total similarity;
3. count unmatched targets as false negatives and unmatched predictions as
   false positives;
4. retain component-level scores for error analysis.

## Evidence Support

Report **unsupported-evidence rate**, not a naive hallucination score.

Check each predicted atomic value against appropriate observable evidence:

- parser input supplied to the mapper;
- canonical clean text for analysis;
- OCR text and OCR word boxes in realistic OCR experiments;
- source word/field annotations only for explicitly labelled oracle analysis.

Track support separately for exact values, fuzzy text values, and generated
summaries. LayoutLMv3 span-derived outputs should retain evidence boxes.

## Parser And OCR Metrics

- character error rate (CER);
- word error rate (WER);
- reading-order score;
- section and table preservation;
- conversion failure rate;
- parser latency and peak memory;
- downstream mapper field score.

Parser conclusions must include downstream extraction quality, not CER/WER
alone.

## Architecture And System Metrics

Record at sample and aggregate level:

- encoder, decoder, postprocess, and total latency;
- peak CPU RAM and GPU VRAM;
- visual tokens before and after compression;
- keep ratio and selector type;
- JSON and schema validity;
- evidence-support rate;
- model/router confidence and repair flags;
- failures, retries, and cost.

## ATS And Screening Metrics

Separate system compatibility from candidate relevance.

Compatibility:

- resume ingestion success and failure category;
- searchable-text coverage and missing-field rate;
- query execution success and latency.

Screening and retrieval:

- precision@k, recall@k, nDCG@k, and mean reciprocal rank;
- false-rejection rate for candidates satisfying explicit eligibility rules;
- ranking overlap and rank correlation between raw-text and structured lanes;
- metrics split by tier, template, domain, and OCR-realistic/source-oracle
  status.

The current dataset has CV targets but no real job descriptions or recruiter
relevance decisions. Domain/skill-derived labels are weak supervision and must
be reported as such. Claims about hiring quality require a separately versioned
human-labeled CV-job relevance set with documented instructions and agreement.

## Leakage Prevention

- Selection and tuning use train and validation only.
- Final ID/OOD metrics are produced only after freezing configs.
- Held-out OOD templates never appear in train, validation, or ID test.
- Source-oracle metadata and boxes cannot enter practical experiments.
- Locked confirmation data is inaccessible to normal trial loaders.
- Job profiles, query rules, and relevance-labeling policy are frozen before
  final screening tests.
- Identity/contact fields are excluded from practical screening and ranking.

## Reporting

Final tables must separate:

- ID vs template-OOD vs locked confirmation;
- source-oracle vs OCR-realistic;
- local/open models vs cloud API upper bounds;
- practical vs oracle routing or pruning.
- legacy ATS/raw-text filtering vs structured filtering;
- weak-label screening results vs human-labeled screening results.

Every table must link to a run/config ID with dataset manifest hash, code
revision, model revision, package lock, hardware, seed, and raw-result location.

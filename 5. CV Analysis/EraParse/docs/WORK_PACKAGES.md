# Work Packages

Each package is agent-sized and must satisfy its acceptance criteria before a
dependent package starts.

## WP0: Repository Foundation

**Deliverables:** project package layout, configuration convention, lint/test
commands, and environment lockfiles.

**Acceptance:** clean install in each compatibility lane; no dataset or secret
files tracked.

## WP1: Dataset Audit And Manifests

**Depends on:** WP0

**Deliverables:** read-only artifact scanner, canonical manifest, deterministic
working/locked selection, working split files, and audit report.

**Acceptance:**

- exactly 4,950 completed samples;
- working and locked corpora contain exactly 2,475 each;
- tier working counts are 625/625/500/375/350;
- split counts are 1,445/310/310/410;
- no overlap and no held-out template in train/validation/ID test;
- orphan `cv_04951.pdf` is excluded.

## WP2: Schema, Evaluator, And Run Store

**Depends on:** WP1

**Deliverables:** canonical reduced schema, normalizers, field-aware matching,
Hungarian nested matching, evidence-support checks, timing/resource metrics,
and DuckDB run schema.

**Acceptance:** unit tests for every matching rule and a deterministic evaluator
golden test.

## WP3: Parser Representation Study

**Depends on:** WP1, WP2

**Deliverables:** adapters for precomputed PyMuPDF/pdfminer, PyMuPDF4LLM
Markdown/JSON, and Docling default/OCR/table/JSON; serialized outputs and
metrics.

**Acceptance:** all methods pass `debug_50`; parser outputs are cached by
content/config hash; best two practical representations selected on validation.

## WP4: Text Mapper Ladder

**Depends on:** WP3

**Deliverables:** NuExtract input-format ablation; mapper comparison on the best
two inputs; section-decomposition ablation; response validation and cache.

**Acceptance:** every output records prompt, raw response, parsed response,
model revision, latency, errors, and repair flags.

## WP4A: ATS Compatibility And Screening Baselines

**Depends on:** WP2, WP3; structured-output comparison also depends on WP4

**Deliverables:** deterministic Boolean and BM25 filters; pinned OpenCATS
integration; optional OpenResume parser/readability lane; versioned job-profile
suite; weak and human relevance judgments; raw-text versus structured-output
screening comparison.

**Acceptance:** ingest/search failures are retained; identity/contact fields are
excluded from ranking; OpenCATS configuration and revision are pinned; retrieval
metrics and false-rejection rates are reported by tier/template; weak-label
results are never described as hiring quality.

## WP5: Direct Visual Baselines

**Depends on:** WP1, WP2

**Deliverables:** fine-tuned Donut baseline and selected modern VLM upper-bound
results.

**Acceptance:** dynamic token/grid handling, valid schema decoding, reproducible
checkpoint metadata, and separate timing components.

## WP6: SG-VTC

**Depends on:** WP5

**Deliverables:** compatibility spike; full, random, norm, oracle,
global-schema-prior, and predicted-class-prior variants at 50% keep ratio.

**Acceptance:** shortened encoder outputs work with generation and attention
masks; visual token counts and coordinate transforms are tested; additional
ratios are blocked until the 50% result justifies them.

WP6 is now conditional on WP5A producing a schema-valid grounded full-token
extractor.

## WP5A: Grounded Schema Extraction

**Depends on:** WP1, WP2, WP5

**Deliverables:** canonical evidence graph; deterministic schema assembler;
free-generation, constrained-generation, decomposed-generation,
token-classification, and SG-ESE field-query comparisons; repeated-record
grouping evaluation.

**Acceptance:** every non-oracle extracted value retains evidence provenance;
assembly guarantees schema validity; perception and extraction errors are
reported separately; SG-ESE novelty is not claimed before formal comparison
with grounded document extraction and schema-conditioned IE research.

## WP7: LayoutLMv3

**Depends on:** WP1, WP2

**Deliverables:** source-oracle token classifier, deterministic schema
assembler, and conditional OCR-realistic lane.

**Acceptance:** source-oracle and OCR-realistic results never share a table
without explicit labels; documents over 512 tokens are chunked and reassembled.

## WP8: Hybrid Router

**Depends on:** WP4, WP5, WP7

**Deliverables:** oracle upper-bound router and practical observable-signal
router.

**Acceptance:** practical features exclude all ground-truth correctness and
metadata; routing decisions and costs are logged.

## WP9: Final Evaluation And Reporting

**Depends on:** selected methods from WP3-WP8 and WP4A

**Deliverables:** frozen configs, ID test, template-OOD test, locked-half
confirmation, error analysis, and final tables.

**Acceptance:** one-time final evaluation after selection; full provenance and
confidence intervals; no unreported retries or post-test tuning.

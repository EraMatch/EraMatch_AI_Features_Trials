# Re-Architected Research Flow

## Core Correction

EraParse must not treat every visual model as a one-shot JSON generator.
Document extraction contains three distinct problems:

1. **perception:** recover text, layout, and visual evidence;
2. **semantic extraction:** assign evidence to CV fields and repeated records;
3. **structure assembly:** produce a valid reduced-schema object.

The failed Donut trial combined all three problems. Its zero schema validity
does not prove that its visual encoder failed to read CVs. It proves that the
complete end-to-end generation contract failed.

## New Main Architecture: Evidence-First Schema Extraction

```text
PDF/pages
  -> visual reader
  -> evidence units: text/span, page, box, confidence, reading order
  -> schema-conditioned extractor
  -> field candidates + evidence pointers + record links
  -> deterministic constrained assembler
  -> valid reduced CV JSON + provenance
```

The canonical intermediate representation is an evidence graph:

- nodes: text spans or visual regions with page/box/confidence;
- node labels: field candidates such as email, skill, company, title, date;
- edges: repeated-record membership and semantic relationships;
- document attributes: parser failures, text density, page count, and visual
  complexity.

The final assembler, not a free-running decoder, owns required keys,
nullability, arrays, deduplication, ordering, and JSON validity.

## Model Roles

### Visual Readers

These models recover evidence but do not need to emit the final CV schema:

- PyMuPDF/Tesseract fallback;
- PaddleOCR-VL 1.6 page parser;
- Granite-Docling;
- Donut encoder or page-level Donut tasks;
- LayoutLMv3 OCR/source word-box inputs.

### Schema Extractors

These map evidence to fields:

- Qwen3/NuExtract text mapper;
- LayoutLMv3 token classifier;
- schema-conditioned field-query classifier;
- relation/grouping head for work, education, project, and certification
  records;
- decomposed visual VQA prompts, one schema section at a time.

### One-Shot Structured VLMs

NuExtract3 and full-schema Donut remain necessary comparison lanes. They test
whether a specialized modern VLM can solve perception, extraction, and
assembly jointly. They are upper bounds/baselines, not the assumed final
architecture.

## Proposed Architecture Contribution: SG-ESE

Working name: **Schema-Guided Evidence Selection and Extraction (SG-ESE)**.
Do not claim novelty until the formal literature comparison is complete.

SG-ESE uses schema field queries to select and classify grounded evidence
before deterministic assembly:

```text
evidence tokens/regions
  + learned or textual schema queries
  -> cross-attention evidence selector
  -> field candidate scores and evidence pointers
  -> relation/grouping head for repeated records
  -> deterministic assembler
```

This architecture addresses the observed failure directly:

- JSON complexity is removed from the model's main learning burden;
- every predicted value can be checked against evidence;
- repeated records become grouping/linking rather than bracket generation;
- visual readers can be compared without changing the schema extractor;
- evidence selection can later support token-efficiency experiments.

SG-VTC becomes a possible efficiency extension of SG-ESE: prune or retain
visual/evidence tokens according to schema-query relevance after a reliable
extractor exists.

## Trial Matrix

### Axis A: Perception

Compare the same extractor using:

1. PyMuPDF text;
2. PyMuPDF/Tesseract fallback;
3. PaddleOCR-VL evidence;
4. source-oracle text/boxes;
5. direct visual features.

### Axis B: Extraction And Assembly

Compare on the same evidence:

1. one-shot free JSON generation;
2. one-shot grammar-constrained generation;
3. section-decomposed generation plus deterministic merge;
4. token/span classification plus deterministic assembly;
5. SG-ESE field queries plus grouping head and deterministic assembly.

### Axis C: Model Family

Run representative, controlled models:

- small text mapper: Qwen3-0.6B;
- specialized structured VLM: NuExtract3;
- OCR/document parser: PaddleOCR-VL 1.6;
- OCR-free encoder-decoder: Donut-base;
- layout-aware classifier: LayoutLMv3.

Avoid a large uncontrolled model leaderboard. Each model must answer a distinct
architectural question.

## Donut Revision

Donut should be tested through progressively harder contracts:

1. tiny-set full-schema overfit;
2. contact-only and flat-core generation;
3. section-specific task prompts;
4. grammar-constrained section generation;
5. Donut encoder features feeding field-query/span heads;
6. page-wise features and cross-page record grouping.

If Donut reads evidence but cannot generate structure, its encoder may still be
useful in SG-ESE. If it cannot overfit simple fields, stop investing in it.

## Decision Logic

- If PaddleOCR-VL -> Qwen3 beats direct VLMs, perception and mapping should
  remain modular.
- If NuExtract3 wins one-shot with strong evidence support, specialized
  structured pretraining matters.
- If constrained/decomposed generation fixes Donut, output structure was the
  primary bottleneck.
- If classification/query extraction beats generation, the thesis contribution
  should center on grounded schema extraction rather than JSON decoding.
- If SG-ESE preserves quality while selecting fewer evidence/visual tokens,
  SG-VTC becomes a justified efficiency extension.

## Scientific Basis

- Unified Information Extraction motivates schema-conditioned structured
  extraction languages rather than ordinary free-text generation.
- LMDX motivates grounded, localized document extraction for singular,
  repeated, and hierarchical entities.
- Grammar-constrained decoding research shows that valid structure cannot be
  assumed from ordinary encoder-decoder fine-tuning.
- Layout-aware instruction-tuning work shows that document layout and
  task-specific structural cues can be explicitly integrated rather than left
  to generic generation.

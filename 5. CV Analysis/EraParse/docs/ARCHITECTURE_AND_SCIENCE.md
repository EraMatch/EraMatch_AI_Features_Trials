# Architecture And Scientific Guide

## Research Framing

EraParse tests where information is lost or made expensive in a CV parsing
pipeline. It separates representation quality, extraction-model capability,
visual understanding, compression, and routing. Architecture claims must be
supported by controlled ablations and relevant prior work.

The main architecture now separates perception, grounded semantic extraction,
and deterministic/constrained structure assembly. See
`REARCHITECTED_RESEARCH_FLOW.md`.

## Evidence-First Extraction

One-shot JSON generation combines reading, semantic assignment, repeated-record
grouping, and syntax generation. EraParse therefore treats it as one baseline,
not the default architecture.

The proposed SG-ESE prototype uses schema-conditioned queries to select grounded
evidence, classify fields, and link repeated records. A deterministic assembler
owns schema validity. This makes output complexity independently testable and
lets visual readers be compared under the same extraction contract.

The next accuracy-oriented architecture is SG-GRSE. It predicts repeated
records as grounded unordered sets using type-specific record queries,
field-conditioned span pointers, document-level candidate evidence, and
Hungarian matching. This directly targets the current weakness of post-hoc
grouping. See `SECOND_CONTRIBUTION_PLAN.md`.

EFSFR is a separate methodology contribution that calibrates and selectively
fuses grounded candidates from multiple extractors using only observable
confidence, agreement, evidence, and structural signals.

## Parser Representations

Plain PDF text is fast but can lose columns, tables, and reading order.
Markdown preserves useful hierarchy for language models. Structured parser
exports retain blocks, boxes, tables, and relationships but require a stable
serializer.

Mandatory parser APIs:

```python
import pymupdf4llm

markdown = pymupdf4llm.to_markdown(pdf_path)
layout_json = pymupdf4llm.to_json(pdf_path)
```

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

options = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
    }
)
document = converter.convert(pdf_path).document
markdown = document.export_to_markdown()
structured = document.export_to_dict()
```

Measure both intrinsic text/layout metrics and downstream field extraction.

## Mapper Architectures

Text mappers receive serialized document evidence and generate the reduced JSON
schema. NuExtract Tiny tests extraction-specialized training; Qwen3 tests a
small general model; Phi tests whether a larger mapper produces a meaningful
gain; Gemma Cloud is an upper bound/repair route.

Section decomposition reduces context and schema complexity but can lose
cross-section relationships. Its merge step must be deterministic and evaluated
against one-shot extraction.

## Donut

[Donut](https://arxiv.org/abs/2111.15664) is an OCR-free document understanding
architecture with a visual encoder and autoregressive text decoder. EraParse
must fine-tune the base model for the CV schema.

Important implications:

- visual encoder cost happens before SG-VTC;
- decoder cross-attention cost depends on encoder token count;
- image resize/padding transforms define the mapping between source boxes and
  visual-token coordinates;
- generated JSON requires validity and evidence checks.

Trial 3 showed that low teacher-forced validation loss does not establish
successful structured generation. The task prompt must start decoding but must
not also appear in the label sequence. After correcting that contract, native
structural targets and repetition controls still failed to produce
schema-valid CV output. SG-VTC is therefore blocked until a full-token visual
baseline passes the schema-generation gate; pruning a failing decoder would
not answer the compression hypothesis.

## SG-VTC Hypothesis

SG-VTC is a project hypothesis for schema-guided visual-token compression after
the Donut encoder and before decoder cross-attention.

Conceptual flow:

```text
page image
  -> Donut visual encoder
  -> visual tokens + inferred spatial map
  -> selector/compressor
  -> shortened encoder hidden state + matching attention mask
  -> Donut decoder
  -> reduced-schema JSON
```

Selector variants:

- random: negative control;
- norm: content-agnostic strength baseline;
- oracle field boxes: upper bound;
- global schema prior: deployable fixed prior;
- predicted-class prior: practical conditional prior;
- true tier/template prior: oracle-only.

Start at a 50% keep ratio. Preserve token order/index mapping unless a tested
merging method explicitly changes it. Build coordinates from actual encoder
shape and processor transforms. T4 source boxes are misaligned with scanned
input and cannot support a realistic practical selector.

SG-VTC can reduce decoder attention and memory, but cannot reduce visual encoder
cost because it runs after the encoder. Always report encoder and decoder time
separately.

## Relevant Compression Work

- [DynamicViT](https://arxiv.org/abs/2106.02034): dynamic token sparsification
  motivates learned content-dependent pruning.
- [TokenLearner](https://arxiv.org/abs/2106.11297): learns a compact set of
  adaptive visual tokens.
- [Token Merging](https://arxiv.org/abs/2210.09461): merges similar tokens
  rather than simply deleting them.
- [VisFocus](https://arxiv.org/abs/2407.12594): focuses document VLM processing
  on relevant visual information.
- [Token-level Correlation-guided Compression](https://arxiv.org/abs/2407.14439):
  studies correlation-guided visual-token compression.
- [Index-Preserving Lightweight Token Pruning](https://arxiv.org/abs/2509.06415):
  directly relevant document token pruning with spatial coherence.
- [DocPrune](https://arxiv.org/abs/2604.22281): training-free progressive
  document token pruning.
- [FastOCR](https://arxiv.org/abs/2605.17447): dynamic visual fixation and KV
  cache pruning for document parsing.
- [RTPrune](https://arxiv.org/abs/2605.00392): high-norm prioritization and
  merging for OCR inference.

Do not claim SG-VTC novelty before comparing its placement, selector signals,
index preservation, training requirements, and evaluation against this work.

## LayoutLMv3

[LayoutLMv3](https://arxiv.org/abs/2204.08387) jointly models text, layout, and
image information. In EraParse it is a non-generative BIO token classifier.

The source-oracle lane uses existing digital-source words and boxes. The
OCR-realistic lane requires OCR word boxes and label alignment. A deterministic
assembler maps predicted spans to the reduced schema and retains evidence
coordinates.

## Hybrid Routing

The oracle router estimates theoretical complementarity and may use
ground-truth correctness only for analysis. The practical router is a deployable
system and may use only observable signals:

- parse/OCR confidence and failures;
- text density and layout complexity;
- JSON/schema validity;
- evidence support;
- model confidence/logprobs;
- repair flags, latency, and document characteristics.

The router must be evaluated for accuracy, cost, calibration, and routing error.

## Evaluation Science

[ANLS and DocVQA](https://arxiv.org/abs/2007.00398) motivate edit-distance-aware
evaluation for document answers. EraParse combines ANLS with exact normalized
matches, list/set scores, token F1, and Hungarian matching because CV fields
have different semantic structures.

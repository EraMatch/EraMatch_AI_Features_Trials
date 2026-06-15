# Baseline Matrix And Thesis Flow

## Thesis Question

EraParse asks where CV information is lost, which recovery mechanism restores
it, and whether visual-token compression can retain that recovery at lower
decoder cost.

The thesis is not a leaderboard of unrelated models. Each baseline isolates a
specific cause or remedy:

1. representation loss;
2. extraction-model capability;
3. OCR recovery;
4. direct visual understanding;
5. token compression;
6. practical routing.

## Controlled Baselines

| Family | Lane | Role | Deployable |
|---|---|---|---|
| representation lower bound | raw PyMuPDF/pdfminer text | measures embedded-text availability and reading-order loss | yes |
| parser alternatives | PyMuPDF4LLM and Docling | tests richer serialization without model changes | yes |
| text mapper baseline | NuExtract Tiny | extraction-specialized small mapper | yes |
| text mapper baseline | Qwen3-0.6B/PyMuPDF | small general mapper; frozen initial baseline | yes |
| OCR recovery baseline | PyMuPDF with Tesseract fallback -> Qwen3 | isolates whether OCR alone repairs scanned T4 failures | yes |
| legacy ATS baseline | deterministic Boolean/BM25 and pinned OpenCATS | measures screening consequences independently of extraction F1 | yes |
| direct visual baseline | fine-tuned Donut base | tests OCR-free end-to-end document understanding | yes |
| modern visual upper bound | selected current document VLM | bounds the benefit available from larger modern architectures | conditional |
| layout-aware baseline | LayoutLMv3 source-oracle and OCR-realistic | tests explicit text/layout modeling | conditional |
| oracle upper bounds | canonical structured, oracle text, source boxes | diagnoses recoverable headroom | no |

Oracle lanes must never be described as deployable results. OpenCATS remains
pending until its actual pinned system integration runs.

Trial 3 selected the Qwen3 OCR-recovery lane on validation. The corrected
Donut-base lane completed full training but failed schema-valid generation, so
it is retained as a diagnostic full-token visual implementation rather than a
quality baseline. The next required comparison is a modern document-VLM upper
bound; SG-VTC remains blocked until a full-token visual model passes the
schema-generation gate.

The first full practical PyMuPDF4LLM/LayoutLMv3 validation trial reached
`0.728579` macro with 100% schema validity and 0% unsupported evidence after
one epoch. The equal-budget SG-ESE query-only architecture reached `0.689791`.
After two equal-budget epochs, standard LayoutLMv3 reached `0.739158` and
SG-ESE reached `0.738284`. Both beat the earlier Qwen3 practical macro
`0.6709`; standard LayoutLMv3 remains the strongest practical validation
baseline by only `0.000875`, making SG-ESE competitive enough for controlled
sparse-token efficiency trials.

## Core Comparisons

The central causal comparisons are:

- raw text vs OCR fallback: value of recovering text from scans;
- OCR fallback vs Donut: OCR-mediated recovery vs OCR-free vision;
- Donut full visual tokens vs SG-VTC variants: compression effect under the
  same trained architecture;
- best single practical lane vs practical router: value of observable routing;
- extraction metrics vs ATS screening metrics: whether improved JSON actually
  improves downstream candidate recovery.

## Dataset Discipline

- engineering gates: `debug_50`, optionally `debug_250`, training samples only;
- model fitting: all 1,445 training CVs;
- method selection: all 310 validation CVs;
- frozen final working-corpus tests: 310 ID and 410 template OOD;
- locked confirmation: 2,475 CVs, once, only after final method selection.

Small gates validate code and prevent expensive invalid runs. They are never
reported as final evidence.

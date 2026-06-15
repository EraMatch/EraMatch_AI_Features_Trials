# Trial 2 Initial Results

## Scope

These are selection-only `debug_50` results from training samples. They are not
validation, ID-test, OOD-test, or locked-confirmation results.

All NuExtract runs use:

- model: `numind/NuExtract-1.5-tiny`
- revision: `63e2e80c804d9c97f3f19a4aa25613e7beca83c9`
- Modal GPU: `A10`
- deterministic decoding
- `max_new_tokens=1600`

## Input Ablation

| Representation | Macro score | JSON valid | Schema valid | Mean model latency |
|---|---:|---:|---:|---:|
| oracle text | 0.7634 | 0.98 | 0.92 | 10.87 s |
| precomputed pdfminer text | 0.4857 | 0.88 | 0.88 | 10.07 s |
| precomputed PyMuPDF text | 0.4738 | 0.88 | 0.84 | 10.37 s |
| Docling Markdown | 0.4110 | 0.78 | 0.78 | 10.84 s |
| PyMuPDF4LLM Markdown | 0.3980 | 0.82 | 0.80 | 11.19 s |

The oracle gap is large enough that representation quality remains the dominant
bottleneck. Fresh Markdown did not improve the mapper result on this subset.
Promote precomputed pdfminer and PyMuPDF text to Trial 2B.

Raw parser JSON was generated but not promoted to mapper inference in this
stage. Mean representation size was about 116,093 characters for PyMuPDF4LLM
JSON and 25,940 for Docling JSON, compared with 1,554 and 1,912 for their
Markdown forms. Feeding the raw exports would mainly measure context truncation.
A later serialization ablation must first define a compact, information-matched
JSON projection.

## Modal Runs

| Purpose | Modal run |
|---|---|
| document representations, corrected run | `ap-aJn4EyOeUaZ4gnvecG7HQH` |
| NuExtract PyMuPDF4LLM Markdown | `ap-7MM3PH4TlOAIDHeovZvsic` |
| NuExtract Docling Markdown | `ap-yPDxZvl0dplcTFoTOIVGTV` |
| Qwen3 pdfminer text | `ap-YU5AQwzs9CMlctrzSYyRuY` |
| Qwen3 PyMuPDF text | `ap-1QzAVFAylU6R3b26jsi6JD` |
| Qwen3 PyMuPDF text, debug-250 | `ap-NiikMKoDYAwXcOcbAqEs2Y` |

## Mapper Comparison

| Model and representation | Macro score | JSON valid | Schema valid | Unsupported evidence | Mean latency |
|---|---:|---:|---:|---:|---:|
| Qwen3, precomputed PyMuPDF | 0.5593 | 0.98 | 0.98 | 0.0271 | 11.49 s |
| Qwen3, precomputed pdfminer | 0.5345 | 1.00 | 1.00 | 0.0310 | 14.71 s |
| NuExtract, precomputed pdfminer | 0.4857 | 0.88 | 0.88 | 0.0022 | 10.07 s |
| NuExtract, precomputed PyMuPDF | 0.4738 | 0.88 | 0.84 | 0.0064 | 10.37 s |

Qwen3/PyMuPDF is the current best practical lane. It had one malformed JSON
output and no outputs that reached the 1,600-token cap. Its higher unsupported
evidence rate requires explicit review before final selection.

## Next Gate

Qwen3/PyMuPDF passed `debug_250` using:

- model: `Qwen/Qwen3-0.6B`
- revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- `enable_thinking=False`

| Documents | Macro score | JSON valid | Schema valid | Unsupported evidence | Mean latency |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.5563 | 0.992 | 0.992 | 0.0312 | 10.90 s |

The result closely reproduces the debug-50 score. Two outputs were malformed
and one reached the 1,600-token cap. The lane is promoted to the 310-sample
validation split. Validation remains selection-only; do not use ID test,
template-OOD test, or locked confirmation yet.

## Full Validation

Qwen3/PyMuPDF completed all 310 validation CVs:

| Documents | Macro score | JSON valid | Schema valid | Unsupported evidence | Mean latency |
|---:|---:|---:|---:|---:|---:|
| 310 | 0.5862 | 1.000 | 1.000 | 0.0283 | 12.65 s |

The exact selected configuration is frozen in
`configs/models/qwen3_pymupdf_frozen_v1.json`. It may now run once on all 310
ID-test and all 410 template-OOD CVs. Those runs are now complete; see
`QWEN3_FROZEN_RESULTS.md`. Locked confirmation remains untouched.

The first validation attempt (`ap-ScB9SzGf8VYV8ugVqHVqLB`) exceeded a
single-call 3,600-second Modal timeout after 230 CVs and produced no response
file. The corrected resumable/chunked run (`ap-VU6KO5ThxSCa0nbLxc6ta0`)
completed all 310 and persisted each completed chunk.

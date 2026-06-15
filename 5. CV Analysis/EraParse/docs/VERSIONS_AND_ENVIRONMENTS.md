# Versions And Environments

Checked on 2026-06-09. These are initial compatibility lanes, not final
lockfiles. Each implementation lane must produce a tested lockfile and smoke
test before reportable trials.

## Why Separate Lanes

Donut, LayoutLMv3, and small text mappers are stable in a Transformers 4.x
environment. NuExtract3 and PaddleOCR-VL-1.6 depend on newer/current VLM support
and should not destabilize the core lane. PaddleOCR also has its own framework
requirements.

## Lane Matrix

| Lane | Python | Initial packages | Purpose |
|---|---|---|---|
| `cpu-parsers` | 3.11 | `docling==2.82.0`, `pymupdf4llm==1.27.2.2`, `duckdb==1.5.0`, `pydantic==2.12.5` | audits, parsers, evaluation, reports |
| `core-transformers4` | 3.11 | `transformers==4.57.3`; tested Torch/bitsandbytes pins added after smoke tests | NuExtract Tiny, Qwen3, Phi, Donut, LayoutLMv3 |
| `modern-vlm` | 3.11 | current tested Transformers 5.x plus model-card dependencies | NuExtract3 and Transformers-based modern VLM checks |
| `paddle-vlm` | model-supported Linux/Python | PaddlePaddle 3.2.1+, `paddleocr[doc-parser]>=3.6.0` | PaddleOCR-VL-1.6 and optional PP-DocBee |
| `modal-authoring` | 3.11 | `modal==1.3.5` target | remote orchestration |

The local machine currently has Modal client `1.2.6`. Upgrade and verify the
client before implementing or executing Modal jobs.

## Current Package Observations

Latest releases observed during planning included:

- Transformers 5.4.0;
- Torch 2.11.0;
- bitsandbytes 0.49.2;
- Modal 1.3.5;
- Docling 2.82.0;
- PyMuPDF4LLM 1.27.2.2;
- DuckDB 1.5.0;
- Pydantic 2.12.5.

Do not blindly combine the latest version of every package. Pin versions only
after lane-specific imports, minimal inference, serialization, and GPU smoke
tests pass.

## Required Smoke Tests

### CPU Parsers

- import all packages;
- convert one digital and one T4 PDF;
- export Markdown and structured JSON/dict;
- verify deterministic serialization.

### Core Transformers 4

- load exact pinned revisions;
- run minimal NuExtract and Qwen generation;
- run Donut processor/encoder/decoder;
- run LayoutLMv3 processor with `apply_ocr=False`;
- validate dtype and GPU memory logging.

### Modern And Paddle VLMs

- follow exact current model-card install instructions;
- verify page-level vs element-level behavior;
- capture generated output formats and license metadata.

### Modal

- run `modal --version` and relevant `modal <command> --help`;
- import all image dependencies remotely using a CPU-only smoke job;
- verify named Volume mounts and model cache paths;
- do not launch paid GPU work during environment validation unless requested.

## Reproducibility Metadata

Every reportable run records:

- Python and OS/container information;
- exact package lock and CUDA/runtime versions;
- model ID and revision;
- code Git revision;
- data manifest hash;
- image/build identity for Modal;
- hardware/GPU name;
- seed and full resolved config.

# Frozen Qwen3 Final Results

## Scope

The selected Qwen3/PyMuPDF configuration was frozen after full validation and
then evaluated once on the real protected working-corpus tests:

- validation: all 310 CVs;
- ID test: all 310 CVs;
- template-OOD test: all 410 CVs;
- locked confirmation: untouched.

Frozen configuration: `configs/models/qwen3_pymupdf_frozen_v1.json`

Model: `Qwen/Qwen3-0.6B`

Revision: `c1899de289a04d12100db370d81485cdf75e47ca`

## Extraction Results

| Split | Documents | Macro | JSON valid | Schema valid | Unsupported evidence | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| validation | 310 | 0.5862 | 100.00% | 100.00% | 2.83% | 12.65 s |
| ID test | 310 | 0.4874 | 82.90% | 82.90% | 2.71% | 15.27 s |
| template OOD | 410 | 0.6686 | 100.00% | 100.00% | 3.86% | 13.14 s |

Tracked summaries:

- `reports/models/qwen3-pymupdf-validation-summary.json`
- `reports/models/qwen3-pymupdf-id-test-summary.json`
- `reports/models/qwen3-pymupdf-ood-test-summary.json`

Run IDs:

- validation: `qwen3-0.6b-pymupdf_text-f9d63bb9dd`
- ID: `qwen3-0.6b-pymupdf_text-63a45bbd20`
- OOD: `qwen3-0.6b-pymupdf_text-7f76756aeb`

## Error Analysis

The ID pool contains 53 invalid or empty predictions. Fifty outputs reached
the 1,600-token generation cap. ID performance by tier:

| Tier | Documents | Valid outputs | Macro |
|---|---:|---:|---:|
| T1 | 94 | 50 | 0.4469 |
| T2 | 95 | 86 | 0.7001 |
| T3 | 46 | 46 | 0.6042 |
| T4 | 56 | 56 | 0.0342 |
| T5 | 19 | 19 | 0.6769 |

T4 is the dominant representation failure: the selected PyMuPDF input is empty
for all 56 T4 ID documents. The strong OOD result does not test scanned T4
documents because template OOD contains only T3 and T5.

## Downstream ATS Result

Applying the frozen ATS profiles to Qwen3 structured predictions increases ID
Boolean false rejection from 18.76% for raw PyMuPDF text to 35.69%. It leaves
T4 at 100% false rejection and introduces substantial T1 losses. See
`ATS_BASELINE_RESULTS.md`.

## Decision

Do not promote this lane to locked confirmation. Preserve it as the initial
small text-mapper baseline.

The next model work is Trial 3:

1. establish an OCR/direct-vision path that can consume T4;
2. fine-tune Donut on the 1,445-sample training split;
3. select with the 310-sample validation split;
4. compare a modern document VLM upper bound before SG-VTC.

The protected ID/OOD findings are reportable error analysis, not permission to
retune and rerun this frozen lane.

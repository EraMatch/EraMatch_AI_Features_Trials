# ATS Baseline Results

## Status

The deterministic legacy ATS baseline was run on the real protected test
sizes:

- ID test: all 310 CVs.
- Template-OOD test: all 410 CVs.
- Locked confirmation: untouched.

Run ID: `ats-baseline-e9b1d2faf5db`

Frozen profiles:
`configs/ats/domain_job_profiles_v1.json`

Tracked compact summary:
`reports/baselines/ats-baseline-e9b1d2faf5db-summary.json`

Complete local rankings:
`artifacts/ats_baselines/ats-baseline-e9b1d2faf5db/query_results.jsonl`

The complete rankings and DuckDB run record are intentionally ignored by Git.
The tracked profiles and compact summary preserve the comparison contract and
headline results.

## Benchmark Interpretation

The ten job profiles were learned from canonical skills in the 1,445 training
CVs only. A candidate is weakly relevant when its existing primary-domain label
matches the job profile. These results measure recovery under the synthetic
benchmark taxonomy, not real hiring quality.

The canonical structured and oracle-text lanes are upper bounds. They are not
deployable ATS results.

## Main Results

Boolean screening:

| Split | Lane | Candidates | False rejection | nDCG@25 | Recall@25 |
|---|---|---:|---:|---:|---:|
| ID | PyMuPDF text | 310 | 18.76% | 0.9340 | 0.6977 |
| ID | pdfminer text | 310 | 18.76% | 0.9340 | 0.6977 |
| ID | oracle text | 310 | 0.00% | 1.0000 | 0.7976 |
| ID | canonical structured | 310 | 0.00% | 1.0000 | 0.7976 |
| OOD | PyMuPDF text | 410 | 0.23% | 1.0000 | 0.6972 |
| OOD | pdfminer text | 410 | 0.23% | 1.0000 | 0.6972 |
| OOD | oracle text | 410 | 0.00% | 1.0000 | 0.6972 |
| OOD | canonical structured | 410 | 0.00% | 1.0000 | 0.6972 |

BM25 produces the same broad conclusion. On ID, raw parser text reaches
`0.9340` nDCG@25 versus `1.0000` for oracle/structured inputs. On OOD, raw
parser text is effectively tied with the upper bounds under these weak labels.

## T4 Failure

Both raw text parser lanes contain 56 empty documents in the 310-CV ID pool.
All 56 are T4 scanned/degraded CVs.

For Boolean screening:

- PyMuPDF T4 false rejection: 56/56, or 100%.
- pdfminer T4 false rejection: 56/56, or 100%.
- oracle-text T4 false rejection: 0/56.
- canonical-structured T4 false rejection: 0/56.

The low OOD false-rejection rate must not be interpreted as stronger OOD
generalization: the template-OOD pool contains T3 and T5 templates but no T4
scanned/degraded documents.

## OpenCATS

OpenCATS `0.9.7.4` remains the required real legacy-system integration
baseline. It was not assigned deterministic-baseline results. The local Docker
CLI is installed, but the Docker daemon was unavailable during this run, so the
OpenCATS container/import/search experiment remains pending.

## Qwen3 Prediction Comparison

The same frozen profiles and metrics were applied to the frozen Qwen3/PyMuPDF
structured predictions.

Run ID: `ats-prediction-f8e79dea23cc`

Tracked compact summary:
`reports/baselines/ats-prediction-f8e79dea23cc-summary.json`

| Split | Input | False rejection | nDCG@25 | Recall@25 |
|---|---|---:|---:|---:|
| ID | raw PyMuPDF text | 18.76% | 0.9340 | 0.6977 |
| ID | Qwen3 structured prediction | 35.69% | 0.8545 | 0.5951 |
| ID | canonical structured upper bound | 0.00% | 1.0000 | 0.7976 |
| OOD | raw PyMuPDF text | 0.23% | 1.0000 | 0.6972 |
| OOD | Qwen3 structured prediction | 1.33% | 1.0000 | 0.6972 |
| OOD | canonical structured upper bound | 0.00% | 1.0000 | 0.6972 |

Qwen3 structured predictions do not improve downstream screening in the
current lane. ID false rejection rises from 18.76% to 35.69%. There are 109
empty structured-search documents on ID and none on OOD.

The ID failure is concentrated in T1 and T4:

| Tier | False rejections | Rate |
|---|---:|---:|
| T1 | 44/94 | 46.81% |
| T2 | 10/95 | 10.53% |
| T3 | 0/46 | 0.00% |
| T4 | 56/56 | 100.00% |
| T5 | 1/19 | 5.26% |

The text-only model cannot restore T4 because its PyMuPDF input is empty.
The next practical comparison must use an OCR or direct-vision lane. Protected
test findings must not be used to tune the already frozen Qwen3 lane.

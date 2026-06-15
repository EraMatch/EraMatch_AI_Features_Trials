# Dataset Contract

## Source And Immutability

Default source path: `../eramatch_benchmark_v4`

The source dataset is read-only. All generated manifests, splits, derived parser
outputs, caches, and reports belong inside EraParse or approved remote storage.

The source currently occupies about 3.4 GB. `ground_truth/*.json` is the
completion marker and defines the canonical sample universe.

## Observed Corpus

| Tier | Completed | Working | Locked |
|---|---:|---:|---:|
| T1 | 1,250 | 625 | 625 |
| T2 | 1,250 | 625 | 625 |
| T3 | 1,000 | 500 | 500 |
| T4 | 750 | 375 | 375 |
| T5 | 700 | 350 | 350 |
| **Total** | **4,950** | **2,475** | **2,475** |

Tier and template must be read from
`layout_annotations/{cv_id}_layout.json`. Do not infer T4 from a filename or
template prefix: T4 samples can use templates whose names begin with T1, T2, or
T3.

## Observed Artifact Contract

For each completed `cv_XXXXX`, the audit expects:

| Artifact | Pattern | Observed count | Notes |
|---|---|---:|---|
| full ground truth | `ground_truth/{cv_id}.json` | 4,950 | canonical completion marker |
| compact target | `donut_targets/{cv_id}.json` | 4,950 | JSON object, not a JSON string wrapper |
| full schema target | `schema_targets/full/{cv_id}.json` | 4,950 | object with aliases and metadata |
| reduced schema target | `schema_targets/reduced/{cv_id}.json` | 4,950 | compact extraction schema |
| layout | `layout_annotations/{cv_id}_layout.json` | 4,950 | tier, template, sections, fields |
| field annotations | `field_annotations/{cv_id}_fields.json` | 4,950 | array with PDF and normalized boxes |
| section annotations | `section_annotations/{cv_id}_sections.json` | 4,950 | ordered section array |
| word annotations | `word_annotations/{cv_id}_words.json` | 4,950 | word array with boxes and labels |
| BIO labels | `token_labels/{cv_id}_bio.json` | 4,950 | word array with BIO labels |
| clean text | `text_ground_truth/{cv_id}.txt` | 4,950 | oracle text representation |
| precomputed PyMuPDF | `extracted_text/pymupdf/*` | 4,950 | parser baseline |
| precomputed pdfminer | `extracted_text/pdfminer/*` | 4,950 | parser baseline |
| rendered PDF | `pdfs/{cv_id}.pdf` | 4,951 | includes orphan `cv_04951.pdf` |
| source PDF | `source_pdfs/{cv_id}_source.pdf` | 4,951 | digital source, including orphan |
| page images | `page_images/{cv_id}-{page}.png` | 5,448 | multi-page, count exceeds samples |
| T4 OCR baseline | `ocr_baselines/tesseract/*` | 750 | text only; no OCR word boxes |
| T4 OCR ground truth | `ocr_ground_truth/{cv_id}.txt` | 750 | T4-only |

The source `splits/` directory is empty and no root `metadata.json` exists.
EraParse must generate its own manifest and split files.

## Ground-Truth Shape

The full ground-truth object contains fields beyond the reduced extraction
schema, including:

- top-level aliases such as `email`, `full_name`, `location`, `companies`,
  `job_titles`, `universities`, and `degrees`;
- nested `contact_info`;
- work-experience aliases such as `company`/`organization` and
  `job_title`/`title`;
- additional metadata including duration, technologies, employment type, and
  parsing metadata.

The reduced schema is the default generative extraction target. The evaluator
must map aliases deliberately; it must not assume that the pasted historical
schema is the complete artifact contract.

## Deterministic Manifest Algorithm

Use seed `20260609` and stable SHA-256 ordering. Never rely on filesystem order
or a language runtime's randomized hash.

1. Enumerate sorted `ground_truth/cv_*.json`.
2. For each ID, validate required artifacts and read tier/template from its
   layout annotation and domain from ground truth.
3. Exclude any ID without the completion marker, including `cv_04951`.
4. Select these working-corpus OOD quotas by deterministic hash while balancing
   primary domain:

| Held-Out Template | Working OOD Quota |
|---|---:|
| `T3_nested_tables` | 100 |
| `T3_europass` | 100 |
| `T5_infographic` | 70 |
| `T5_magazine` | 70 |
| `T5_dark` | 70 |
| **Total** | **410** |

5. Fill each tier's remaining working quota from non-held-out templates using
   proportional Hamilton allocation over `(template, primary_domain)` strata.
   Resolve fractional and hash ties lexicographically.
6. The unselected complement becomes `locked_confirmation`.
7. Split the 2,065 non-OOD working samples into train 1,445, validation 310,
   and ID test 310 using the same proportional allocation and stable hash.
8. Write manifest rows with source artifact paths, tier, template, domain,
   split, selection seed, and source-file hashes.

## Split Invariants

- `template_ood_test` contains only the five held-out templates.
- The five held-out templates appear in no train, validation, or ID-test row.
- Every completed sample appears exactly once in either a working split or the
  locked confirmation corpus.
- The locked corpus is not loaded by debug, training, validation, or test code.
- Changes to the algorithm or source artifacts must change the manifest hash
  and invalidate dependent run caches.

## T4 Contract

T4 is scanned/degraded data. Existing word boxes and BIO labels are derived
from the digital source PDF, so using them is a **source-oracle** upper bound.

The realistic OCR lane requires generating OCR word boxes from rendered/scanned
input and aligning labels to those boxes. Tesseract text alone is insufficient.
Source-oracle and OCR-realistic results must be trained, evaluated, and reported
separately.

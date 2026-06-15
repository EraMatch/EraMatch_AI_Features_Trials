# Second Contribution Plan

## Decision

The second accuracy-oriented architecture contribution will be
**Schema-Guided Grounded Record Set Extraction (SG-GRSE)**.

The separate methodology/community contribution will be
**Evidence-First Selective Fusion and Repair (EFSFR)**.

These contributions target different questions:

1. SG-GRSE asks whether repeated CV records can be extracted more accurately by
   predicting grounded record sets directly instead of classifying tokens and
   grouping them after inference.
2. EFSFR asks whether calibrated agreement, evidence support, and schema
   constraints can combine multiple extractors into a more accurate and safer
   practical system.

Neither contribution is called novel until the formal literature comparison is
complete.

## Why Another Architecture Is Justified

The fair two-epoch practical validation results are:

| Method | Macro | Schema valid | Unsupported evidence |
|---|---:|---:|---:|
| standard LayoutLMv3 | **0.739158** | 1.000000 | 0.000000 |
| SG-ESE query-only | **0.738284** | 1.000000 | 0.000000 |

SG-ESE is competitive but does not outperform standard LayoutLMv3. The weakest
standard LayoutLMv3 fields are:

| Field | Validation score |
|---|---:|
| skills | 0.462888 |
| GitHub URL | 0.493548 |
| projects | 0.542950 |
| LinkedIn URL | 0.596774 |
| work experience | 0.627738 |

The current model predicts field labels per token and then groups repeated
records using a deterministic reading-order heuristic. The learned all-pairs
grouping head performed worse. This means the next architecture should optimize
record structure directly rather than add another post-hoc grouping heuristic.

The first practical oracle-ceiling check confirmed this direction:

| Mode | Validation macro |
|---|---:|
| standard LayoutLMv3, two epochs | 0.739158 |
| oracle labels + sequence grouping | 0.748347 |
| oracle labels + oracle grouping | 0.755676 |

This shows both candidate quality headroom and a meaningful grouping gap.
Projects and work experience benefit the most from oracle grouping.

The first work-bank preparation pass adds another design constraint:

- `job_title`, `company`, `start_date`, and `end_date` have approximately
  `0.945-0.960` direct span coverage on validation;
- `duration` has only `0.0956` direct coverage and should initially be derived
  from selected start/end evidence instead of treated as a primary span target.

## SG-GRSE Architecture

### Core Flow

```text
PDF
  -> PyMuPDF4LLM evidence graph
  -> page-level LayoutLMv3 encoder
  -> grounded candidate-span bank across all pages
  -> type-specific record-set queries
  -> Hungarian-matched record and field predictions
  -> deterministic grounded JSON
```

### Components

1. **Shared page encoder**
   - use the existing fine-tuned LayoutLMv3 encoder;
   - preserve token boxes, page identity, reading order, and evidence IDs;
   - pool candidate spans from every page into one document-level bank.

2. **Scalar branch**
   - retain the stronger of the standard token-classification and SG-ESE
     query-conditioned scalar heads;
   - scalar fields are not forced through the record-set decoder.

3. **Type-specific record queries**
   - allocate learned slots independently for work experience, education,
     projects, and certifications;
   - each slot predicts record existence and one grounded span per schema
     field;
   - unused slots predict `null`, supporting variable record counts.

4. **Field-conditioned span pointers**
   - compose each record query with a schema-field embedding;
   - predict start/end evidence spans or null;
   - preserve selected evidence IDs for every value.
   - treat `duration` as derived from selected start/end evidence in the first
     SG-GRSE version unless a later direct-evidence experiment justifies a
     separate duration pointer.

5. **Set-based matching**
   - match predicted record slots to truth records with Hungarian matching;
   - matching cost combines record existence, field span accuracy, normalized
     value similarity, and evidence support;
   - record order is not required during training.

6. **Document-level structure**
   - combine page candidate banks before record decoding;
   - prevent multi-page records from being treated as unrelated page outputs;
   - optionally add section/block embeddings and spatial/reading-order edges.

### Initial Objective

```text
loss =
  1.00 * scalar_token_loss
+ 1.00 * matched_record_field_loss
+ 0.50 * record_existence_loss
+ 0.25 * span_boundary_loss
+ 0.25 * evidence_support_loss
+ 0.10 * schema_dependency_loss
```

Weights are hypotheses and require ablation. Do not promote all auxiliary losses
at once without showing their individual effect.

## Accuracy Targets

The minimum success condition is a statistically credible improvement over
standard LayoutLMv3 `0.739158` on all 310 validation CVs. The practical bar to
beat is now the current strongest methodology lane at `0.775538`, with the
best current combined SG-GRSE + EFSFR result at `0.775809`.

| Target | Minimum | Stretch |
|---|---:|---:|
| overall macro | > 0.745 | > 0.755 |
| work experience | > 0.650 | > 0.680 |
| projects | > 0.580 | > 0.620 |
| education | > 0.820 | > 0.840 |
| schema-valid rate | 1.000 | 1.000 |
| unsupported evidence | 0.000 | 0.000 |

The contribution fails as a practical-leading accuracy claim if it does not
beat the strongest competing held-out lane under comparable conditions.
Negative and near-tie results remain reportable.

## Required SG-GRSE Ablations

| Trial | Purpose |
|---|---|
| standard LayoutLMv3 | frozen accuracy baseline |
| SG-ESE query-only | current architecture comparison |
| scalar branch + deterministic grouping | isolates candidate quality |
| record-set decoder without schema-field queries | tests set prediction alone |
| schema-field queries without document-level fusion | tests query conditioning |
| document-level record-set decoder | tests cross-page record modeling |
| full SG-GRSE | proposed architecture |
| oracle candidate spans | measures decoder/grouping ceiling |
| oracle record grouping | measures extraction ceiling |

Run one and two epochs first. Promote longer training only when validation is
still improving under identical budgets.

## EFSFR Methodology Contribution

EFSFR is an evidence-grounded, field-aware selective fusion protocol. It is
separate from SG-GRSE and can combine any extractors.

### Flow

```text
multiple grounded extractors
  -> normalize field candidates
  -> measure agreement, confidence, support, and structural consistency
  -> calibrated field/record selector
  -> optional targeted repair only for uncertain fields
  -> deterministic JSON plus selector trace
```

### Observable Selection Signals

- calibrated candidate confidence;
- agreement between independent extraction heads;
- evidence support and span coverage;
- schema and record consistency;
- parser/OCR confidence;
- field presence probability;
- margin and entropy;
- document layout and text-density features.

Ground-truth tier, template, values, and field scores are prohibited at
inference.

### Methodology Outputs

- field-level risk/coverage curves;
- record-level correction-cost estimates;
- calibrated abstention and repair thresholds;
- accuracy/latency/cost Pareto frontier;
- selector traces that explain why a candidate was accepted, rejected, or sent
  for repair;
- ATS downstream effect before and after fusion.

### EFSFR Targets

- exceed the best single practical extractor by at least `0.005` macro;
- stretch target: exceed it by `0.010`;
- preserve 100% schema validity and 0% unsupported evidence;
- reduce expensive-model usage by at least 70% relative to always using the
  repair model;
- improve nested-record correction cost and ATS false-rejection rate.

### Current EFSFR Status

The first implemented EFSFR slice was deliberately narrow: grounded
`work_experience` repair over frozen practical LayoutLMv3 predictions.

Held-out validation result:

| Method | Macro | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|
| standard LayoutLMv3 | 0.739158 | 0.627738 | 1.000000 | 0.000000 |
| + EFSFR work repair | **0.763803** | **0.923473** | 1.000000 | 0.000000 |

The broader current EFSFR lane now adds anchor-based nested-record rebuilding
for education, projects, and certifications.

| Method | Macro | Education | Projects | Certifications | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard LayoutLMv3 | 0.739158 | 0.806531 | 0.542950 | 0.904499 | 0.627738 | 1.000000 | 0.000000 |
| + full EFSFR nested repair | **0.775538** | **0.837617** | **0.645540** | **0.911643** | **0.923473** | 1.000000 | 0.000000 |

Interpretation:

- the first EFSFR slice already cleared the minimum macro-improvement target;
- the broader nested-repair slice now exceeds the baseline by `+0.036380`
  macro while preserving perfect schema validity and zero unsupported evidence;
- this means EFSFR is already a credible standalone methodology contribution,
  not just a helper for SG-GRSE;
- SG-GRSE must beat the `0.775538` EFSFR lane to become the lead practical
  system.

### Current SG-GRSE Status

The first SG-GRSE slice is a work-experience decoder over grounded candidate
slots with adjacent-slot merging.

Held-out validation result:

| Method | Macro | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|
| standard LayoutLMv3 | 0.739158 | 0.627738 | 1.000000 | 0.000000 |
| + SG-GRSE work decoder | **0.764074** | **0.926728** | 1.000000 | 0.000000 |
| full EFSFR nested repair | 0.775538 | 0.923473 | 1.000000 | 0.000000 |
| full EFSFR + SG-GRSE work | **0.775809** | **0.926728** | 1.000000 | 0.000000 |

Interpretation:

- the first SG-GRSE result is positive on held-out validation;
- it beats full EFSFR only narrowly, by `+0.000271` macro;
- the current bootstrap comparison interval still crosses zero, so the present
  gain should be treated as suggestive rather than decisive;
- therefore the next SG-GRSE work must aim for robustness and a clearer margin
  before any strong novelty or superiority claim.

### Current Selective SG-GRSE Status

The strongest current practical SG-GRSE lane is selective rather than
unconditional: SG-GRSE work records are accepted only when they increase the
count of near-complete work records relative to EFSFR.

Held-out validation result:

| Method | Macro | Work experience | Schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|
| full EFSFR nested repair | 0.775538 | 0.923473 | 1.000000 | 0.000000 |
| selective EFSFR + SG-GRSE work | **0.776183** | **0.931212** | 1.000000 | 0.000000 |

Comparison summary:

- accepted SG-GRSE on `7 / 310` validation CVs;
- mean macro delta: `+0.000645`;
- macro wins / losses / ties: `7 / 0 / 303`;
- macro delta 95% bootstrap CI:
  `[0.000169, 0.001270]`.

Interpretation:

- this is the first SG-GRSE result that is both positive and statistically
  cleaner than the unconditional replacement;
- the current evidence supports SG-GRSE most strongly as a selective
  specialist module rather than a universal decoder swap.

## Community Contribution

The releasable contribution should be broader than a CV model checkpoint:

1. a schema-driven record-set decoder reusable for invoices, receipts, forms,
   and other repeated-record KIE tasks;
2. evidence graph and deterministic assembly contracts;
3. grouped-record evaluation with Hungarian matching and correction-cost
   reporting;
4. template-OOD and evidence-grounding trial protocol;
5. reproducible field-aware selective-fusion implementation;
6. complete positive, negative, failure, cost, and ablation logs.

## Scientific Positioning

- LayoutLMv3 establishes the multimodal token-classification baseline.
- Recent visual-document relation-extraction work supports explicitly modeling
  relations rather than relying only on entity labels.
- KIEval supports evaluating grouped structured information, not only isolated
  entities.
- Work on visual label dependencies supports explicitly learning schema/label
  relationships.
- Graph-based document-structure research supports modeling spatial and logical
  document relations.
- Semantic-block approaches motivate localized section/block representations,
  particularly for unseen templates.

Direct references:

- LayoutLMv3: <https://arxiv.org/abs/2204.08387>
- LayoutLMv3 relation extraction:
  <https://arxiv.org/abs/2404.10848>
- Learning label dependencies for VIE:
  <https://www.ijcai.org/proceedings/2024/0731.pdf>
- KIEval: <https://arxiv.org/abs/2503.05488>
- Graph-based Document Structure Analysis:
  <https://arxiv.org/abs/2502.02501>
- BLOCKIE semantic blocks: <https://arxiv.org/abs/2505.13535>

## Execution Order

1. Build a document-level candidate-span bank from existing predictions.
2. Measure oracle candidate and oracle grouping ceilings before training.
3. Implement Hungarian-matched record-set targets and evaluator tests.
4. Implement the smallest record-set decoder for work experience only.
5. Run tiny overfit, `debug_50`, then all validation CVs.
6. Add projects, education, and certifications.
7. Add document-level cross-page candidate fusion.
8. Run required ablations and compare with equal-budget LayoutLMv3 and SG-ESE.
9. Implement EFSFR using frozen validation predictions from selected models.
10. Freeze successful configurations before ID/OOD evaluation.

## Latest Addition: Positive Project Modules

A second methodological slice first landed on top of the strongest SG-GRSE
lane:

- `eraparse sge repair-project-tech`

This is not a new model. It is a narrow deterministic module that:

- keeps project record count, names, and URLs fixed;
- repairs only noisy project technologies;
- uses candidate evidence plus predicted document skills as alignment hints;
- refuses the broader project-URL synthesis heuristic that failed validation.

Held-out validation effect:

| Method | Macro | Projects | Work experience |
|---|---:|---:|---:|
| selective EFSFR + SG-GRSE work | 0.776183 | 0.645540 | 0.931212 |
| + project technology repair | **0.776322** | **0.647206** | 0.931212 |

Interpretation:

- this is a small but clean second contribution already beating the previous
  practical lane;
- it strengthens the thesis story that modular evidence-grounded repair can
  outperform a monolithic extraction pass even when the change is lightweight;
- the next stronger second contribution was then implemented as a train-derived
  project-URL selector.

### Train-Derived Project URL Selector

Implemented:

- `eraparse sge repair-project-url`

This extension:

- learns URL-presence priors from train-split truth only;
- uses observable project features only;
- synthesizes a GitHub repo URL only when a matching train bucket supports it;
- avoids the earlier unconditional URL heuristic that hurt validation.

Held-out validation effect:

| Method | Macro | Projects | Work experience |
|---|---:|---:|---:|
| + project technology repair | 0.776322 | 0.647206 | 0.931212 |
| + train-derived project URL selector | **0.776740** | **0.652224** | 0.931212 |

Interpretation:

- this is the current strongest practical lane in the project;
- it is a clearer community-style contribution than another isolated checkpoint,
  because it is explicit, reproducible, train-derived, and model-agnostic
  within the evidence-graph pipeline;
- the architecture-edit lane remains important, but the present practical
  winner is modular selective fusion plus deterministic repair.

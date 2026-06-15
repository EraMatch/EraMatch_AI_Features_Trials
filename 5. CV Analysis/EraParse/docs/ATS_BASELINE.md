# ATS Compatibility And Screening Baseline

## Research Question

Does improved CV parsing materially improve candidate retrieval and reduce
false rejection compared with a legacy raw-text ATS workflow?

This is a downstream-utility experiment, not a claim that the benchmark can
decide who should be hired.

## Required Baselines

### Deterministic Legacy Filters

Implement these first because they are inspectable and reproducible:

1. Boolean keyword filter with explicit required, optional, and excluded terms.
2. BM25 ranking over precomputed PyMuPDF/pdfminer text.
3. Deterministic eligibility rules over dates, skills, education, and
   experience when the job profile explicitly includes them.

Record tokenization, normalization, stopword, stemming, synonym, and query
expansion configuration. No hidden LLM query expansion is allowed in the
legacy baseline.

### OpenCATS System Baseline

Use OpenCATS as a legacy open-source ATS system comparator:

- repository: <https://github.com/opencats/OpenCATS>
- official documentation: <https://documentation.opencats.org/>
- pinned release: `0.9.7.4`
- pinned commit: `5781f41814f09493c7ce05b9251b5e09c1aff0cd`

Measure PDF ingestion, searchable-text availability, full-text query behavior,
latency, failures, and returned candidate ordering. Run it in an isolated
container with synthetic benchmark identities only. OpenCATS is a single
legacy system baseline; do not generalize its result to every commercial ATS.

The deterministic Boolean/BM25 baseline is implemented and must be completed
before OpenCATS. OpenCATS results remain a separate system-integration result;
never label deterministic baseline output as OpenCATS output.

### Optional ATS-Readability Comparator

OpenResume may be tested as an optional parser/readability lane:

- repository: <https://github.com/xitanggg/open-resume>
- parser purpose: testing ATS readability

OpenResume is not an applicant-tracking or candidate-filtering system. Label
its results as parser/readability compatibility, never as an ATS ranking
baseline.

## Job Profile And Relevance Contract

The current EraMatch dataset contains CVs and primary domains, but it does not
contain real job descriptions, recruiter shortlists, interview outcomes, or
hiring decisions. Therefore:

1. Create versioned job profiles from the training taxonomy only.
2. Each profile declares required skills, optional skills, exclusions, and any
   explicit experience/education constraints.
3. Generate weak relevance labels deterministically from canonical ground
   truth using those declared rules.
4. Keep query construction separate from candidate predictions.
5. Freeze profiles, rules, and cutoffs before final ID/OOD evaluation.
6. Create a human-labeled CV-job pair subset before making claims about real
   screening quality.

Weak labels answer whether a pipeline recovers candidates satisfying the
declared benchmark rules. They do not answer whether a person should be hired.

## Comparison Lanes

For identical job profiles and candidate pools, compare:

1. OpenCATS raw PDF ingestion and full-text search.
2. Boolean and BM25 over precomputed parser text.
3. Boolean and BM25 over oracle text as an upper bound.
4. Deterministic filtering/ranking over canonical structured targets as an
   oracle upper bound.
5. The same structured filtering/ranking over selected EraParse predictions.

This isolates whether losses come from PDF ingestion, text extraction,
structured parsing, or filtering logic.

## Candidate Pools And Staging

- `debug_50`: integration smoke test only.
- `debug_250`: query and failure-distribution check.
- train/validation: create and select profiles, synonyms, weights, and cutoffs.
- ID test: final in-distribution screening result.
- template-OOD test: final formatting/generalization result.
- locked confirmation: one-time final confirmation using frozen profiles and
  configuration.

Do not mix train CVs into final retrieval pools merely to increase pool size.

The frozen training-derived profile artifact is
`configs/ats/domain_job_profiles_v1.json`. Changing the profile-learning
algorithm or training manifest requires a new version.

## Metrics

Report:

- ingestion and query success rates;
- searchable-text and required-field coverage;
- precision@k, recall@k, nDCG@k, and mean reciprocal rank;
- false-rejection rate among rule-eligible candidates;
- rank correlation and top-k overlap;
- latency and resource cost;
- results by tier, template, primary domain, and OCR condition.

Report human-label agreement and adjudication policy when human judgments
exist.

## Safety And Interpretation

- Exclude name, email, phone, LinkedIn, GitHub, and other identity/contact
  fields from ranking.
- Do not infer or rank on protected attributes.
- Treat location constraints as a separate explicitly labelled experiment only
  when a job profile requires location.
- Report disparities as diagnostic benchmark behavior, not as fairness
  certification.
- Never claim that higher benchmark retrieval score means better hiring.

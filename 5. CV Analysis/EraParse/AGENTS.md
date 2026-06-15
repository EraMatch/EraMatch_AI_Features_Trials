# Agent Operating Contract

This file is mandatory reading for every coding or research agent working in
this repository.

## Required Reading Order

1. `README.md`
2. `docs/PROJECT_PLAN.md`
3. `docs/DATASET_CONTRACT.md`
4. `docs/EVALUATION_PROTOCOL.md`
5. `docs/TRIAL_RUNBOOK.md`
6. The relevant subsystem guide:
   - models: `docs/MODEL_CATALOG.md`
   - architecture edits: `docs/ARCHITECTURE_AND_SCIENCE.md`
   - environments: `docs/VERSIONS_AND_ENVIRONMENTS.md`
   - Modal: `docs/MODAL_EXECUTION.md`
7. `docs/DECISIONS_AND_CORRECTIONS.md`
8. For screening work: `docs/ATS_BASELINE.md`
9. For current model work: `docs/SG_ESE_STAGE_LOG.md`, then
   `docs/NEXT_MODEL_ACTION_PLAN.md`

## Non-Negotiable Rules

- Treat `../eramatch_benchmark_v4` as read-only. Never rename, delete, rewrite,
  or generate files inside it.
- Use `ground_truth/*.json` as the set of completed samples.
- Ignore `pdfs/cv_04951.pdf`; it has no completed ground truth.
- Generate manifests, splits, parser outputs, caches, and reports inside this
  project or approved remote storage only.
- Never touch the locked confirmation half until the final method and
  hyperparameters are frozen.
- Never train on validation, ID test, template-OOD test, or locked-confirmation
  samples.
- Never claim final results from training or debug subsets.
- Never describe weak domain/skill-derived ATS relevance labels as real hiring
  decisions or hiring quality.
- Never use name, email, phone, LinkedIn, GitHub, or other identity/contact
  fields as practical candidate-ranking signals.
- Never claim SG-VTC is novel until a formal literature and novelty review is
  complete.
- Never use ground-truth tier, template, field F1, or correctness labels as
  inputs to a practical router.
- Record exact model IDs, Hugging Face revisions, package lockfiles, seed,
  config, hardware, and data manifest hash for every reportable run.
- Record every promoted, stopped, failed, or corrected SG-ESE trial in
  `docs/SG_ESE_STAGE_LOG.md`.
- Do not commit secrets, model caches, datasets, checkpoints, run databases, or
  generated trial outputs.

## Stop Gates

Stop downstream work and report the failure when:

- required artifacts or manifest counts fail the dataset audit;
- a split has overlap or a held-out template leaks into training;
- a model output cannot be parsed or validated reliably on `debug_50`;
- a compatibility spike fails, especially shortened Donut encoder outputs;
- a run lacks reproducibility metadata;
- a paid API, Modal deployment, GPU job, secret creation, or large upload was
  not explicitly requested.

## Implementation Discipline

- Follow the work-package dependency order in `docs/WORK_PACKAGES.md`.
- Start every new method on `debug_50`; promote only after acceptance gates.
- Prefer structured parsers and typed APIs over ad hoc string manipulation.
- Keep source-oracle and realistic OCR results separate.
- Treat random pruning as a negative control, never as the proposed method.
- Add focused tests with each work package and run the audit before dependent
  trials.

## Source Verification

Before implementing volatile APIs, re-check:

- Context7 for Transformers, Docling, and Modal usage;
- the exact Hugging Face model card and pinned revision;
- official Modal GPU names and storage semantics;
- primary papers for architecture claims.

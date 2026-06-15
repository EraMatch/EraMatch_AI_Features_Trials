# Phased Task Plan

Created **June 13, 2026**. Organizes remaining thesis work into three phases
with review checkpoints between them. Clean metric = evidence-absent URL cells
excluded (see `RESULTS_AND_PARETO.md`).

Thesis contribution restated: a CV parser that extracts **complex nested fields**
(work_experience, education, projects, certifications) accurately, is **fast**,
and beats the **parser -> strong-LLM-mapper** approach.

Lane labels used throughout:
- **A** = parser (PyMuPDF4LLM) -> Qwen3-4B mapper (the "novel approach" baseline)
- **B** = NuExtract3 VLM direct + deterministic assembly
- **C** = selective router (B base + selective escalation to A on routed fields)

Cost rule: stop before any launch if projected cumulative stage spend > $15.
Prefer local M1 Pro for readers, fusion, evaluation, calibration, reports.

---

## PHASE 1 — Held-out validation comparison + projects fix  (CURRENT)

**Goal:** produce the fair, clean-metric, held-out (310-validation) comparison
table for A / B / C on the *same* CVs, with projects improved.

**Exit checkpoint:** one table with A, B, C on the 310 validation CVs (clean
metric + latency + per-field), projects score improved, cert-routing decision
made. Review with user before Phase 2.

### 1.1 Projects evidence-absent audit  — local, $0  [COMPLETE]
- [x] Audited NuExtract3 projects failure pattern: 100% miss on T1_functional,
      T1_executive, T3_table, T5_minimal (178/1445 train CVs).
- [x] Verified: 0/N CVs in these templates have project names in pymupdf text —
      the PDF layout simply never renders a projects section. Same artifact
      family as the URL phantom labels.
- [x] Extended `scripts/analyze_labels_and_router.py` with `project_evidence_absent()`
      and `PROJECTS_ABSENT_TEMPLATES` constant. Both URL and project exclusions
      now applied together in `clean_macro()`.
- [x] Updated `docs/RESULTS_AND_PARETO.md` with fully-clean metric.

**Outcome:** projects 0.788 raw → **0.898 clean** (+0.110). No model repair needed.
Fully-clean train macro: B=0.9417, C=0.9531. No LayoutLMv3 port is needed.
The SGE `apply_project_skill_tech_repairs` is irrelevant to NuExtract3 because
the technology extraction is not broken — the template detection is the artifact.

**No LoRA needed for projects** (task 3.1 scope reduced). The remaining nested
bottleneck is work_experience (0.858 B → 0.920 C via routing).

### 1.2 Validation reader/request prep  — local, $0  [verify; mostly done]
- [x] `pymupdf4llm_markdown` for validation: 310/310 present.
- [x] `validation.nuextract3-full.requests.jsonl`: prepared.
- [ ] Prepare full-schema Qwen mapper requests for validation from
      `validation` pymupdf4llm markdown (lane A input).

### 1.3 NuExtract3 on validation (lane B)  — Modal, ~$1.2
- [ ] Run `modal_apps/nuextract3_vllm_trial.py` on
      `validation.nuextract3-full.requests.jsonl` (310), `--use-mtp
      --include-evidence-text`.
- [ ] Ingest `--repair-work-records --full-schema`.
- [ ] No additional projects repair needed (1.1 finding: template artifact,
      not a model problem).

### 1.4 Qwen3-4B full-schema on validation (lane A)  — Modal, ~$1.2
- [ ] Run `modal_apps/mapper_trial.py` full-schema on all 310 validation CVs.
- [ ] Ingest -> lane A on validation.
- [ ] **Reuse note:** this full Qwen run ALSO supplies the router's specialist
      predictions for 1.5 (pick routed fields from it) — no separate focused run
      needed for the *accuracy* table. A small focused run is only needed for the
      *production latency* number (see 1.6).

### 1.5 Router on validation (lane C)  — local, $0
- [ ] Apply primary-only router to validation NuExtract3 (1.3).
- [ ] Fuse routed fields from the full Qwen output (1.4) -> lane C.
- [ ] Run both-fields and work-exp-only variants.

### 1.6 Clean comparison table + latency  — local, $0  [DELIVERABLE]
- [ ] Apply URL exclusion to A, B, C on validation (reuse
      `scripts/analyze_labels_and_router.py` logic).
- [ ] Table: macro, nested-macro, per-field, mean/p50/p95 latency, unsupported
      (faithfulness), escalation rate.
- [ ] Optional small focused-Qwen validation run (~$0.4) to confirm the
      production latency of C if the train timing (3.4 s/CV) is not sufficient
      evidence.

### 1.7 Certifications routing decision  — local, $0
- [ ] Compare both-fields (0.9439 / 843 esc on train) vs work-exp-only
      (0.9377 / 537 esc) on validation.
- [ ] Pick the Pareto point for the thesis default; keep the other as ablation.

**>>> CHECKPOINT 1: review A/B/C validation table with user before freezing.**

---

## PHASE 2 — Freeze + frozen ID/OOD evaluation

**Goal:** lock every config, run once on ID (310) and template-OOD (410),
produce the generalization + Pareto story.

**Exit checkpoint:** frozen config hashes; ID + OOD numbers; full Pareto table;
p50/p95 latency; per-tier/template generalization. Review before Phase 3.

### 2.1 Freeze configs  — local, $0
- [ ] Freeze: NuExtract3 prompt/serving config, router thresholds, repair stack,
      projects repair, URL-exclusion rule, cert-routing choice.
- [ ] Record manifest + config hashes. No further selection after this point.

### 2.2 ID-test runs (310)  — Modal, ~$2.4
- [ ] NuExtract3 (310) + full Qwen3-4B (310) -> A, B, C on ID test.
- [ ] Apply frozen repair + router only (no re-tuning).

### 2.3 Template-OOD runs (410)  — Modal, ~$3.2
- [ ] NuExtract3 (410) + full Qwen3-4B (410) -> A, B, C on OOD test.
- [ ] Same frozen configs.

### 2.4 End-to-end latency benchmark  — local, $0
- [ ] One uncached end-to-end timing per lane: reader + model + repair.
- [ ] Report p50 and p95, not just mean. Escalation-weighted latency for C.

### 2.5 Pareto table + faithfulness  — local, $0  [DELIVERABLE]
- [ ] Accuracy vs latency vs $/1000 CVs vs escalation rate.
- [ ] Faithfulness axis (unsupported-evidence rate) — B/C grounded vs A.
- [ ] Accuracy-per-second and accuracy-per-dollar columns.

### 2.6 Generalization analysis  — local, $0
- [ ] Does the router gain hold on template-OOD? Per-tier / per-template
      breakdown of nested fields.
- [ ] Report any OOD degradation honestly.

**>>> CHECKPOINT 2: review ID/OOD + Pareto with user before Phase 3.**

---

## PHASE 3 — Optional accuracy push + locked confirmation + write-up

**Goal:** optional fine-tune for remaining nested headroom, final locked-set
confirmation (once), thesis artifacts.

**Exit checkpoint:** final confirmed numbers on locked set; reproducible tables
and figures; negative results documented.

### 3.1 (Optional) NuExtract3 LoRA fine-tune  — Modal, est. TBD
- [ ] Only if Phase 2 shows nested headroom worth the cost.
- [ ] **Reduced scope** (1.1 finding): projects does not need fine-tuning —
      the 0.788 raw score was entirely an evidence-absent label artifact.
      Target is work_experience (0.858 B-only) and certifications (0.828 B-only)
      if the router doesn't fully close those gaps in Phase 2.
- [ ] Train on `train` (1,445); **validate on template-OOD** to catch overfitting.
- [ ] **Stop rule:** abandon if OOD nested macro does not improve, or if
      unsupported-evidence rises > 0.005.
- [ ] If adopted, re-freeze (back to 2.1 discipline) before locked use.

### 3.2 Locked confirmation (2,475)  — Modal, est. TBD  [ONCE ONLY]
- [ ] Run the single frozen winning config(s) on the locked half exactly once,
      after all selection is complete.
- [ ] Estimate cost first (measured $/CV x 2,475 x 1.3). Confirm under budget.
- [ ] No method changes allowed after seeing locked results.

### 3.3 Thesis artifacts  — local, $0
- [ ] Final comparison tables (A/B/C, clean metric, all splits).
- [ ] Pareto figures (accuracy/latency/cost/faithfulness).
- [ ] Field-complementarity figure (why the router works).
- [ ] Negative results: SG-ESE/SG-GRSE underperformance, projects-repair outcome,
      cert non-selectivity, URL label artifact.
- [ ] Reproducibility manifest (config + manifest hashes per table).

**>>> CHECKPOINT 3: final review.**

---

## Cost rollup (rough)

| Phase | Paid runs | Est. cost |
|---|---|---:|
| 1 | validation NuExtract3 + full Qwen (+ optional focused) | ~$2.4–2.8 |
| 2 | ID (310) + OOD (410), both lanes | ~$5.6 |
| 3 | optional LoRA + locked (2,475) | TBD, estimate before launch |

Local M1 Pro work (readers, fusion, repair, calibration, evaluation, reports,
latency benchmarks) is $0 and should be maximized.

## Dependency notes

- 1.1 (projects repair) should land BEFORE 1.3 ingest so validation captures the
  improved projects score in one pass.
- 1.4 full Qwen serves double duty: lane A baseline AND router specialist source.
- Nothing in Phase 2 may re-select thresholds/prompts/rules — Phase 1 freezes them.
- Phase 3 locked set is touched exactly once, ever.

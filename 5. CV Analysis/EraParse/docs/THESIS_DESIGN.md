# Thesis Design — Faithful, Schema-Targeted Nested Extraction with Small Models

**Status:** design spec for review (created 2026-06-14). Supersedes the
"router-only" framing. The router is now positioned as a *practical* artifact;
the thesis core is a *heavy-DL* contribution.

**One-line thesis:** Existing tools achieve schema-targeted extraction by leaning
on large models, hiding a reliability gap that appears with *small* models and
*nested* schemas. We show a sub-4B model can be made a reliable, **faithful**,
schema-conformant nested extractor — via fine-tuning, constrained decoding, and a
novel set-prediction extraction head — matching 12B/VLM quality at a fraction of
cost and latency.

---

## 1. Problem & gap

**Task:** map a CV (PDF) to a fixed target JSON schema with **nested records**
(`work_experience[]`, `education[]`, `projects[]`, `certifications[]`) plus flat
fields.

**Corrected gap (validated against the literature, 2026-06-14).** It is *not*
true that "tools don't target a schema" — Docling, LangExtract, NuExtract,
Outlines/Instructor all do. The gap that survives scrutiny:

1. **Nested-record reliability.** Schema-targeting works for flat fields; nested
   multi-entry records degrade (PARSE 2025, DeepJSONEval 2025). Confirmed by our
   prior trials: a 1B mapper emits *valid* JSON 82–100% of the time but the
   nested content is wrong (see `PRIOR_KAGGLE_TRIALS.md`). **Validity ≠ accuracy.**
2. **Faithfulness.** Generative schema-targeting hallucinates values absent from
   the document, and almost nobody measures it for CV parsing (FaithLens 2025:
   faithfulness is ~3× less studied than factuality). We already track
   `unsupported_evidence_rate` — this is a moat.
3. **Cost/faithfulness frontier for small models is uncharacterized.** Tools hide
   the gap behind big models; we measure it directly down to 1B.

## 2. Research questions

- **RQ1 (characterization):** How do approaches trade off **accuracy ×
  faithfulness × speed × cost** for schema-targeted nested extraction, across
  schema complexity?
- **RQ2 (the contribution / training):** Can fine-tuning make a 1–4B model
  reliably produce valid, *accurate*, *faithful* nested JSON? How does this scale
  with **model size × schema complexity**?
- **RQ3 (method / decoding):** Does constrained decoding (guaranteed validity)
  **help or hurt** content accuracy and faithfulness, and how does it interact
  with fine-tuning?
- **RQ4 (architecture):** Does a **set-prediction extraction head with
  copy/pointer sub-heads** on a pretrained backbone beat sequence-generation for
  nested-record accuracy and faithfulness — at equal or lower cost?
- **RQ5 (practical, secondary):** Does per-field routing/distillation from a
  strong teacher add value beyond a single small model? *(the router, demoted to
  a practical result)*
- **RQ6 (build-your-own VLM):** Can we replicate the NuExtract *recipe* (general
  VLM base + schema-conditioned SFT) on a **fresher/faster** base, specialized to
  CVs, and match NuExtract3's accuracy at lower size/latency with better
  faithfulness? How does it scale with base size (2.2B vs 4B)?

## 3. Approach taxonomy (comparison space for RQ1)

| # | Approach | Role |
|---|---|---|
| 1 | Parser {PyMuPDF, pymupdf4llm, Docling} → mapper LLM {1B, 4B, 12B} | baseline |
| 2 | VLM direct→JSON (NuExtract3, Qwen-VL) | baseline |
| 3 | External schema-targeting tools (Docling-extract, LangExtract) | tool baseline |
| 4 | Token-classification + assembly (LayoutLMv3 + EFSFR) | grounded/faithful baseline |
| 5 | Selective router (1 + 2) | **practical** synthesis (RQ5) |
| 6 | ★ Fine-tuned model ladder {Qwen3-4B, Gemma-3-1B, NuExtract-1.5, NuExtract-1.5-tiny} + constrained decoding | **contribution track A** (RQ2/RQ2b/RQ3) |
| 7 | ★ Set-prediction head + copy sub-heads on pretrained backbone (NuExtract/LayoutLMv3 encoder) | **contribution track B** (RQ4) |
| 8 | ★ Build-your-own extraction VLM (EraExtract) on a fresh/fast base {SmolVLM2-2.2B, Qwen3.5-VL-4B} | **contribution track C** (RQ6) |

Prior parser results (`PRIOR_KAGGLE_TRIALS.md`) already indicate **pymupdf4llm**
is the best speed/quality text parser → use it as the default parser feed.

## 4. The novel contribution (heavy-DL core)

Run **two tracks** so the thesis has a guaranteed result even if the high-risk
track underperforms.

### Track A — Reliable small generative extractor (lower risk, guaranteed result)
- Fine-tune a **model ladder** on the full `train` split (both schemas),
  faithfulness-aware supervision, vLLM-served:

  | Role | Model | Size |
  |---|---|---|
  | General-SLM baseline | Qwen3-4B-Instruct-2507 | 4B |
  | General-SLM tiny | Gemma-3-1B | 1B |
  | ★ Purpose-built tiny | NuExtract-1.5-tiny (Qwen2.5-0.5B base) | 0.5B |
  | ★ Purpose-built mid | NuExtract-1.5 (Phi-3.5-mini base) | 3.8B |

- **Ablation (RQ2b):** *does starting from a structured-extraction-pretrained
  base (NuExtract) beat a general SLM (Gemma/Qwen) when both are fine-tuned on the
  same data?* Unanswered for document/CV IE → a publishable finding. The 0.5B
  purpose-built extractor is the vehicle for the headline claim ("smallest
  reliable faithful nested extractor").
- **NuExtract template caveat:** NuExtract-1.5 uses a custom
  `<|input|>…<|output|>` schema-as-template, not chat format — fine-tuning must
  match it (verify before committing).
- **Constrained decoding** (Outlines / vLLM grammars) as the validity guarantee;
  measure validity *and* the accuracy/faithfulness delta of forcing validity.
- Optional: distill from the strong teacher (router C / 12B / VLM).
- Claim: *match the teacher's nested accuracy + better faithfulness at N× lower
  cost/latency.*

> **Prior small-model evidence (zero-shot, motivates fine-tuning):** Qwen3-0.6B
> raw macro 0.586 (faithful, weak nested); **Phi-4-mini 0.579 but hallucinated
> 21%** (`unsupported=0.215`); Qwen3-4B 0.775 (faithful). Every prior small-model
> trial was zero-shot — fine-tuning is the untested lever.

### Track B — Set-prediction extraction head (high risk, the architectural novelty)
The crown contribution. Targets the actual gap (nested content + faithfulness):

- **Backbone (pretrained, not from scratch — SG-ESE lesson):** a grounded
  encoder — LayoutLMv3 (text + 2D layout), a strong text/VLM encoder, or **a
  NuExtract encoder** (extraction-pretrained grounding — strongest architectural
  story, since the backbone already encodes schema-extraction priors; decide
  concretely at Track-B build time).
- **Schema-as-queries:** schema fields + N learnable **record-slot queries**
  cross-attend to document tokens (DETR-style object queries).
- **Set-prediction loss:** bipartite **Hungarian matching** between predicted
  record-slots and gold records — the same matching used in evaluation — with a
  "no-record" class for empty slots. Handles variable cardinality natively.
- **Copy/pointer sub-heads (A2):** extractive sub-fields (company, title, dates,
  institution, skills) are decoded as **spans pointing into source tokens** →
  *cannot hallucinate* → faithfulness by construction. Only abstractive fields
  (summary, normalized dates) generate.
- **Validity by construction:** only known schema slots are filled; output is
  always schema-valid — no constrained decoding needed.
- **Speed:** parallel slot decode, no long autoregression; optional
  schema-global/document-local sparse attention (B2) for long CVs.

If Track B underperforms the baselines, it is still a valid **comparative/negative
result** and Track A carries the practical claim.

### Track C — Build-your-own extraction VLM ("EraExtract")
Replicate the **NuExtract recipe** on a fresher/faster base, specialized to CVs.
The recipe (from NuExtract 1.5/2.0/3): *general VLM base + schema-conditioned SFT
on `(document, schema-template, JSON)` triples + schema-as-template prompt format,
trained over MANY schemas so it follows arbitrary schemas zero-shot.*

- **We already hold the hardest ingredient:** 4,500 real `(CV image, gold JSON)`
  pairs (NuMind had to synthesize theirs).
- **Schema augmentation:** generate many schema variants per CV (random field
  subsets / renames / reorders) so the model learns to follow *arbitrary* target
  schemas, not just our fixed one — the key to a NuExtract-like generalist.
- **Bases (size ablation):**
  | Base | Size | Story |
  |---|---|---|
  | SmolVLM2-2.2B | 2.2B | "faster" — document-pretrained (Docmatix), Apache-2.0; match NuExtract3 at ~half size |
  | Qwen3.5-VL-4B | 4B | "potential" — same lineage as NuExtract3; specialize to CVs to beat it |
- **Training:** SFT (LoRA or full) of the VLM on CV image → JSON, faithfulness in
  the objective.
- **Claim:** *a domain-specialized, faster extraction VLM that matches NuExtract3
  on accuracy at lower size/latency with better faithfulness.* The win is
  **speed + domain-fit + faithfulness**, not necessarily raw-accuracy SOTA.
- **Composes with Track B:** the set-prediction head can be built on the EraExtract
  backbone.
- **Cost note:** VLM image-input SFT is the most GPU-intensive track (tens of $,
  several iterations) — schedule after Tracks A/B validate the data + metric.

## 5. Evaluation framework

**Axes (always reported together):**
- **Accuracy:** clean macro / nested-macro / per-field (Hungarian + ANLS).
- **Validity:** schema-valid %, JSON-parse %.
- **Faithfulness:** `unsupported_evidence_rate` (hallucination) — first-class axis.
- **Speed:** p50/p95 latency, vLLM-served (not naive HF generate).
- **Cost:** $/1,000 CVs.

**Clean metric:** exclude evidence-absent label cells (URL phantoms + 4
non-rendering project templates), applied identically to every approach — see
`RESULTS_AND_PARETO.md`.

**Two schema complexities (difficulty axis):**
- **Reduced** 12-field (primary; what A/B/C/router already use).
- **Rich** schema (Kaggle pilot: `seniority_level`, `primary_domain`,
  `years_of_experience`, nested `contact_info`, etc.) as a stress test —
  show how each approach degrades as schema complexity grows.

## 6. Data

- `eraparse` benchmark v4 splits: train (1,445), validation (310), id_test,
  template_ood (410), **locked confirmation (2,475 — touched once, ever).**
- Regenerate ShareGPT-format train/eval from these splits for **both schemas**, so
  fine-tunes use the same splits as A/B/C (the Kaggle 610/128 was a subset →
  expand to the full train split on Modal).

## 7. Phased plan

```
Phase A — Baseline characterization
  A1 finish A/B/C validation + faithfulness        ← in progress
  A2 parsers {pymupdf, pymupdf4llm, docling} × mappers {1B,4B,12B}, vLLM-served
  A3 external baselines: Docling-extract, LangExtract
  → CHECKPOINT A

Phase B — Contribution (Tracks A and B run IN PARALLEL)
  B1 evaluate the EXISTING fine-tuned 1B adapter (re-run; cheap; answers "does FT fix 31%?")
  ── Track A (generative, lower risk) ──
  B2 fine-tune ladder {Qwen3-4B, Gemma-3-1B, NuExtract-1.5, NuExtract-1.5-tiny} × {reduced,rich}, vLLM
  B3 constrained decoding on/off — validity & accuracy/faithfulness delta
  ── Track B (set-prediction head, crown novelty) ──
  B4 set-prediction head + copy sub-heads on pretrained backbone
  ── shared ──
  B5 (optional) distillation from teacher
  → CHECKPOINT B  (compare Track A vs Track B head-to-head)

Phase C — Build-your-own extraction VLM (EraExtract) [most GPU-intensive]
  C1 schema-augmentation data pipeline (CV image → many schema-template/JSON triples)
  C2 SFT EraExtract on {SmolVLM2-2.2B, Qwen3.5-VL-4B}, faithfulness-aware
  C3 eval vs NuExtract3 on accuracy × speed × cost × faithfulness; size ablation
  → CHECKPOINT C

Phase D — Frozen evaluation + write-up
  D1 freeze all configs; run id_test + template_ood once
  D2 accuracy × speed × cost × faithfulness frontier (Pareto, all tracks)
  D3 locked confirmation (2,475) — once only
  D4 thesis tables, figures, negative results, reproducibility manifest
  → CHECKPOINT D
```

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Track B (custom head) underperforms (SG-ESE precedent) | Build on pretrained backbone; Track A guarantees a result; negative result still publishable |
| Fine-tune doesn't beat teacher | Frame as "match at lower cost," not "beat"; cost/faithfulness is the win |
| Compute cost (real training) | Cap per-phase; Modal spot GPUs; LoRA not full FT; start at 1B |
| max_seq truncation (Kaggle bug) | Longer context or compact-schema prompt; vLLM serving |
| Rich-schema invalidates reduced numbers | Keep schemas as separate reported tracks, never mixed |
| Track C (EraExtract) can't beat NuExtract3 on accuracy | Frame win as speed + size + faithfulness, not raw SOTA; size ablation still a result |
| Track C VLM SFT is GPU-heavy | Schedule after A/B validate data+metric; LoRA first; start with SmolVLM2-2.2B (cheaper) |

## 9. Prior work
See `PRIOR_KAGGLE_TRIALS.md` (parser benchmark v3 + 1B QLoRA pilot v4) and
`RESULTS_AND_PARETO.md` (rigorous A/B/C + clean metric). Key reading:
FrugalGPT / routing survey (2603.04445), FaithLens (2512.20182), conformal
extraction (2603.00924), schema-driven IE (2305.14336), constrained decoding
("Think Inside the JSON", 2502.14905).

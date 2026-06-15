# EraParse Project Plan

## Research Objective

Build a reproducible CV parsing benchmark that separates the effects of:

1. document representation and parser quality;
2. grounded semantic extraction capability;
3. structure-assembly and decoding strategy;
4. direct visual document understanding;
5. schema-guided evidence/token selection;
6. evidence-aware hybrid routing.

The primary output is a defensible research comparison, not a single production
parser. Accuracy, evidence support, latency, memory, and generalization must all
be reported.

## Contribution Hierarchy

1. **Evaluation contribution:** field-aware, evidence-aware CV extraction
   evaluation with true unseen-template OOD testing.
2. **Representation contribution:** controlled comparison of plain text,
   Markdown, and structured layout serialization.
3. **Architecture hypothesis:** SG-ESE, schema-guided evidence selection and
   extraction with deterministic assembly. It is not called novel until the
   literature review supports that claim.
4. **Accuracy architecture hypothesis:** SG-GRSE, a schema-guided grounded
   record-set decoder that directly predicts variable repeated records using
   document-level evidence and Hungarian matching.
5. **Methodology contribution:** EFSFR, an evidence-first selective-fusion and
   repair protocol with calibrated field/record decisions and selector traces.
6. **Efficiency hypothesis:** SG-VTC as a conditional extension that retains
   evidence/visual tokens according to schema relevance after a reliable
   full-token extractor exists.
7. **System contribution:** a practical observable-signal router that selects or
   repairs extraction outputs without using ground-truth metadata.
8. **Downstream utility contribution:** measure whether better extraction
   reduces false rejection and improves candidate retrieval compared with a
   legacy keyword/ATS workflow.

## Experiment Corpus

- Completed source corpus: 4,950 samples.
- Working corpus: exact stratified half, 2,475 samples.
- Locked confirmation corpus: complementary 2,475 samples.
- Working splits: train 1,445; validation 310; ID test 310; template-OOD test
  410.
- Dataset source remains read-only.

See `DATASET_CONTRACT.md` for the deterministic selection and split contract.

## Implementation Phases

### Phase 0: Foundation

- Audit observed artifacts and generate a deterministic project-local manifest.
- Generate leakage-free working and locked splits.
- Implement schemas, evaluator, evidence checks, and run database.
- Freeze artifact contracts before model trials.

### Phase 1: Parser Representations

- Compare precomputed PyMuPDF and pdfminer text.
- Generate PyMuPDF4LLM Markdown and JSON.
- Generate Docling Markdown and structured exports using controlled configs.
- Select the best two practical representations using validation results.

### Phase 2: Text Mappers

- Run NuExtract Tiny input-format ablation.
- Compare NuExtract Tiny, Qwen3-0.6B, and conditional Phi-4 Mini on the best two
  representations.
- Evaluate section decomposition for small general mappers.
- Use Gemma Cloud only as a selected upper bound or repair route.

### Phase 2A: ATS Compatibility And Screening

- Build a deterministic Boolean-keyword and BM25 legacy-screening baseline.
- Integrate pinned OpenCATS as a system-level ingestion and full-text-search
  comparator.
- Compare raw-text legacy filtering with filtering over EraParse structured
  outputs.
- Treat OpenResume as an optional parser/readability comparator, not as an ATS.
- Build job profiles and relevance judgments under the contract in
  `ATS_BASELINE.md`; do not equate keyword overlap with hiring quality.

### Phase 3: Direct Visual Models

- Treat the completed Donut full-schema run as a failed quality baseline.
- Run Donut tiny-overfit, schema-complexity, decomposition, and page-wise
  ablations before any additional full Donut fine-tune.
- Evaluate NuExtract3 first as the structured visual upper bound.
- Evaluate PaddleOCR-VL-1.6 as a representation upper bound mapped through the
  frozen text mapper.
- Keep optional models from delaying the main path.

### Phase 4: Grounded Schema Extraction

- Build the canonical evidence graph with text/span, page, box, confidence, and
  reading-order provenance.
- Compare free JSON, grammar-constrained generation, section decomposition,
  token/span classification, and SG-ESE field-query extraction.
- Use deterministic schema assembly for non-generative lanes.
- Compare repeated-record grouping/linking separately from scalar extraction.

### Phase 4A: Grounded Record-Set Extraction

- Build a document-level candidate-span bank across pages.
- Predict repeated work, education, project, and certification records as
  grounded unordered sets.
- Train with Hungarian-matched record slots and field-conditioned span
  pointers.
- Compare record-set extraction with deterministic and learned post-hoc
  grouping.
- Require an equal-budget validation improvement over standard LayoutLMv3
  before making an accuracy-contribution claim.

### Phase 5: Efficiency Architecture Trials

- Require a schema-valid full-token visual model before the shortened-output
  compatibility spike.
- Profile encoder, decoder, total latency, memory, and visual-token counts.
- Evaluate SG-VTC controls and schema-aware variants.
- Evaluate LayoutLMv3 source-oracle and OCR-realistic modes separately.

### Phase 5A: Evidence-First Selective Fusion And Repair

- Calibrate field and record confidence using validation only.
- Fuse grounded candidates using agreement, support, confidence, and schema
  consistency.
- Repair only uncertain fields or records.
- Report risk/coverage, correction cost, accuracy, latency, and model-use cost.

### Phase 6: Routing And Confirmation

- Build oracle and practical hybrid routers.
- Freeze the selected methods and hyperparameters.
- Evaluate once on ID test and template-OOD test.
- Use the locked half only for final confirmation.

## Success Criteria

- Every final table is reproducible from a versioned config and manifest hash.
- No split leakage or use of locked data during method selection.
- Results report accuracy, support, latency, memory, and failure categories.
- Downstream screening results report retrieval quality, false rejection,
  ingestibility, and tier/template disparities without using identity/contact
  fields as ranking features.
- Architecture conclusions remain valid even when SG-VTC fails: oracle failure
  rejects compression; oracle success plus practical failure identifies region
  selection as the bottleneck.

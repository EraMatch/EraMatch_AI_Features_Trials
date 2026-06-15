# Decisions And Corrections

This log overrides earlier pasted plans and assumptions.

| Earlier assumption | Correct project decision |
|---|---|
| Dataset has 5,000 completed samples | It has 4,950 completed ground-truth samples; `cv_04951.pdf` is orphaned. |
| Existing metadata and splits can be consumed | No root metadata file exists and source `splits/` is empty; generate project-local files. |
| Donut targets wrap a JSON string in `target` | Observed Donut targets are JSON objects. |
| The pasted schema is the complete ground-truth shape | Full ground truth includes aliases, nested contact information, and extra work metadata. |
| Template prefix identifies tier | Read tier from layout annotations; T4 can reuse T1/T2/T3 template names. |
| Existing template-OOD split is sufficient | True OOD requires complete exclusion of held-out templates from training and validation. |
| The existing `cv-parsing-data` Modal volume can be reused for Donut | A matching filename produced a different SHA-256 hash. Trial 3 uploads audited local images to a dedicated EraParse volume. |
| Donut labels should include `<s_eraparse>` | The task prompt is supplied as `decoder_start_token_id`; reference fine-tuning targets contain only the structured target plus EOS. Duplicating the task prompt caused degenerate generation and invalidated the first Donut adaptations. |
| T4 boxes represent realistic OCR | Existing boxes are source-PDF oracle annotations; realistic OCR needs generated boxes and alignment. |
| `donut-base` is ready for schema extraction | It is a base model and must be fine-tuned for the reduced schema. |
| Donut has a fixed 30x80/2,400-token grid | Infer encoder token/grid shape and processor transforms at runtime. |
| Post-encoder pruning reduces encoder cost | It reduces decoder cross-attention/memory; judge latency using decoder share. |
| Schema-prior pruning may use true tier/template | Global priors are deployable; predicted-class priors are practical; true metadata is oracle-only. |
| Random pruning is a proposed method | It is a negative control only. |
| Shortened Donut encoder output will work automatically | Require a compatibility spike with `BaseModelOutput` and a matching attention mask. |
| Practical router may use field F1 or ground truth | Practical routing uses observable signals only. |
| Ollama model is `gemma3:12b` | The cloud tag is `gemma3:12b-cloud`; prompt JSON, validate, repair, and cache because cloud structured-output enforcement is unavailable. |
| PP-DocBee is a small Transformers baseline | `PaddlePaddle/PP-DocBee-2B` is an optional Paddle-based 2B comparator. |
| A CV-only dataset is enough to measure ATS filtering quality | It can measure ingestion and weak rule-based retrieval, but real screening-quality claims require job profiles and independent relevance judgments. |
| Debug-50/debug-250 results are the experiment result | Debug subsets are promotion gates only. Reportable selection uses validation, and frozen final methods use the complete ID/OOD and locked-confirmation splits. |
| OpenResume is an open-source ATS filtering baseline | OpenResume is a parser/readability comparator. OpenCATS is the selected legacy open-source ATS system baseline. |
| Qwen3-0.6B has exactly 0.6B parameters | Hugging Face currently reports about 751.6M parameters. |
| LayoutLMv3 can be used without license review | `microsoft/layoutlmv3-base` is CC-BY-NC-SA-4.0; this project assumes research/thesis-only use. |
| LayoutLMv3 accepts only words and boxes | Use image, words, normalized boxes, integer labels, pixel values, and chunk documents over 512 tokens. |
| PyMuPDF4LLM needs a custom layout serializer | Use native `to_markdown()` and `to_json()` first. |
| Docling format options use a string `"pdf"` key | Use typed `InputFormat.PDF: PdfFormatOption(...)`. |
| One environment can host every model | Use separate CPU parser, Transformers 4, modern VLM, and Modal lanes. |
| Reload every Modal Volume before reading | Commit after writes; reload only when a long-lived container needs another container's commit. |

## Standing Assumptions

- This is a research/thesis project, not a commercial deployment.
- The dataset is immutable and available at `../eramatch_benchmark_v4`.
- Initial version pins are compatibility lanes; implementation work must add
  lockfiles and smoke-test exact revisions.
- Source and model information was last checked on 2026-06-09.

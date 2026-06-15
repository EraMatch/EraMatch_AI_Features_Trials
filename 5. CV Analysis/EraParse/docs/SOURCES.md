# Sources

Sources were checked on **2026-06-09**. Agents must re-check volatile model
cards, package APIs, Modal hardware names, and revisions before implementation.

## Model Cards And Usage

- [NuExtract 1.5 Tiny](https://huggingface.co/numind/NuExtract-1.5-tiny)
- [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Phi-4 Mini Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Donut Base](https://huggingface.co/naver-clova-ix/donut-base)
- [LayoutLMv3 Base](https://huggingface.co/microsoft/layoutlmv3-base)
- [NuExtract3](https://huggingface.co/numind/NuExtract3)
- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [Granite Docling 258M](https://huggingface.co/ibm-granite/granite-docling-258M)
- [PP-DocBee 2B](https://huggingface.co/PaddlePaddle/PP-DocBee-2B)
- [Gemma 3 12B Cloud](https://ollama.com/library/gemma3:12b-cloud)
- [Ollama Cloud](https://docs.ollama.com/cloud)
- [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)

## Library And Platform Documentation

- [Transformers documentation](https://huggingface.co/docs/transformers/)
- [Transformers 4.57.3 Donut guide](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/donut)
- [Reference Donut CORD fine-tuning notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_%28CORD%29_with_PyTorch_Lightning.ipynb)
- [Docling minimal example](https://docling-project.github.io/docling/examples/minimal/)
- [Docling custom conversion](https://docling-project.github.io/docling/examples/custom_convert/)
- [PyMuPDF4LLM documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [Modal GPU guide](https://modal.com/docs/guide/gpu)
- [Modal pricing](https://modal.com/pricing)
- [Modal Volumes](https://modal.com/docs/guide/volumes)
- [Modal model weights](https://modal.com/docs/guide/model-weights)
- [Modal Hugging Face guide](https://modal.com/docs/guide/huggingface)

## ATS And Readability Baselines

Checked on **2026-06-11**.

- [OpenCATS repository](https://github.com/opencats/OpenCATS)
- [OpenCATS documentation](https://documentation.opencats.org/)
- [OpenCATS 0.9.7.4 release](https://github.com/opencats/OpenCATS/releases/tag/0.9.7.4)
- [OpenResume repository and parser/readability description](https://github.com/xitanggg/open-resume)

## Primary Architecture And Evaluation Papers

- [Unified Structure Generation for Universal Information Extraction](https://aclanthology.org/2022.acl-long.395/)
- [LMDX: Language Model-based Document Information Extraction and Localization](https://aclanthology.org/2024.findings-acl.899/)
- [Grammar-constrained decoding for structured information extraction](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1406857/full)
- [Layout-Aware Instruction Tuning for Visual Information Extraction](https://link.springer.com/chapter/10.1007/978-981-97-8511-7_20)
- [Donut: OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)
- [LayoutLMv3](https://arxiv.org/abs/2204.08387)
- [DynamicViT](https://arxiv.org/abs/2106.02034)
- [TokenLearner](https://arxiv.org/abs/2106.11297)
- [Token Merging](https://arxiv.org/abs/2210.09461)
- [VisFocus](https://arxiv.org/abs/2407.12594)
- [Token-level Correlation-guided Compression](https://arxiv.org/abs/2407.14439)
- [Index-Preserving Lightweight Token Pruning](https://arxiv.org/abs/2509.06415)
- [DocPrune](https://arxiv.org/abs/2604.22281)
- [FastOCR](https://arxiv.org/abs/2605.17447)
- [RTPrune](https://arxiv.org/abs/2605.00392)
- [PP-DocBee2](https://arxiv.org/abs/2503.04065)
- [Qwen3](https://arxiv.org/abs/2505.09388)
- [PaddleOCR-VL-1.6](https://arxiv.org/abs/2606.03264)
- [ANLS and DocVQA](https://arxiv.org/abs/2007.00398)

## Verification Notes

- Context7 was used for Transformers 4.57.3, Docling, and Modal API patterns.
- Hugging Face Hub live metadata was used for exact model IDs, architecture
  tags, sizes, licenses, and current model-card usage.
- Official Modal documentation is authoritative for current GPU names and
  Volume semantics.
- Paper summaries are orientation aids; architecture claims must cite and read
  the primary papers directly.

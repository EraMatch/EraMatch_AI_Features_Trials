# EraMatch AI Features Trials

Research, experiments, and model fine-tuning for EraMatch's AI recruitment features.

## Structure

### [1. GitHub Inspired Question](./1.%20GitHub%20Inspired%20Question)
ACE-Recruiter engine: evaluating developer profiles using GitHub-inspired rubrics, parallelization, and latency optimization.

### [2. Question Generation and Extraction](./2.%20Question%20Generation%20and%20Extraction)
Automated QAG and answer evaluation. Fine-tuning notebooks for general QAG, MCQ, and essay scoring (Llama, Qwen, T5, RoBERTa).

### [3. Job Description QAG & Keywords Transformation](./3.%20Job%20Description%20QAG%20&%20Keywords%20Transformation)
JD analysis: keyword extraction and transformation into screening questions. Fine-tuning notebooks and data generation pipelines.

### [4. Behavioural Analysis](./4.%20Behavioural%20Analysis)
Behavioral trait parsing and analysis from candidate interviews.

### [5. CV Analysis](./5.%20CV%20Analysis)
CV parsing, feature extraction, and JD alignment.
- `dataset/` — v3 and v4 generation pipelines, generated data, EraParse 4,950-CV manifest, v4 dataset (split zip)
- `benchmarking/` — Extraction + LLM structuring benchmarks, Kaggle kernel outputs
- `phase_1_finetunes/` — QLoRA Gemma-3-1B fine-tune on CVSchema, benchmark results
- `EraParse/` — Next-gen CV parsing pipeline (under construction)
- `scripts/` — CV-JD alignment generation

### [6. Code Generation](./6.%20Code%20Generation)
Automated code generation and LeetCode dataset analysis.

### [7. Code Plagiarism Detection](./7.%20Code%20Plagiarism%20Detection)
Code plagiarism and similarity detection experiments.

### [8. Avatar Detection](./8.%20Avatar%20Detection)
AI-generated face detection for live interview verification.
- `src/` — Dual-branch model (DCT + SRM frequency features, RGB spatial)
- `notebooks/` — EfficientNet → CNN-LSTM → multimodal → DCT frequency domain
- `modal/` — Serverless training (DCT, SRM, cross-modal, video-level)
- `kaggle/` — SRM + ConvNeXt deployment trials
- `tests/` — Pytest suite
- `results/` — Thesis summary, checkpoints, plots

---

See subdirectory READMEs for details on each module.

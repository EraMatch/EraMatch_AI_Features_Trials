# EraMatch AI Features Trials

Welcome to the **EraMatch AI Features Trials** repository! 

This repository serves as a centralized workspace for all research, experimentation, model fine-tuning, and data generation pipelines that power the AI-driven recruitment and candidate evaluation features of EraMatch.

## 🗂️ Project Structure

The repository is organized into distinct feature modules, each containing its own specialized scripts, Jupyter notebooks, and datasets.

### [1. GitHub Inspired Question](./1.%20GitHub%20Inspired%20Question)
Trials and experiments for the **ACE-Recruiter** engine. It focuses on evaluating candidate developer profiles using GitHub-inspired performance rubrics, testing parallelization strategies, and optimizing latency.

### [2. Question Generation and Extraction](./2.%20Question%20Generation%20and%20Extraction)
This module explores automated Question Answer Generation (QAG) and Answer Evaluation.
- **`notebooks/`**: Task-categorized fine-tuning notebooks for general QAG, Multiple Choice Questions (MCQ), and essay scoring using state-of-the-art LLMs (Llama, Qwen, T5, RoBERTa, etc.).
- **`scripts/`**: Automation scripts for generating variants and evaluating rubrics.

### [3. Job Description QAG & Keywords Transformation](./3.%20Job%20Description%20QAG%20&%20Keywords%20Transformation)
Pipelines designed to analyze job descriptions, extract core keywords, and transform them into structured, verifiable screening questions.
- **`notebooks/`**: Fine-tuning notebooks specifically targeting keyword extraction and JD-based QAG tasks.
- **`scripts/`**: Job description data generation and processing pipelines.

### [4. Behavioural Analysis](./4.%20Behavioural%20Analysis)
Experiments focused on parsing and analyzing behavioral traits of candidates. Currently contains the core behavioral analysis trials (`eramatch_behavioral_analysis.ipynb`).

### [5. CV Analysis](./5.%20CV%20Analysis) Under Construction (Currently working on)
Focuses on the automated parsing, feature extraction, and alignment of candidate CV structures against specific job description requirements. Contains data generation scripts for parsing alignment.

### [6. Code Generation](./6.%20Code%20Generation)
Trials and notebooks relating to automated code generation and LeetCode dataset analysis.

### [7. Code Plagiarism Detection](./7.%20Code%20Plagiarism%20Detection)
Experiments and notebooks for detecting code plagiarism and similarity.

---

*Note: For detailed information regarding the individual trial pipelines, execution instructions, or fine-tuning approaches, please refer to the `README.md` files located inside each specific subdirectory.*

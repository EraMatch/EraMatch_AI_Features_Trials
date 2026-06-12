# Question Generation and Extraction Trials

This directory is focused on experimenting with and fine-tuning models for Question Answer Generation (QAG) and Answer Evaluation. 

## Directory Structure

### `scripts/`
Contains Python scripts used for various phases of question and rubric generation, as well as variant evaluation.
- `run_answers_variants.py`
- `run_qa_allocation.py`
- `run_rubric_generator.py`
- `run_variant_evaluator.py`

### `notebooks/`
Contains Jupyter notebooks used for fine-tuning state-of-the-art LLMs and analyzing their performance, organized by task:

- **`qag_essay/`**: Fine-tuning notebooks for Essay Question Answer Generation (QAG) using models like:
  - Llama 3.2 3B, Llama 3.1 8B Instruct
  - Qwen 2.5 (1.5B)
  - Mistral 7B, Phi 3.5 Mini
  - Flan-T5 (Base/Large), BART Large
- **`qag_essay_rubric/`**: Fine-tuning notebooks for QA Rubric Generation using Llama 3.1 8B and Qwen 2.5 7B.
- **`qag_mcq/`**: Fine-tuning notebooks specifically for Multiple Choice Question (MCQ) generation.
  - `fine-tuning-qag-mcq-qwen-2-5-7b-instruct.ipynb`
- **`essay_scoring/`**: Essay grading and evaluation notebooks using RoBERTa, DeBERTa v3, and ELECTRA.
- **`analysis/`**: Contains CSVs and plots (loss comparisons, ROUGE metrics) evaluating model performance.

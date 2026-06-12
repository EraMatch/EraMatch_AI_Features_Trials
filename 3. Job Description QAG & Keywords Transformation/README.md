# Job Description QAG & Keywords Transformation Trials

This directory contains trials and scripts for generating job descriptions, extracting keywords, and transforming them into structured screening questions (QAG).

## Directory Structure

### `scripts/`
Contains data generation pipelines for automating JD analysis and processing.
- `generate_job_descriptions_phase1.py`
- `generate_scoring_questions_phase2.py`
- `generate_key_words_jd_phase_3.py`

### `notebooks/`
Contains fine-tuning notebooks specifically for transforming Job Descriptions into questions and extracting keywords, organized by task:

- **`keyword_extraction/`**: Notebooks for fine-tuning models to extract key skills and requirements from JDs.
  - `fine_tuning_jd_keyword_extraction_llama_3_2_3b.ipynb`
  - `fine_tuning_jd_keyword_extraction_qwen_2_5_3b.ipynb`
  - `fine-tunning-jd-keyword_extraction_qwen_2_5_7b.ipynb`
  - `fine_tunning_jd_keyword_extraction_gemma_2_9b.ipynb`
- **`qag/`**: Notebooks for fine-tuning models to generate screening questions from JDs.
  - `fine_tuning_jd_qag_llama_3_2_3b.ipynb`
  - `fine_tuning_jd_qag_qwen_2_5_3b.ipynb`
  - `fine-tuning-jd-qag-qwen-2-5-7b.ipynb`

### `templates/`
Contains markdown templates and data used by the generation scripts.
- `positions.md`: Comprehensive list of various tech roles across different domains.
- `qag_generation.md`: Prompt instructions and guidelines for an LLM to act as an expert technical recruiter generating an HD Eval + QAG screening rubric.

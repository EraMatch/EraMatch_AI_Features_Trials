"""
Rubric and Evaluation Generator for Essay Questions.

This script:
1. Scans data/generated_qa for *_essay.jsonl files.
2. Generates a grading rubric and 15 yes/no evaluation questions for each essay record.
3. Saves the augmented records to data/generated_rubrics with checkpointing.
"""

import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from tqdm import tqdm

project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def normalize_key(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s?]", "", normalized)
    return normalized.strip()


def get_rubric_generation_prompt(context: str, question: str, answer: str) -> str:
    return f"""
You are an expert technical interviewer and grader.
Below is a Context, a Question derived from it, and the Gold Standard Reference Answer.

[CONTEXT]
{context}

[QUESTION]
{question}

[REFERENCE ANSWER]
{answer}

---
TASK:
1. Create a detailed Grading Rubric that defines what a perfect answer must include based on the context.
2. Based on that rubric, generate exactly 15 Yes/No questions.
3. These questions will be used by an LLM to assess a candidate's response.
4. Each question must be objective rather than subjective.

OUTPUT FORMAT:
Return ONLY a JSON object with this structure:
{{
  "rubric_criteria": "The rubric description here...",
  "evaluation_steps": [
    "Question 1?",
    "Question 2?",
    ...
    "Question 15?"
  ]
}}
"""


def load_processed_keys(path: str) -> set[str]:
    keys = set()
    if not os.path.exists(path):
        return keys

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = record.get("question", "")
            if question:
                keys.add(normalize_key(question))

    return keys


def extract_json_payload(raw_text: str) -> dict | None:
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if object_match:
        try:
            parsed = json.loads(object_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def call_rubric_provider(prompt: str, config: dict) -> dict | None:
    if config["provider"] != "ollama":
        raise ValueError(f"Unsupported provider for rubric generation: {config['provider']}")

    payload = {
        "model": config["model"],
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }

    headers = {}
    if config.get("api_key"):
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Ollama-Api-Key": config["api_key"],
        }

    response = requests.post(
        f"{config['base_url']}/api/generate",
        json=payload,
        timeout=240,
        headers=headers,
    )
    response.raise_for_status()

    raw_text = response.json().get("response", "")
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    return extract_json_payload(raw_text)

def process_essay_file(file_info: tuple) -> int:
    input_path, output_path, config = file_info

    seen_questions = load_processed_keys(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:
        logging.error("Could not read file %s: %s", input_path, exc)
        return 0

    results_count = 0
    
    # Inner helper function for a single line to feed into ThreadPool
    def process_single_line(line):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return None

        question_text = record.get("question", "")
        if not question_text or normalize_key(question_text) in seen_questions:
            return None

        prompt = get_rubric_generation_prompt(
            record.get("context", ""),
            question_text,
            record.get("answer", ""),
        )

        try:
            rubric_json = call_rubric_provider(prompt, config)
            if not rubric_json:
                return None

            yes_no_questions = rubric_json.get("evaluation_steps", [])
            if not isinstance(yes_no_questions, list) or not yes_no_questions:
                return None

            record["rubric_description"] = rubric_json.get("rubric_criteria", "")
            record["yes_no_questions"] = yes_no_questions
            return record
        except Exception as exc:
            logging.error("Error processing question '%s...': %s", question_text[:30], exc)
            return None

    # Process lines inside the file using the workers config
    filename_base = os.path.basename(input_path)
    with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
        # Wrap the executor map with tqdm to track line-by-line progress
        for record in tqdm(executor.map(process_single_line, lines), total=len(lines), desc=filename_base, leave=False):
            if record is not None:
                # Thread-safe append to the output file
                with open(output_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_count += 1
                seen_questions.add(normalize_key(record["question"]))

    return results_count

def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_dir = os.getenv("GENERATED_QA_DIR", "data/generated_qa")
    output_dir = os.getenv("GENERATED_RUBRICS_DIR", "data/generated_rubrics")

    config = {
        "provider": "ollama",
        "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "model": os.getenv("QA_MODEL_NAME", "llama3.1:8b"),
        "api_key": os.getenv("OLLAMA_API_KEY"),
        "workers": max(1, int(os.getenv("QA_WORKER_COUNT", "3"))),
    }

    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    essay_files = [f for f in os.listdir(input_dir) if f.endswith("_essay.jsonl")]
    if not essay_files:
        print(f"No essay files found in {input_dir}. Make sure you have run the QA generator script first.")
        return

    print(f"Found {len(essay_files)} essay files. Starting Rubric generation...")
    print(f"Using model: {config['model']} with {config['workers']} workers.")

    tasks = [
        (
            os.path.join(input_dir, filename),
            os.path.join(output_dir, filename.replace(".jsonl", "_rubrics.jsonl")),
            config,
        )
        for filename in essay_files
    ]

    total_generated = 0
    with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
        futures = {executor.submit(process_essay_file, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Overall Progress"):
            try:
                total_generated += future.result()
            except Exception as exc:
                logging.error("Task failed: %s", exc)

    print(f"\nSuccessfully processed {total_generated} rubrics.")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
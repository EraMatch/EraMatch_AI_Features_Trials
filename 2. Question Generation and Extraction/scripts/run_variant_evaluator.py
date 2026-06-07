"""Evaluate answer variants against stored rubric yes/no questions.

This script:
1. Scans data/generated_variants for *_variants.jsonl files.
2. For each record, evaluates every answer variant against the stored
   rubric_description and yes_no_questions using the configured LLM.
3. Saves the augmented records to data/generated_variant_evaluations with
   checkpointing so reruns skip already-processed questions.
"""

import json
import logging
import os
import re
import sys

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


def extract_variant_evaluations_payload(payload: dict, expected_variants: list[str]) -> dict[str, list] | None:
    if not isinstance(payload, dict):
        return None

    nested = payload.get("variant_evaluations")
    if isinstance(nested, dict):
        return nested

    direct_keys = [variant for variant in expected_variants if variant in payload]
    if direct_keys:
        direct_payload = {variant: payload[variant] for variant in direct_keys if isinstance(payload.get(variant), list)}
        if direct_payload:
            return direct_payload

    if all(isinstance(payload.get(key), list) for key in expected_variants if key in payload):
        return {key: payload[key] for key in expected_variants if key in payload}

    return None


def split_into_batches(items: list, batch_sizes: list[int]) -> list[list]:
    batches = []
    start = 0
    for batch_size in batch_sizes:
        end = min(len(items), start + batch_size)
        if start < end:
            batches.append(items[start:end])
        start = end
    if start < len(items):
        batches.append(items[start:])
    return batches


def build_variant_evaluation_prompt(
    context: str,
    question: str,
    reference_answer: str,
    yes_no_questions: list[str],
    answer_variants: dict[str, str],
) -> str:
    questions_text = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(yes_no_questions))
    variants_text = "\n\n".join(
        f"[{variant_name.upper()}]\n{variant_text}"
        for variant_name, variant_text in answer_variants.items()
        if isinstance(variant_text, str) and variant_text.strip()
    )

    return f"""
You are an expert technical evaluator. Your job is to judge one answer variant against a rubric and a list of yes/no checks.

[CONTEXT]
{context}

[QUESTION]
{question}

[REFERENCE ANSWER]
{reference_answer}

[YES/NO QUESTIONS]
{questions_text}

[ANSWER VARIANTS]
{variants_text}

---
TASK:
Evaluate each answer variant against each yes/no question.

Rules:
1. Be strict and objective.
2. Base your judgment on each answer variant, with the context and reference answer as grounding.
3. Return one result per question in the same order for each variant.
4. Each result must include whether the criterion is satisfied and a short reason.

OUTPUT FORMAT:
Return ONLY a JSON object with this structure:
{{
    "variant_evaluations": {{
        "excellent": [
            {{"question": "...", "satisfied": true, "reasoning": "..."}},
            ...
        ],
        "average": [...],
        "poor": [...],
        "pedantic": [...],
        "hallucinated": [...]
    }}
}}
"""


def call_evaluation_provider(prompt: str, config: dict) -> dict | None:
    if config["provider"] != "ollama":
        raise ValueError(f"Unsupported provider for variant evaluation: {config['provider']}")

    payload = {
        "model": config["model"],
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 3200,
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
        timeout=config["request_timeout"],
        headers=headers,
    )
    response.raise_for_status()

    raw_text = response.json().get("response", "")
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    return extract_json_payload(raw_text)


def summarize_evaluations(evaluations: list[dict], total_questions: int) -> dict:
    yes_count = 0
    normalized_items = []

    for evaluation in evaluations:
        satisfied = bool(evaluation.get("satisfied", False))
        if satisfied:
            yes_count += 1
        normalized_items.append(
            {
                "question": evaluation.get("question", ""),
                "satisfied": satisfied,
                "reasoning": evaluation.get("reasoning", ""),
            }
        )

    no_count = max(0, total_questions - yes_count)
    score = yes_count / total_questions if total_questions else 0.0

    return {
        "score": score,
        "yes_count": yes_count,
        "no_count": no_count,
        "total_questions": total_questions,
        "evaluations": normalized_items,
    }


def merge_variant_evaluations(existing: list[dict], new_items: list[dict]) -> list[dict]:
    merged_by_question = {}
    ordered_questions = []

    for item in existing:
        question = item.get("question", "")
        if question and question not in merged_by_question:
            merged_by_question[question] = item
            ordered_questions.append(question)

    for item in new_items:
        question = item.get("question", "")
        if not question:
            continue
        if question not in merged_by_question:
            ordered_questions.append(question)
        merged_by_question[question] = item

    return [merged_by_question[question] for question in ordered_questions if question in merged_by_question]


def process_variant_record(record: dict, config: dict) -> dict | None:
    answer_variants = record.get("answer_variants", {})
    yes_no_questions = record.get("yes_no_questions", [])

    if not isinstance(answer_variants, dict) or not answer_variants:
        return None
    if not isinstance(yes_no_questions, list) or not yes_no_questions:
        return None

    expected_variants = [
        variant_name
        for variant_name in answer_variants.keys()
        if isinstance(answer_variants.get(variant_name), str) and answer_variants.get(variant_name, "").strip()
    ]

    question_batches = split_into_batches(yes_no_questions, [8, 7])
    variant_results = {
        variant_name: {"evaluations": []}
        for variant_name in expected_variants
    }

    for question_batch in question_batches:
        prompt = build_variant_evaluation_prompt(
            record.get("context", ""),
            record.get("question", ""),
            record.get("answer", ""),
            question_batch,
            answer_variants,
        )

        try:
            evaluation_json = call_evaluation_provider(prompt, config)
            if not evaluation_json:
                logging.info("No JSON returned from provider for question: '%s'", record.get("question", "")[:80])
                continue

            raw_variant_evaluations = extract_variant_evaluations_payload(evaluation_json, expected_variants)
            if not isinstance(raw_variant_evaluations, dict):
                logging.info("Provider returned unexpected structure for question '%s': %s", record.get("question", "")[:80], type(raw_variant_evaluations))
                continue

            for variant_name, evaluations in raw_variant_evaluations.items():
                if not isinstance(evaluations, list):
                    continue
                if variant_name not in variant_results:
                    continue
                variant_results[variant_name]["evaluations"] = merge_variant_evaluations(
                    variant_results[variant_name]["evaluations"],
                    evaluations,
                )
        except Exception as exc:
            logging.error(
                "Error evaluating variants for question '%s...': %s",
                record.get("question", "")[:30],
                exc,
            )

    finalized_results = {}
    for variant_name, payload in variant_results.items():
        evaluations = payload.get("evaluations", [])
        if not evaluations:
            continue
        finalized_results[variant_name] = summarize_evaluations(evaluations, len(yes_no_questions))

    if not finalized_results:
        logging.info("No finalized variant evaluations for question: '%s'", record.get("question", "")[:80])
        return None

    record["variant_evaluations"] = finalized_results
    return record


def process_variant_file(file_info: tuple) -> int:
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

    def process_single_line(line: str) -> dict | None:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return None

        question_text = record.get("question", "")
        if not question_text or normalize_key(question_text) in seen_questions:
            logging.debug("Skipping question (empty or already seen): '%s'", question_text[:80])
            return None

        processed = process_variant_record(record, config)
        if processed is None:
            # Ensure we still checkpoint this question so we don't retry repeatedly
            # Write a placeholder indicating evaluation was attempted but produced
            # no finalized results (e.g., provider returned no usable JSON).
            logging.info("Evaluation produced no finalized results for question '%s' - writing placeholder and continuing", question_text[:80])
            record["variant_evaluations"] = {}
            record["_evaluation_status"] = "skipped_no_finalized_results"
            return record

        return processed

    filename_base = os.path.basename(input_path)
    for record in tqdm((process_single_line(line) for line in lines), total=len(lines), desc=filename_base, leave=False):
        if record is not None:
            with open(output_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_count += 1
            logging.info("Wrote evaluated record for question '%s' to %s", record.get("question", "")[:80], output_path)
            seen_questions.add(normalize_key(record["question"]))

    return results_count


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_dir = os.getenv("GENERATED_VARIANTS_DIR", "data/generated_variants")
    output_dir = os.getenv("GENERATED_VARIANT_EVALS_DIR", "data/generated_variant_evaluations")

    config = {
        "provider": "ollama",
        "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "model": os.getenv("QA_MODEL_NAME", "qwen3.6:latest"),
        "api_key": os.getenv("OLLAMA_API_KEY"),
        "workers": 1,
        "request_timeout": max(60, int(os.getenv("QA_REQUEST_TIMEOUT", "900"))),
    }

    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    variant_files = [f for f in os.listdir(input_dir) if f.endswith("_variants.jsonl")]
    if not variant_files:
        print(f"No variant files found in {input_dir}.")
        return

    print(f"Found {len(variant_files)} variant files. Starting evaluation...")
    print(f"Using model: {config['model']} with {config['workers']} workers.")

    tasks = [
        (
            os.path.join(input_dir, filename),
            os.path.join(output_dir, filename.replace("_variants.jsonl", "_evaluated.jsonl")),
            config,
        )
        for filename in variant_files
    ]

    total_processed = 0
    for task in tqdm(tasks, total=len(tasks), desc="Overall Progress"):
        try:
            total_processed += process_variant_file(task)
        except Exception as exc:
            logging.error("Task failed: %s", exc)

    print(f"\nSuccessfully processed {total_processed} variant records.")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
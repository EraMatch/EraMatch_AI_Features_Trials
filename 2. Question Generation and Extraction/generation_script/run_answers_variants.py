import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Setup paths
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def normalize_key(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s?]", "", normalized)
    return normalized.strip()

def get_variants_generation_prompt(context: str, question: str, reference_answer: str) -> str:
    return f"""
You are a synthetic data generator. Your goal is to create a dataset of varying answer qualities for a technical assessment.

[CONTEXT]
{context}

[QUESTION]
{question}

[GOLD STANDARD REFERENCE]
{reference_answer}

---
TASK:
Generate exactly 5 variants of an answer to the question above. Each variant must follow these strict quality guidelines:

1. "excellent": A top-tier answer that is accurate, clear, and follows the gold standard.
2. "average": A passing but mediocre answer. It is correct but misses 1 or 2 key nuances or details from the context.
3. "poor": A failing answer. It is either too brief, slightly off-topic, or fundamentally confuses the main concept.
4. "pedantic": A technically correct answer that is intentionally annoying. It focuses excessively on minor technicalities, uses unnecessary jargon, or is overly wordy.
5. "hallucinated": A confident-sounding answer that includes technical details, names, or facts that are NOT present in the provided context (even if they are true in the real world, they are "hallucinations" relative to this specific context).

OUTPUT FORMAT:
Return ONLY a JSON object with this structure:
{{
  "excellent": "text",
  "average": "text",
  "poor": "text",
  "pedantic": "text",
  "hallucinated": "text"
}}
"""

def extract_json_payload(raw_text: str) -> dict | None:
    try:
        return json.loads(raw_text)
    except:
        pass
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except:
            pass
    return None

def call_variant_provider(prompt: str, config: dict) -> dict | None:
    payload = {
        "model": config["model"],
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.8} # Higher temperature for creative variety
    }
    try:
        response = requests.post(f"{config['base_url']}/api/generate", json=payload, timeout=240)
        response.raise_for_status()
        return extract_json_payload(response.json().get("response", ""))
    except Exception as exc:
        logging.error(f"Provider error: {exc}")
        return None

def process_rubric_file(file_info: tuple) -> int:
    input_path, output_path, config = file_info
    
    # Checkpoint logic
    processed_questions = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed_questions.add(normalize_key(json.loads(line).get("question", "")))
                except: continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:
        logging.error(f"Could not read {input_path}: {exc}")
        return 0

    results_count = 0

    def process_line(line):
        try:
            record = json.loads(line)
        except: return None

        if normalize_key(record.get("question", "")) in processed_questions:
            return None

        prompt = get_variants_generation_prompt(
            record.get("context", ""),
            record.get("question", ""),
            record.get("answer", "") # Original gold answer
        )

        variants = call_variant_provider(prompt, config)
        if not variants:
            return None

        # Add the variants to the record
        record["answer_variants"] = variants
        return record

    filename = os.path.basename(input_path)
    with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
        for record in tqdm(executor.map(process_line, lines), total=len(lines), desc=filename, leave=False):
            if record:
                with open(output_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_count += 1
    
    return results_count

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Note: Scanning the RUBRICS directory now
    input_dir = os.getenv("GENERATED_RUBRICS_DIR", "data/generated_rubrics")
    output_dir = os.getenv("GENERATED_VARIANTS_DIR", "data/generated_variants")

    config = {
        "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "model": os.getenv("QA_MODEL_NAME", "llama3.1:8b"),
        "workers": max(1, int(os.getenv("QA_WORKER_COUNT", "4"))),
    }

    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found.")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith("_rubrics.jsonl")]
    
    tasks = [
        (
            os.path.join(input_dir, f),
            os.path.join(output_dir, f.replace("_rubrics.jsonl", "_variants.jsonl")),
            config
        ) for f in files
    ]

    total = 0
    with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
        futures = [executor.submit(process_rubric_file, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Total Progress"):
            total += future.result()

    print(f"Finished. Generated variants for {total} questions.")

if __name__ == "__main__":
    main()

# Key Improvements in this script:

# 1.  Temperature Management: I set the temperature to 0.8 for this task. While
#     rubrics require precision (low temp), generating "Poor" or "Hallucinated"
#     answers requires the LLM to be more creative and deviate from the source,
#     which works better at a higher temperature.
# 2.  Context-Aware Hallucinations: The prompt explicitly tells the LLM that a
#     hallucination is defined as "information not present in the context," which
#     is the industry-standard way to test RAG (Retrieval-Augmented Generation)
#     systems.
# 3.  Checkpointing: It checks data/generated_variants so if the script crashes or
#     you stop it, it won't re-generate (and pay for/wait for) questions you've
#     already processed.
# 4.  Preservation: It keeps your rubric_description and yes_no_questions in the
#     same JSON object, so your final dataset is "all-in-one" for the scoring
#     phase.

# What to do next:

# Once this script finishes, your JSONL records will look like this:

# {
#   "context": "...",
#   "question": "...",
#   "answer": "...",
#   "rubric_description": "...",
#   "yes_no_questions": [...],
#   "answer_variants": {
#     "excellent": "...",
#     "average": "...",
#     "poor": "...",
#     "pedantic": "...",
#     "hallucinated": "..."
#   }
# }

# You can then write your Scorer script to iterate through the 5 variants and run
# the 15 yes_no_questions against each one.

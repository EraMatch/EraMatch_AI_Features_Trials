"""Allocation-driven QA generation with controlled Ollama execution.

Distributes a requested total number of MCQs across topics and subcategories
according to available chunk counts and user-specified percentages.

This script supports only a local Ollama model pulled into Ollama.
Set the model via `QA_MODEL_NAME` or `OLLAMA_MODEL_NAME`.

This script calls `src.qa_generator.generator.generate_qa_with_provider` directly and
appends QA records to JSONL files under `data/generated_qa/`.

Usage: set environment vars as needed (OLLAMA_URL and QA_MODEL_NAME / OLLAMA_MODEL_NAME)
then run this script. It will run until the target MCQs are produced
(may exceed target slightly due to per-chunk rounding).

Note: Use `QA_WORKER_COUNT=1` for safe sequential execution. Set a higher value only
if your Ollama server/model can handle concurrent requests.
"""

import os
import sys
import glob
import json
import math
import random
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.qa_generator.generator import generate_qa_with_provider
from src.qa_generator.scenarios import TOTAL_MASKS, get_mask
from src.qa_generator.prompts import NUM_SCENARIOS_PER_CHUNK as DEFAULT_SCENARIOS


def discover_chunks(chunks_dir: str) -> dict:
    """Return mapping topic->subcategory->list_of_chunk_paths."""
    mapping = {}
    for topic_path in sorted(glob.glob(os.path.join(chunks_dir, "*"))):
        if not os.path.isdir(topic_path):
            continue
        topic = os.path.basename(topic_path)
        mapping.setdefault(topic, {})
        for sub_path in sorted(glob.glob(os.path.join(topic_path, "*"))):
            if not os.path.isdir(sub_path):
                continue
            sub = os.path.basename(sub_path)
            chunk_files = sorted(glob.glob(os.path.join(sub_path, "**", "chunk_*.txt"), recursive=True))
            mapping[topic][sub] = chunk_files
    return mapping


def write_mcq_record(output_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_question_text(question: str) -> str:
    """Normalize a question so duplicates can be detected across providers and reruns."""
    normalized = question.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[\u2018\u2019'\"`]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s?]", "", normalized)
    return normalized.strip()


def load_existing_question_keys(path: str) -> set[str]:
    """Load already-written question keys from an existing JSONL file."""
    keys: set[str] = set()
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
                keys.add(normalize_question_text(question))
    return keys


def build_provider_jobs() -> list[dict]:
    """Build the set of QA backends to run for each prompt.
    
    Architecture:
    - Ollama (single local model): Handles all topics.
    """
    providers: list[dict] = []
    
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("QA_MODEL_NAME", os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b"))

    # Provider 1: Ollama (single local model)
    providers.append({
        "name": "ollama-local",
        "provider": "ollama",
        "base_url": ollama_url,
        "model_name": ollama_model,
        "api_key": os.getenv("OLLAMA_API_KEY"),
        "enabled": True,
    })

    return providers


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "429",
            "rate limit",
            "quota exceeded",
            "resource_exhausted",
            "too many requests",
        )
    )

def is_valid_mcq(record: dict) -> bool:
    """Rigorous real-life domain validation for MCQ generated records."""
    scenario = str(record.get("scenario_context", "")).strip()
    question = str(record.get("question", "")).strip()
    
    if len(scenario) < 10 or len(question) < 5:
        return False
        
    text_lower = (scenario + " " + question).lower()
    if any(phrase in text_lower for phrase in ["as an ai", "i cannot", "i'm sorry"]):
        return False
    
    # Extract choices securely
    choices = []
    if "choices" in record and isinstance(record["choices"], list):
        choices = [str(c).strip() for c in record["choices"]]
    elif "choices/0" in record:
        choices = [str(record.get(f"choices/{i}", "")).strip() for i in range(4)]
    else:
        choices = [str(record.get(f"choice_{c}", "")).strip() for c in ["A", "B", "C", "D"]]
        
    if len(choices) != 4 or any(not c for c in choices):
        return False
        
    # 1. Logic: Choices must be distinct (prevent A and B from being exactly the same)
    if len(set([c.lower() for c in choices])) < 4:
        return False
        
    ans = str(record.get("correct_choice", record.get("correct_answer", record.get("answer", "")))).strip()
    if not ans: return False
    
    # 2. Logic: The question shouldn't trivially "leak" the answer string inside of it
    # E.g., Question: "What is an Integer?" Correct: "Integer"
    if ans.lower() in question.lower() and len(ans) > 4:
        # Give some leeway if it's a long sentence, but if the answer is the exact subject, fail it
        if question.lower().endswith(f"{ans.lower()}?"):
            return False

    return True

def is_valid_essay(record: dict) -> bool:
    """Rigorous real-life domain validation for Essay generated records."""
    scenario = str(record.get("scenario_context", "")).strip()
    question = str(record.get("question", "")).strip()
    answer = str(record.get("model_answer", "")).strip()
    
    if len(scenario) < 10 or len(question) < 5 or len(answer) < 10:
        return False
        
    # 1. AI Refusals
    if any(phrase in answer.lower() for phrase in ["as an ai", "i cannot", "i'm sorry", "i don't have personal"]):
        return False
        
    # 2. Logic: Answer shouldn't just repeat the question exactly
    if answer.lower() == question.lower() or question.lower() in answer.lower()[:len(question) + 10]:
        return False
        
    return True


def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunks_dir = os.getenv("CHUNKS_DIR", "data/chunks")
    output_dir = os.getenv("GENERATED_QA_DIR", "data/generated_qa")
    temperature = float(os.getenv("QA_TEMPERATURE", "0.7"))

    provider_jobs = build_provider_jobs()
    if not provider_jobs:
        print("No QA provider configured. Set OLLAMA_URL and QA_MODEL_NAME / OLLAMA_MODEL_NAME.")
        return

    target_total_mcqs = int(os.getenv("TARGET_TOTAL_MCQS", "200"))
    max_workers = max(1, int(os.getenv("QA_WORKER_COUNT", "1")))

    # User-specified internal percentages per topic (as fractions)
    # Keys correspond to topic folder names in data/chunks
    percentages = {
        "ML_DS": {"Abstract_Theory": 0.20, "Real_Wolrd_Scenarios": 0.30, "Applied_Enginering": 0.50},
        "DSA": {"Abstract_Theory": 0.20, "Applied_Enginering": 0.50, "Real_Wolrd_Scenarios": 0.30},
        "SWE": {"Abstract_Theory": 0.20, "Applied_Enginering": 0.50, "Real_Wolrd_Scenarios": 0.30},
    }

    mapping = discover_chunks(chunks_dir)
    # Count chunks per topic/sub
    counts = {}
    total_chunks = 0
    for topic, subs in mapping.items():
        counts[topic] = {}
        for sub, files in subs.items():
            counts[topic][sub] = len(files)
            total_chunks += len(files)

    if total_chunks == 0:
        print(f"No chunks found in {chunks_dir}")
        return

    # Allocate per-topic targets proportional to available chunks
    topic_chunk_totals = {t: sum(v.values()) for t, v in counts.items()}
    topic_targets = {}
    for topic, c in topic_chunk_totals.items():
        share = c / total_chunks
        topic_targets[topic] = int(round(share * target_total_mcqs))

    # Now split within topic by requested percentages
    sub_targets = {}
    for topic, target in topic_targets.items():
        sub_targets[topic] = {}
        topic_perc = percentages.get(topic, {})
        # If topic not in percentages, split evenly across its subs
        if not topic_perc:
            subs = list(counts.get(topic, {}).keys())
            for s in subs:
                sub_targets[topic][s] = int(round(target / max(1, len(subs))))
            continue
        for sub, perc in topic_perc.items():
            sub_targets[topic][sub] = int(round(target * perc))

    # Compute per-chunk calls needed (scenarios per chunk)
    per_chunk_calls = {}
    for topic, subs in sub_targets.items():
        per_chunk_calls[topic] = {}
        for sub, tgt in subs.items():
            n_chunks = counts.get(topic, {}).get(sub, 0)
            if n_chunks == 0:
                per_chunk_calls[topic][sub] = 0
            else:
                per_chunk_calls[topic][sub] = int(math.ceil(tgt / n_chunks))

    # Print plan summary
    print("Generation plan summary:")
    print(f"Total chunks: {total_chunks}")
    for topic in sorted(mapping.keys()):
        print(f"- Topic {topic}: chunks={topic_chunk_totals.get(topic,0)} target_mcqs={topic_targets.get(topic,0)}")
        for sub in sorted(mapping.get(topic, {}).keys()):
            print(f"    {sub}: chunks={counts[topic].get(sub,0)} target_mcqs={sub_targets.get(topic,{}).get(sub,0)} calls_per_chunk={per_chunk_calls[topic].get(sub,0)}")

    topic_names = sorted(mapping.keys())  # [DSA, ML_DS, SWE]
    active_jobs = [job for job in provider_jobs if job.get("enabled", True)]
    if not active_jobs:
        print("No active providers found.")
        return

    # Assign the first active provider to all topics
    single_provider = active_jobs[0]
    topic_provider_map = {
        topic: single_provider
        for topic in topic_names
    }
    
    print("\nProvider assignment (sequential execution):")
    for topic, job in topic_provider_map.items():
        print(f"  {topic:8s} → {job['name']:20s} ({job['provider']})")

    def process_topic(topic: str, assigned_job: dict) -> tuple[int, int]:
        total_mcqs_local = 0
        total_calls_local = 0
        global_mask_idx_local = 0

        processed_log_path = os.path.join(output_dir, f"processed_chunks_{topic}.txt")
        processed_chunks = set()
        if os.path.exists(processed_log_path):
            with open(processed_log_path, "r", encoding="utf-8") as f:
                processed_chunks = set(line.strip() for line in f)

        topic_total = topic_chunk_totals.get(topic, 0)
        print(f"Loaded {len(processed_chunks)} processed chunks for topic {topic} ({topic_total} chunks total).")

        subs = mapping.get(topic, {})
        for sub, chunk_files in subs.items():
            calls_per_chunk = per_chunk_calls.get(topic, {}).get(sub, 0)
            if calls_per_chunk <= 0:
                continue

            random.shuffle(chunk_files)
            out_path = os.path.join(output_dir, f"{topic}_{sub}_mcq.jsonl")
            essay_path = os.path.join(output_dir, f"{topic}_{sub}_essay.jsonl")
            seen_mcq_questions = load_existing_question_keys(out_path)
            seen_essay_questions = load_existing_question_keys(essay_path)

            for chunk_path in tqdm(chunk_files, desc=f"{topic}/{sub}", leave=False):
                if chunk_path in processed_chunks:
                    continue

                with open(chunk_path, "r", encoding="utf-8") as f:
                    context = f.read().strip()
                if not context:
                    continue

                success = False
                for _ in range(calls_per_chunk):
                    mask = get_mask(global_mask_idx_local)
                    try:
                        res = generate_qa_with_provider(
                            context,
                            mask,
                            assigned_job["provider"],
                            assigned_job["model_name"],
                            temperature,
                            assigned_job["api_key"],
                            assigned_job["base_url"],
                        )
                    except Exception as exc:
                        logging.error("%s worker failed on topic %s: %s", assigned_job["name"], topic, exc)
                        if is_rate_limit_error(exc):
                            assigned_job["enabled"] = False
                            logging.error("Disabling %s for the rest of the run due to rate/quota limits.", assigned_job["name"])
                        break

                    if res:
                        success = True

                    batch_mcq_keys: set[str] = set()
                    batch_essay_keys: set[str] = set()
                    for r in res:
                        record = dict(r)
                        record["provider"] = assigned_job["name"]
                        question_key = normalize_question_text(record.get("question", ""))
                        if not question_key:
                            continue

                        if record.get("type") == "mcq":
                            # Hard validation check
                            if not is_valid_mcq(record):
                                logging.warning("Dropped invalid MCQ record")
                                continue
                            if question_key in seen_mcq_questions or question_key in batch_mcq_keys:
                                continue
                            batch_mcq_keys.add(question_key)
                            seen_mcq_questions.add(question_key)
                            write_mcq_record(out_path, record)
                            total_mcqs_local += 1
                        elif record.get("type") == "essay":
                            # Hard validation check
                            if not is_valid_essay(record):
                                logging.warning("Dropped invalid Essay record")
                                continue
                            if question_key in seen_essay_questions or question_key in batch_essay_keys:
                                continue
                            batch_essay_keys.add(question_key)
                            seen_essay_questions.add(question_key)
                            write_mcq_record(essay_path, record)

                    global_mask_idx_local = (global_mask_idx_local + 1) % TOTAL_MASKS
                    total_calls_local += 1

                if success:
                    with open(processed_log_path, "a", encoding="utf-8") as f:
                        f.write(chunk_path + "\n")
                    processed_chunks.add(chunk_path)

        return total_mcqs_local, total_calls_local

    # Use a configurable worker count. Default is 1 for safe sequential execution.
    print(f"Using Ollama model: {single_provider['model_name']}")
    print(f"Using max worker threads: {max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_topic = {
            executor.submit(process_topic, topic, job): topic
            for topic, job in topic_provider_map.items()
        }

        total_mcqs = 0
        total_calls = 0
        for future in as_completed(future_to_topic):
            topic = future_to_topic[future]
            try:
                topic_mcqs, topic_calls = future.result()
            except Exception as exc:
                logging.error("Topic worker failed for %s: %s", topic, exc)
                continue
            total_mcqs += topic_mcqs
            total_calls += topic_calls

    print(f"Finished generation pass. Total MCQs: {total_mcqs} (calls: {total_calls})")


if __name__ == "__main__":
    main()


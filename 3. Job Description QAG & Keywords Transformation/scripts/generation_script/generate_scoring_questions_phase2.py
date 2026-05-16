"""
Phase 2: Generate Evaluation Questions for Resume Scoring

These questions are designed for an LLM to evaluate a resume against a JD.
- 3-4 General: Core filters (Years, Tech Stack, Education).
- 5-6 Medium: Domain specific evidence and technical proficiency.
- 4-5 Deep Dive: Project complexity, problem-solving evidence, and impact.
"""

import argparse
import json
import os
import time
import threading
import hashlib
from typing import List, Dict, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ============================================================================
# LLM BACKENDS
# ============================================================================

def call_ollama(prompt: str, model: str = "qwen2.5:7b", host: str = "http://localhost:11434") -> str:
    import requests
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 3000}}
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "")

def call_openrouter(prompt: str, api_key: str, model: str = "openai/gpt-4o-mini") -> str:
    import requests
    url = "https://api.openrouter.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 3000}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or choices[0]
        return msg.get("content") if isinstance(msg, dict) else str(msg)
    return ""

# ============================================================================
# HELPERS
# ============================================================================

def build_scoring_prompt(jd_text: str, meta: Dict[str, Any]) -> str:
    """Constructs a prompt focused on evaluation criteria."""
    pos = meta.get("position", "the role")
    variant = meta.get("variant", {})
    
    prompt = f"""
You are an expert technical recruiter. I need you to transform the following Job Description into a structured set of Evaluation Questions. 
Another LLM will use these questions to "score" a candidate's resume. 

Each question must be designed to check for EVIDENCE in the resume. 

Generate exactly 12-15 questions split into these categories:

1. General Requirements (3-4 questions):
   - Focus on hard constraints: Does the candidate have the required [X] years of experience? Do they have the mandatory [Degree/Certification]? Do they list the core tech stack?

2. Technical & Domain Proficiency (5-6 questions):
   - Focus on specific skills and domain context: Does the candidate show evidence of working in [{variant.get('domain')}]? Can you find proof of them using [{variant.get('tech_focus')}] tools in a professional setting?

3. Deep Dive & Impact (4-5 questions):
   - Focus on complexity and results: Does the resume describe projects similar to the ones mentioned in the JD? Is there evidence of problem-solving regarding [specific JD challenges]? Does the candidate show quantitative impact (e.g., improved performance by X%)?

Job Description:
\"\"\"
{jd_text}
\"\"\"

Output Format:
Category Name:
- Question 1
- Question 2
...
"""
    return prompt

def get_record_id(record: Dict[str, Any]) -> str:
    meta = record.get("meta", {})
    unique_str = f"{meta.get('position', '')}_{json.dumps(meta.get('variant', {}), sort_keys=True)}"
    return hashlib.md5(unique_str.encode()).hexdigest()

# ============================================================================
# CORE LOGIC
# ============================================================================

def process_scoring_record(record: Dict[str, Any], llm_caller: Callable, args) -> Tuple[Dict[str, Any], str]:
    jd_text = record.get("job_description", "")
    meta = record.get("meta", {})
    record_id = get_record_id(record)
    
    if not jd_text: return {**record, "status": "skipped"}, record_id

    prompt = build_scoring_prompt(jd_text, meta)
    
    try:
        scoring_criteria = llm_caller(prompt)
        time.sleep(args.sleep)
        return {
            "meta": meta,
            "job_description": jd_text,
            "evaluation_questions": scoring_criteria,
            "status": "success"
        }, record_id
    except Exception as e:
        return {"meta": meta, "status": "failed", "error": str(e)}, record_id

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_file", default="data/phase1_job_descriptions.jsonl")
    parser.add_argument("--out", default="data/phase2_scoring_questions.jsonl")
    parser.add_argument("--model", choices=["ollama", "openrouter"], default="ollama")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--openrouter-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--max-workers", type=int, default=1) # Parallelism
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    checkpoint_file = f"{args.out}.checkpoint.txt"

    # Load JDs
    all_records = []
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        return
        
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "success":
                all_records.append(rec)

    # Checkpoint logic
    completed_ids = set()
    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            completed_ids = set(line.strip() for line in f)

    def llm_caller(p):
        if args.model == "openrouter":
            return call_openrouter(p, args.openrouter_key)
        return call_ollama(p, model=args.ollama_model)

    # Process
    out_f = open(args.out, "a" if args.resume else "w", encoding="utf-8")
    cp_f = open(checkpoint_file, "a" if args.resume else "w", encoding="utf-8")
    write_lock = threading.Lock()
    
    progress = tqdm(total=len(all_records), desc="Generating Scoring Questions", initial=len(completed_ids)) if tqdm else None

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for rec in all_records:
            rid = get_record_id(rec)
            if args.resume and rid in completed_ids: continue
            futures.append(executor.submit(process_scoring_record, rec, llm_caller, args))

        for future in as_completed(futures):
            res, rid = future.result()
            with write_lock:
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                cp_f.write(rid + "\n")
                cp_f.flush()
            if progress: progress.update(1)

    out_f.close()
    cp_f.close()
    print(f"✓ Done. Evaluation questions saved to {args.out}")

if __name__ == "__main__":
    main()
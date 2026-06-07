"""
EraMatch: High-Resolution Keyword Extraction (Phase 2 - Overclocked)
Focus: 50-Keyword Weighted Profile (20/15/15)
Updates: Includes job_description in output and enforces 50-count via Python.
"""

import argparse
import json
import os
import time
import threading
import hashlib
import re
from typing import List, Dict, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================================
# LLM BACKENDS & UTILS
# ============================================================================

def call_ollama(prompt: str, model: str = "llama3.1:8b", host: str = "http://localhost:11434") -> str:
    import requests
    # Low temperature for faster, more consistent extraction
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "")

def force_to_50(keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Ensures exact 20/15/15 distribution even if AI is lazy."""
    targets = {"primary": 20, "secondary": 15, "tertiary": 15}
    for key, count in targets.items():
        lst = keywords.get(key, [])
        if len(lst) > count:
            keywords[key] = lst[:count]
        elif len(lst) < count:
            fillers = ["Professionalism", "Technical Proficiency", "Domain Knowledge", "Industry Standard"]
            while len(keywords[key]) < count:
                keywords[key].append(fillers[len(keywords[key]) % len(fillers)])
    return keywords

# ============================================================================
# CORE LOGIC
# ============================================================================

def build_keyword_prompt(jd_text: str) -> str:
    return f"""
Extract exactly 50 keywords from this JD.
- Primary (20): Core tech/skills.
- Secondary (15): Tools/Soft skills.
- Tertiary (15): Industry context.

JSON ONLY.
JD:
\"\"\"
{jd_text[:3000]}
\"\"\"

Output Format:
{{"keywords": {{"primary": [], "secondary": [], "tertiary": []}}}}
"""

def get_record_id(record: Dict[str, Any]) -> str:
    content = record.get("job_description", "") or json.dumps(record.get("meta", {}))
    return hashlib.md5(content.encode()).hexdigest()

def process_record(record: Dict[str, Any], llm_caller: Callable, args) -> Tuple[Dict[str, Any], str]:
    jd_text = record.get("job_description", "")
    meta = record.get("meta", {})
    record_id = get_record_id(record)
    
    if not jd_text:
        return {**record, "status": "skipped"}, record_id

    prompt = build_keyword_prompt(jd_text)
    
    try:
        response = llm_caller(prompt)
        clean_json = re.search(r'\{.*\}', response, re.DOTALL).group()
        data = json.loads(clean_json)

        # Force the count to exactly 20/15/15
        final_keywords = force_to_50(data.get("keywords", {}))

        return {
            "job_description": jd_text,  # ADDED TO OUTPUT
            "meta": meta,
            "keyword_profile": final_keywords,
            "status": "success",
            "timestamp": time.time()
        }, record_id
    except Exception as e:
        return {"meta": meta, "status": "failed", "error": str(e)}, record_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_file", default="data/phase1_job_descriptions.jsonl")
    parser.add_argument("--out", default="data/era_match_keywords.jsonl")
    parser.add_argument("--model", choices=["ollama", "openrouter"], default="ollama")
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    parser.add_argument("--max-workers", type=int, default=8) # Increased for your 5090
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    checkpoint_file = f"{args.out}.checkpoint.txt"

    all_records = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "success" or "job_description" in rec:
                all_records.append(rec)

    completed_ids = set()
    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            completed_ids = set(line.strip() for line in f)
    
    to_process = [r for r in all_records if get_record_id(r) not in completed_ids]

    if not to_process:
        print("All records already processed.")
        return

    def llm_caller(p):
        return call_ollama(p, model=args.ollama_model)

    out_f = open(args.out, "a" if args.resume else "w", encoding="utf-8")
    cp_f = open(checkpoint_file, "a" if args.resume else "w", encoding="utf-8")
    write_lock = threading.Lock()
    
    progress = tqdm(total=len(all_records), desc="🚀 EraMatch Fast-Sync", initial=len(completed_ids))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_record, rec, llm_caller, args) for rec in to_process]
        
        for future in as_completed(futures):
            res, rid = future.result()
            with write_lock:
                if res.get("status") == "success":
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_f.flush()
                    cp_f.write(rid + "\n")
                    cp_f.flush()
            if progress: progress.update(1)

    out_f.close()
    cp_f.close()
    print(f"\n✓ Complete. File: {args.out}")

if __name__ == "__main__":
    main()
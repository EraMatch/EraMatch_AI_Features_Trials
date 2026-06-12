"""
Technical Hiring Platform — Fine-Tuning Dataset Pipeline
=========================================================
Dataset : aoxo/leetcode-problem-solutions-for-finetuning (HuggingFace)
Target  : Fine-tuned model that takes natural-language question specs
          and returns a fully structured coding interview question.

Pipeline stages
---------------
1. Load  – stream from HuggingFace datasets
2. Transform – build diverse (system, user, assistant) triples
3. Validate  – send each sample through Ollama qwen2.5-coder to verify
               the assistant turn is coherent / parseable
4. Export  – write validated JSONL ready for Axolotl / LLaMA-Factory /
             HuggingFace TRL SFT

Run
---
    pip install datasets tqdm requests
    ollama pull qwen2.5-coder:7b          # or :14b / :32b
    python hiring_dataset_pipeline.py
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Dataset ──────────────────────────────────────────────────────────────
    hf_dataset_id: str       = "aoxo/leetcode-problem-solutions-for-finetuning"
    hf_split: str            = "train"
    max_records: int         = 500          # set -1 to process the whole dataset
    shuffle_seed: int        = 42

    # ── Ollama ───────────────────────────────────────────────────────────────
    ollama_base_url: str     = "http://localhost:11434"
    ollama_model: str        = "qwen2.5-coder:7b"   # change to :14b / :32b as needed
    ollama_timeout: int      = 120          # seconds per request
    validate_with_ollama: bool = True       # set False to skip LLM validation pass
    ollama_max_tokens: int   = 1024

    # ── Output ───────────────────────────────────────────────────────────────
    output_path: str         = "hiring_dataset_qwen.jsonl"
    rejected_path: str       = "hiring_dataset_rejected.jsonl"
    stats_path: str          = "pipeline_stats.json"

    # ── Quality filters ──────────────────────────────────────────────────────
    min_problem_len: int     = 40
    min_solution_len: int    = 10
    min_test_len: int        = 10
    required_sections: list  = field(default_factory=lambda: [
        "### Problem Statement",
        "### Starter Code",
        "### Optimal Solution",
        "### Test Cases",
    ])

    # ── Prompt diversity ─────────────────────────────────────────────────────
    # Each template receives: difficulty, tags_str, lang
    user_prompt_templates: list = field(default_factory=lambda: [
        # 1 – Standard
        "Generate a complete {difficulty} interview question about {tags_str} in {lang}. "
        "Include: problem description, starter code, optimal solution, and test cases.",

        # 2 – Recruiter persona
        "As a technical recruiter screening candidates, I need a {difficulty} coding "
        "challenge covering {tags_str} ({lang}). Provide the full package: description, "
        "starter code, optimal solution, and tests. Use strict Markdown.",

        # 3 – Hiring Manager persona
        "I'm a Hiring Manager building a technical assessment. Create a {difficulty} "
        "task that evaluates {tags_str} skills in {lang}. I need a problem statement, "
        "starting function signature, complete working solution, and assertions.",

        # 4 – Candidate practice
        "Give me a {difficulty} mock interview problem about {tags_str} in {lang}. "
        "Show the problem formulation, base code I should start with, the target "
        "solution, and test assertions.",

        # 5 – Terse / urgent
        "Quick: {difficulty} {lang} question on {tags_str}. "
        "Need problem, starter code, solution, tests.",

        # 6 – Structured JSON-style spec
        "Platform request — specs: difficulty={difficulty}, topics=[{tags_str}], "
        "language={lang}. Generate a complete interview question with all four sections.",

        # 7 – Senior engineer persona
        "As a Senior Engineer conducting technical interviews, write a {difficulty} "
        "problem testing {tags_str} ({lang}). The output must include a clear problem "
        "statement, boilerplate starter code, a well-commented optimal solution, "
        "and comprehensive test cases.",
    ])


CFG = PipelineConfig()

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers – Ollama client
# ──────────────────────────────────────────────────────────────────────────────

def ollama_health_check(cfg: PipelineConfig) -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        r = requests.get(f"{cfg.ollama_base_url}/api/tags", timeout=10)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(cfg.ollama_model in m for m in models):
            log.warning(
                "Model '%s' not found locally. Available: %s",
                cfg.ollama_model,
                models,
            )
            log.warning("Run:  ollama pull %s", cfg.ollama_model)
            return False
        log.info("Ollama OK — model '%s' ready.", cfg.ollama_model)
        return True
    except requests.RequestException as exc:
        log.warning("Ollama not reachable: %s", exc)
        return False


def ollama_chat(
    cfg: PipelineConfig,
    messages: list[dict],
    *,
    max_retries: int = 2,
) -> Optional[str]:
    """Send a chat request to Ollama; return the assistant reply or None."""
    payload = {
        "model": cfg.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": cfg.ollama_max_tokens,
            "temperature": 0.2,
        },
    }
    for attempt in range(1, max_retries + 2):
        try:
            r = requests.post(
                f"{cfg.ollama_base_url}/api/chat",
                json=payload,
                timeout=cfg.ollama_timeout,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (requests.RequestException, KeyError) as exc:
            if attempt <= max_retries:
                wait = 2 ** attempt
                log.debug("Ollama attempt %d failed (%s). Retrying in %ds…", attempt, exc, wait)
                time.sleep(wait)
            else:
                log.warning("Ollama call failed after %d attempts: %s", attempt, exc)
                return None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers – quality checks
# ──────────────────────────────────────────────────────────────────────────────

def _has_required_sections(text: str, sections: list[str]) -> bool:
    return all(sec in text for sec in sections)


def _extract_code_blocks(text: str) -> list[str]:
    """Return content of all ```python … ``` blocks."""
    return re.findall(r"```python\s*(.*?)```", text, re.DOTALL)


def quality_check_record(record: dict, cfg: PipelineConfig) -> tuple[bool, str]:
    """Return (ok, reason). Checks raw dataset fields before transformation."""
    prob = record.get("problem_description", "")
    code = record.get("completion", "")
    test = record.get("test", "")

    if len(prob.strip()) < cfg.min_problem_len:
        return False, f"problem_description too short ({len(prob)} chars)"
    if len(code.strip()) < cfg.min_solution_len:
        return False, f"completion too short ({len(code)} chars)"
    if len(test.strip()) < cfg.min_test_len:
        return False, f"test too short ({len(test)} chars)"
    if not record.get("starter_code", "").strip():
        return False, "starter_code is empty"
    return True, "ok"


def quality_check_assistant(assistant_text: str, cfg: PipelineConfig) -> tuple[bool, str]:
    """Return (ok, reason). Checks generated assistant turn structure."""
    if not _has_required_sections(assistant_text, cfg.required_sections):
        missing = [s for s in cfg.required_sections if s not in assistant_text]
        return False, f"Missing sections: {missing}"
    blocks = _extract_code_blocks(assistant_text)
    if len(blocks) < 2:
        return False, f"Expected ≥2 code blocks, found {len(blocks)}"
    return True, "ok"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers – Ollama validation pass
# ──────────────────────────────────────────────────────────────────────────────

VALIDATION_SYSTEM = (
    "You are a code quality reviewer. "
    "You will receive a coding interview question formatted in Markdown with four sections: "
    "Problem Statement, Starter Code, Optimal Solution, and Test Cases. "
    "Respond ONLY with a JSON object and nothing else — no markdown fences, no prose:\n"
    '{"valid": true/false, "issues": ["<issue1>", ...], "fixed_solution": "<corrected Python or null>"}'
)


def ollama_validate_sample(cfg: PipelineConfig, assistant_text: str) -> dict:
    """
    Ask Ollama to review the assistant turn.
    Returns {"valid": bool, "issues": list, "fixed_solution": str|None}
    """
    messages = [
        {"role": "system", "content": VALIDATION_SYSTEM},
        {"role": "user",   "content": assistant_text},
    ]
    raw = ollama_chat(cfg, messages)
    if raw is None:
        return {"valid": False, "issues": ["ollama_unreachable"], "fixed_solution": None}
    try:
        # Strip accidental markdown fences
        clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
        result = json.loads(clean)
        result.setdefault("valid", False)
        result.setdefault("issues", [])
        result.setdefault("fixed_solution", None)
        return result
    except json.JSONDecodeError:
        log.debug("Ollama validation JSON parse failed. Raw: %s", raw[:200])
        return {"valid": False, "issues": ["json_parse_error"], "fixed_solution": None}


# ──────────────────────────────────────────────────────────────────────────────
# Core – transform a single dataset record
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert AI technical hiring assistant. "
    "Your role is to generate comprehensive coding interview assessments in a strictly "
    "formatted Markdown structure with exactly four sections:\n"
    "### Problem Statement\n"
    "### Starter Code\n"
    "### Optimal Solution\n"
    "### Test Cases\n"
    "Always include Python code inside ```python … ``` fences."
)


def build_assistant_turn(record: dict) -> str:
    return (
        f"### Problem Statement\n{record.get('problem_description', '').strip()}\n\n"
        f"### Starter Code\n```python\n{record.get('starter_code', '').strip()}\n```\n\n"
        f"### Optimal Solution\n```python\n{record.get('completion', '').strip()}\n```\n\n"
        f"### Test Cases\n```python\n{record.get('test', '').strip()}\n```"
    )


def build_user_turn(record: dict, cfg: PipelineConfig) -> str:
    difficulty  = record.get("difficulty", "Medium")
    tags        = record.get("tags", [])
    tags_str    = ", ".join(tags) if tags else "general algorithmic concepts"
    lang        = "Python"
    template    = random.choice(cfg.user_prompt_templates)
    return template.format(difficulty=difficulty, tags_str=tags_str, lang=lang)


def transform_record(record: dict, cfg: PipelineConfig) -> dict:
    """Build a ChatML message triple from a raw dataset record."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": build_user_turn(record, cfg)},
            {"role": "assistant", "content": build_assistant_turn(record)},
        ],
        "_meta": {
            "task_id":    record.get("task_id", ""),
            "difficulty": record.get("difficulty", ""),
            "tags":       record.get("tags", []),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core – full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig) -> None:
    log.info("=" * 60)
    log.info("Technical Hiring Dataset Pipeline")
    log.info("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.error("'datasets' not installed. Run:  pip install datasets")
        sys.exit(1)

    log.info("Loading dataset '%s' (split=%s) …", cfg.hf_dataset_id, cfg.hf_split)
    ds = load_dataset(cfg.hf_dataset_id, split=cfg.hf_split)
    log.info("Dataset loaded — %d records total.", len(ds))

    if cfg.shuffle_seed >= 0:
        ds = ds.shuffle(seed=cfg.shuffle_seed)
    if cfg.max_records > 0:
        ds = ds.select(range(min(cfg.max_records, len(ds))))
    log.info("Processing %d records.", len(ds))

    # ── 2. Ollama health check ───────────────────────────────────
    ollama_ok = ollama_health_check(cfg) if cfg.validate_with_ollama else False
    if cfg.validate_with_ollama and not ollama_ok:
        log.warning(
            "Ollama unavailable — running WITHOUT LLM validation. "
            "Start Ollama and re-run to enable full validation."
        )

    # ── 3. Process records ───────────────────────────────────────
    out_path      = Path(cfg.output_path)
    rejected_path = Path(cfg.rejected_path)

    stats = {
        "total": len(ds),
        "accepted": 0,
        "rejected_quality": 0,
        "rejected_structure": 0,
        "rejected_ollama": 0,
        "ollama_fixes_applied": 0,
        "skipped_ollama_unavailable": 0,
    }

    with out_path.open("w", encoding="utf-8") as f_out, \
         rejected_path.open("w", encoding="utf-8") as f_rej:

        for record in tqdm(ds, desc="Processing", unit="rec"):
            record = dict(record)

            # ── 3a. Raw quality gate ─────────────────────────────
            ok, reason = quality_check_record(record, cfg)
            if not ok:
                stats["rejected_quality"] += 1
                f_rej.write(json.dumps({"reason": reason, "task_id": record.get("task_id")}) + "\n")
                continue

            # ── 3b. Build ChatML triple ──────────────────────────
            sample    = transform_record(record, cfg)
            assistant = sample["messages"][2]["content"]

            # ── 3c. Structural check ─────────────────────────────
            ok, reason = quality_check_assistant(assistant, cfg)
            if not ok:
                stats["rejected_structure"] += 1
                f_rej.write(json.dumps({"reason": reason, "task_id": record.get("task_id")}) + "\n")
                continue

            # ── 3d. Ollama validation ────────────────────────────
            if ollama_ok:
                vresult = ollama_validate_sample(cfg, assistant)

                if not vresult["valid"]:
                    # If Ollama proposed a fix, try to use it
                    fixed = vresult.get("fixed_solution")
                    if fixed and len(fixed.strip()) > cfg.min_solution_len:
                        # Rebuild the assistant turn with the fixed solution
                        record["completion"] = fixed
                        sample    = transform_record(record, cfg)
                        assistant = sample["messages"][2]["content"]
                        ok2, _    = quality_check_assistant(assistant, cfg)
                        if ok2:
                            sample["messages"][2]["content"] = assistant
                            stats["ollama_fixes_applied"] += 1
                            log.debug("Ollama fix applied for task_id=%s", record.get("task_id"))
                        else:
                            stats["rejected_ollama"] += 1
                            f_rej.write(
                                json.dumps({
                                    "reason": "ollama_invalid",
                                    "issues": vresult["issues"],
                                    "task_id": record.get("task_id"),
                                }) + "\n"
                            )
                            continue
                    else:
                        stats["rejected_ollama"] += 1
                        f_rej.write(
                            json.dumps({
                                "reason": "ollama_invalid",
                                "issues": vresult["issues"],
                                "task_id": record.get("task_id"),
                            }) + "\n"
                        )
                        continue
            else:
                stats["skipped_ollama_unavailable"] += 1

            # ── 3e. Write accepted sample ────────────────────────
            # Strip _meta before writing (training data shouldn't have it)
            output_sample = {"messages": sample["messages"]}
            f_out.write(json.dumps(output_sample, ensure_ascii=False) + "\n")
            stats["accepted"] += 1

    # ── 4. Write stats ───────────────────────────────────────────
    Path(cfg.stats_path).write_text(json.dumps(stats, indent=2))

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("  Accepted  : %d", stats["accepted"])
    log.info("  Rejected  : %d (quality) + %d (structure) + %d (ollama)",
             stats["rejected_quality"],
             stats["rejected_structure"],
             stats["rejected_ollama"])
    log.info("  Ollama fixes applied : %d", stats["ollama_fixes_applied"])
    log.info("  Output  → %s", cfg.output_path)
    log.info("  Rejected→ %s", cfg.rejected_path)
    log.info("  Stats   → %s", cfg.stats_path)
    log.info("=" * 60)


def demo() -> None:
    """Run the pipeline on 3 hard-coded records to verify everything works."""
    log.info("Running DEMO mode (no HuggingFace download required).")

    sample_records = [
        {
            "task_id": "two-sum",
            "difficulty": "Easy",
            "tags": ["Array", "Hash Table"],
            "problem_description": (
                "Given an array of integers nums and an integer target, return indices "
                "of the two numbers such that they add up to target. "
                "You may assume that each input would have exactly one solution, "
                "and you may not use the same element twice."
            ),
            "starter_code": (
                "class Solution:\n"
                "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
                "        pass"
            ),
            "completion": (
                "class Solution:\n"
                "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
                "        d = {}\n"
                "        for i, x in enumerate(nums):\n"
                "            if (y := target - x) in d:\n"
                "                return [d[y], i]\n"
                "            d[x] = i"
            ),
            "test": (
                "def check(candidate):\n"
                "    assert candidate(nums=[2,7,11,15], target=9) == [0, 1]\n"
                "    assert candidate(nums=[3,2,4], target=6) == [1, 2]\n"
                "    assert candidate(nums=[3,3], target=6) == [0, 1]"
            ),
        },
        {
            "task_id": "valid-parentheses",
            "difficulty": "Easy",
            "tags": ["String", "Stack"],
            "problem_description": (
                "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', "
                "determine if the input string is valid. An input string is valid if: "
                "open brackets must be closed by the same type of brackets, "
                "and open brackets must be closed in the correct order."
            ),
            "starter_code": (
                "class Solution:\n"
                "    def isValid(self, s: str) -> bool:\n"
                "        pass"
            ),
            "completion": (
                "class Solution:\n"
                "    def isValid(self, s: str) -> bool:\n"
                "        stack = []\n"
                "        mapping = {')': '(', '}': '{', ']': '['}\n"
                "        for char in s:\n"
                "            if char in mapping:\n"
                "                top = stack.pop() if stack else '#'\n"
                "                if mapping[char] != top:\n"
                "                    return False\n"
                "            else:\n"
                "                stack.append(char)\n"
                "        return not stack"
            ),
            "test": (
                "def check(candidate):\n"
                "    assert candidate('()') == True\n"
                "    assert candidate('()[]{}') == True\n"
                "    assert candidate('(]') == False\n"
                "    assert candidate('([)]') == False\n"
                "    assert candidate('{[]}') == True"
            ),
        },
        {
            "task_id": "binary-search",
            "difficulty": "Easy",
            "tags": ["Array", "Binary Search"],
            "problem_description": (
                "Given an array of integers nums which is sorted in ascending order, "
                "and an integer target, write a function to search target in nums. "
                "If target exists, then return its index. Otherwise, return -1. "
                "You must write an algorithm with O(log n) runtime complexity."
            ),
            "starter_code": (
                "class Solution:\n"
                "    def search(self, nums: List[int], target: int) -> int:\n"
                "        pass"
            ),
            "completion": (
                "class Solution:\n"
                "    def search(self, nums: List[int], target: int) -> int:\n"
                "        lo, hi = 0, len(nums) - 1\n"
                "        while lo <= hi:\n"
                "            mid = (lo + hi) // 2\n"
                "            if nums[mid] == target:\n"
                "                return mid\n"
                "            elif nums[mid] < target:\n"
                "                lo = mid + 1\n"
                "            else:\n"
                "                hi = mid - 1\n"
                "        return -1"
            ),
            "test": (
                "def check(candidate):\n"
                "    assert candidate([-1,0,3,5,9,12], 9) == 4\n"
                "    assert candidate([-1,0,3,5,9,12], 2) == -1\n"
                "    assert candidate([5], 5) == 0\n"
                "    assert candidate([5], 3) == -1"
            ),
        },
    ]

    cfg = PipelineConfig(
        max_records=-1,
        validate_with_ollama=True,
        output_path="demo_output.jsonl",
        rejected_path="demo_rejected.jsonl",
        stats_path="demo_stats.json",
    )

    ollama_ok = ollama_health_check(cfg)

    accepted, rejected = [], []
    for record in sample_records:
        ok, reason = quality_check_record(record, cfg)
        if not ok:
            log.warning("DEMO record '%s' failed quality: %s", record["task_id"], reason)
            rejected.append(record["task_id"])
            continue

        sample    = transform_record(record, cfg)
        assistant = sample["messages"][2]["content"]

        ok, reason = quality_check_assistant(assistant, cfg)
        if not ok:
            log.warning("DEMO record '%s' failed structure: %s", record["task_id"], reason)
            rejected.append(record["task_id"])
            continue

        if ollama_ok:
            vresult = ollama_validate_sample(cfg, assistant)
            log.info(
                "DEMO Ollama validation for '%s': valid=%s issues=%s",
                record["task_id"],
                vresult["valid"],
                vresult["issues"],
            )
            if not vresult["valid"] and not vresult.get("fixed_solution"):
                rejected.append(record["task_id"])
                continue

        accepted.append(record["task_id"])
        # Pretty-print the first sample
        if len(accepted) == 1:
            log.info("\n── Sample output (task_id=%s) ──", record["task_id"])
            print(json.dumps({"messages": sample["messages"]}, indent=2, ensure_ascii=False))

    log.info("\nDEMO results: accepted=%s  rejected=%s", accepted, rejected)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hiring Dataset Pipeline")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with 3 built-in records (no HuggingFace download).",
    )
    parser.add_argument("--max-records", type=int, default=500,
                        help="Max records to process from the dataset (default 500, -1 = all).")
    parser.add_argument("--model",       type=str, default=CFG.ollama_model,
                        help="Ollama model tag (e.g. qwen2.5-coder:14b).")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip Ollama validation pass.")
    parser.add_argument("--output",      type=str, default=CFG.output_path,
                        help="Output JSONL path.")
    args = parser.parse_args()

    CFG.max_records          = args.max_records
    CFG.ollama_model         = args.model
    CFG.validate_with_ollama = not args.no_validate
    CFG.output_path          = args.output

    if args.demo:
        demo()
    else:
        run_pipeline(CFG)

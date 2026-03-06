import os
import time
import json
import hashlib
import requests
from typing import List, Dict, Any
from collections import defaultdict

# ─── Configuration ────────────────────────────────────────────────────────────
# The EVALUATOR is a fixed model that scores all generators.
# Change this to any model you trust to be a good judge.
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "qwen3-coder-next:cloud")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Cache directory: generated questions are saved here and reused across runs
CACHE_DIR = "trials/cache"

# Fixed context — same for all models, all runs
BENCHMARK_CONTEXT = """
Machine learning (ML) is a subset of artificial intelligence that enables computers to learn
from data and make predictions or decisions without being explicitly programmed.
In the 1950s, early ideas like a computer playing checkers showed that machines could learn simple tasks.
By the 1990s, ML began showing up in practical tools like email spam filters.
The 2000s saw companies like Netflix and Amazon use ML to suggest movies or products.
Today, ML powers voice assistants, self-driving cars, and healthcare tools.
Key considerations include Bias and Fairness, where model effectiveness is tied to training data quality;
and Explainability and Transparency, which is essential in sectors like healthcare and criminal justice.
"""

GENERATION_PROMPT = """
Generate exactly 2 multiple choice questions (MCQ) and 1 essay question based ONLY on the following context.
Return a valid JSON array. Each object must have:
  - question_type: "mcq" or "essay"
  - question_text: string
  - question_config: {{ "options": [...], "evidence": "exact quote from context" }} for MCQ, {{ "evidence": "..." }} for essay
  - correct_answer: array of 0-based option indices for MCQ (e.g. [0]), or a reference answer string for essay
  - difficulty: integer 1-5
  - tags: array of 2-3 strings
  - points: 10

Context:
{context}

Return ONLY a valid JSON array. No extra text.
"""

FAITHFULNESS_PROMPT = """
Rate how well this question is grounded in the provided context. Score 0.0 to 1.0.

Context: {context}
Question: {question}
Correct Answer (text): {answer}

Return ONLY valid JSON: {{ "faithfulness_score": 0.9, "reasoning": "..." }}
"""

DISTRACTOR_PROMPT = """
Rate the quality of these MCQ distractors (incorrect options) from 1 to 10.
Criteria: plausible but clearly wrong compared to the correct answer. Avoid "all of the above".

Question: {question}
Correct: {correct}
Distractors: {distractors}

Return ONLY valid JSON: {{ "score": 7, "reasoning": "..." }}
"""

QUEST_PROMPT = """
Evaluate this question using the QUEST pedagogical framework. Return integer scores 1-5 for each dimension.

Context snippet: {context}
Question: {question}

Return ONLY valid JSON:
{{
  "quality": {{ "score": 1, "feedback": "..." }},
  "uniqueness": {{ "score": 1, "feedback": "..." }},
  "effort": {{ "score": 1, "feedback": "..." }},
  "structure": {{ "score": 1, "feedback": "..." }},
  "transparency": {{ "score": 1, "feedback": "..." }},
  "overall_pedagogy_score": 1
}}
"""


# ─── Core LLM Call ───────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str, temperature: float = 0.0) -> str:
    """
    Raw Ollama call.
    temperature=0 makes outputs DETERMINISTIC and reproducible.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": 42          # Fixed seed for reproducibility
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "")


def _safe_json(text: str):
    """Robustly parse JSON from LLM output."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            lines[1:-1] if lines and lines[-1].startswith("```") else lines[1:]
        ).strip()

    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char) + 1
        if start != -1 and end > 0:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                import re
                clean = re.sub(r',\s*([\]}])', r'\1', text[start:end])
                try:
                    return json.loads(clean)
                except Exception:
                    continue
    return None


# ─── Question Cache ───────────────────────────────────────────────────────────

def _cache_key(model_name: str) -> str:
    """Generate a stable cache key for a model + context combo."""
    content = model_name + BENCHMARK_CONTEXT + GENERATION_PROMPT
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _load_cached_questions(model_name: str):
    """Load previously generated questions from cache if they exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(model_name)
    model_safe = model_name.replace(":", "_").replace("/", "-")
    path = os.path.join(CACHE_DIR, f"{model_safe}_{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"  [CACHE HIT] Using cached questions for {model_name}")
        return data.get("questions"), data.get("latency_sec", 0)
    return None, None


def _save_cached_questions(model_name: str, questions: list, latency: float):
    """Save generated questions to cache for future reproducible runs."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(model_name)
    model_safe = model_name.replace(":", "_").replace("/", "-")
    path = os.path.join(CACHE_DIR, f"{model_safe}_{key}.json")
    with open(path, "w") as f:
        json.dump({"model": model_name, "questions": questions, "latency_sec": latency}, f, indent=2)


# ─── Evaluation (always runs fresh but deterministically) ─────────────────────

def _evaluate_question(question: Dict, context: str) -> Dict:
    """Evaluates a single question using the FIXED evaluator at temperature=0."""
    options = question.get("question_config", {}).get("options", [])
    correct_indices = question.get("correct_answer", [])
    q_text = question.get("question_text", "")
    is_mcq = question.get("question_type") == "mcq"

    if is_mcq and isinstance(correct_indices, list):
        correct_texts = [options[i] for i in correct_indices if i < len(options)]
        distractor_texts = [opt for i, opt in enumerate(options) if i not in correct_indices]
    else:
        correct_texts = [str(question.get("correct_answer", ""))]
        distractor_texts = []

    scores = {}

    # 1. Faithfulness (RAGAS-style)
    faith_raw = _call_ollama(
        FAITHFULNESS_PROMPT.format(
            context=context[:2000], question=q_text, answer=", ".join(correct_texts)
        ),
        EVALUATOR_MODEL
    )
    faith_data = _safe_json(faith_raw) or {}
    scores["ragas_faithfulness"] = faith_data.get("faithfulness_score", 0.0)
    scores["ragas_reasoning"] = faith_data.get("reasoning", "")

    # 2. Distractor quality (DeepEval-style, MCQ only)
    if is_mcq and distractor_texts:
        dist_raw = _call_ollama(
            DISTRACTOR_PROMPT.format(
                question=q_text,
                correct=json.dumps(correct_texts),
                distractors=json.dumps(distractor_texts)
            ),
            EVALUATOR_MODEL
        )
        dist_data = _safe_json(dist_raw) or {}
        scores["geval_score"] = dist_data.get("score", 0)
        scores["geval_reasoning"] = dist_data.get("reasoning", "")
    else:
        scores["geval_score"] = None
        scores["geval_reasoning"] = "N/A for essay"

    # 3. QUEST
    quest_raw = _call_ollama(
        QUEST_PROMPT.format(context=context[:1500], question=q_text),
        EVALUATOR_MODEL
    )
    scores["quest"] = _safe_json(quest_raw) or {}

    # 4. LLM-as-a-Judge (LLJ) — holistic judge verdict
    try:
        from services.eval_service import evaluate_llm_as_judge
        llj_result = evaluate_llm_as_judge(question, context)
        scores["llj"] = llj_result
        scores["llj_score"] = llj_result.get("llj_score", 0)
    except Exception as e:
        print(f"  LLJ error: {e}")
        scores["llj"] = {}
        scores["llj_score"] = 0

    return scores


# ─── Model Discovery ──────────────────────────────────────────────────────────

def get_available_ollama_models() -> List[str]:
    """Retrieves all models available in the local Ollama instance."""
    try:
        url = OLLAMA_URL.replace("/generate", "/tags")
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception as e:
        print(f"Could not fetch models from Ollama: {e}")
        return []


# ─── Single Model Benchmark ───────────────────────────────────────────────────

def run_model_benchmark(model_name: str, force_regenerate: bool = False) -> Dict[str, Any]:
    """
    Benchmarks a single model.
    - Questions are CACHED: generated once, reused across runs for reproducibility.
    - Evaluation uses temperature=0 and seed=42 for deterministic scoring.
    - force_regenerate=True clears the cache and regenerates questions.
    """
    print(f"\nBenchmarking Generator: {model_name} | Evaluator: {EVALUATOR_MODEL}")

    # Step 1: Load or generate questions
    if not force_regenerate:
        cached_questions, cached_latency = _load_cached_questions(model_name)
    else:
        cached_questions, cached_latency = None, None

    if cached_questions:
        questions = cached_questions
        latency = cached_latency
    else:
        print(f"  Generating new questions with {model_name}...")
        start = time.time()
        try:
            raw = _call_ollama(
                GENERATION_PROMPT.format(context=BENCHMARK_CONTEXT),
                model_name,
                temperature=0.0  # Deterministic generation
            )
            latency = round(time.time() - start, 2)
            questions = _safe_json(raw)
        except Exception as e:
            return {
                "model": model_name, "status": "failed",
                "reason": f"Generation error: {e}",
                "latency_sec": round(time.time() - start, 2),
                "overall_quality_score": 0, "avg_faithfulness": 0,
                "avg_distractor_quality": 0, "avg_pedagogy_quest": 0,
                "raw_questions": []
            }

        if not questions or not isinstance(questions, list) or len(questions) == 0:
            return {
                "model": model_name, "status": "failed",
                "reason": "Could not generate valid JSON questions",
                "latency_sec": latency, "overall_quality_score": 0,
                "avg_faithfulness": 0, "avg_distractor_quality": 0,
                "avg_pedagogy_quest": 0, "raw_questions": []
            }

        _save_cached_questions(model_name, questions, latency)

    # Step 2: Evaluate each question (deterministic at temp=0)
    faithfulness_scores, geval_scores, quest_scores = [], [], []

    for q in questions:
        try:
            eval_res = _evaluate_question(q, BENCHMARK_CONTEXT)
            q["eval_results"] = eval_res
            faithfulness_scores.append(eval_res.get("ragas_faithfulness", 0))
            if eval_res.get("geval_score") is not None:
                geval_scores.append(eval_res.get("geval_score", 0))
            qs = eval_res.get("quest", {}).get("overall_pedagogy_score", 0)
            if qs:
                quest_scores.append(qs)
        except Exception as e:
            print(f"  Evaluation error: {e}")

    avg_f = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    avg_g = sum(geval_scores) / len(geval_scores) if geval_scores else 0
    avg_q = sum(quest_scores) / len(quest_scores) if quest_scores else 0

    # Collect LLJ scores
    llj_scores = []
    for q in questions:
        llj = q.get("eval_results", {}).get("llj_score", 0)
        if llj:
            llj_scores.append(llj)
    avg_llj = sum(llj_scores) / len(llj_scores) if llj_scores else 0

    # Weighted composite: Faith 30%, GEval 30%, QUEST 20%, LLJ 20%
    composite = round(
        (avg_f * 10 * 0.30) +
        (avg_g * 0.30) +
        (avg_q * 2 * 0.20) +
        (avg_llj * 0.20),
        2
    )

    return {
        "model": model_name,
        "status": "success",
        "latency_sec": latency,
        "avg_faithfulness": round(avg_f, 2),
        "avg_distractor_quality": round(avg_g, 2),
        "avg_pedagogy_quest": round(avg_q, 2),
        "avg_llj_score": round(avg_llj, 2),
        "overall_quality_score": composite,
        "raw_questions": questions
    }


# ─── Full Benchmark Suite ─────────────────────────────────────────────────────

def run_full_benchmark(force_regenerate: bool = False) -> List[Dict[str, Any]]:
    """
    Runs benchmark for all discovered models.
    Each invocation creates a unique directory under trials/run_<timestamp>/.
    Results are reproducible because:
    1. Questions are cached per model (generated once, reused forever)
    2. All LLM calls use temperature=0 and seed=42
    """
    models = get_available_ollama_models()
    results = []

    # Unique directory for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("trials", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Starting benchmark for {len(models)} models.")
    print(f"Questions will be {'REGENERATED' if force_regenerate else 'CACHED (reproducible)'}")
    print(f"Evaluator: {EVALUATOR_MODEL}")
    print(f"Run directory: {run_dir}")

    # Save a run metadata file so each directory is self-documenting
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump({
            "timestamp": timestamp,
            "evaluator_model": EVALUATOR_MODEL,
            "models_tested": models,
            "force_regenerate": force_regenerate
        }, f, indent=2)

    for model in models:
        res = run_model_benchmark(model, force_regenerate=force_regenerate)
        results.append(res)
        # Save individual model result inside the run directory
        model_safe = model.replace(":", "_").replace("/", "-")
        with open(os.path.join(run_dir, f"{model_safe}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"  -> {model}: score={res.get('overall_quality_score')} status={res.get('status')}")

    # Sort: successful first by score, failures last
    results.sort(key=lambda x: (x.get("status") != "success", -x.get("overall_quality_score", 0)))

    # Save the leaderboard summary inside the run directory
    with open(os.path.join(run_dir, "leaderboard.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark complete. Results saved to {run_dir}/")
    return results


if __name__ == "__main__":
    run_full_benchmark()


# ─── N-Run Ranking Stability Analysis ────────────────────────────────────────

def run_benchmark_n_times(n: int = 20, force_regenerate_first: bool = False) -> Dict[str, Any]:
    """
    Runs the benchmark N times and aggregates ranking frequency statistics.
    
    - Questions are CACHED: generated once and reused across all N runs.
    - Only the evaluator varies slightly (stochastic), giving meaningful score ranges.
    - Produces a final ranking_stability report showing:
        * How many times each model was ranked #1, #2, etc.
        * Median composite score across all runs.
        * Score standard deviation (lower = more stable).
    """
    print(f"\n{'='*60}")
    print(f"STARTING {n}-RUN RANKING STABILITY ANALYSIS")
    print(f"Evaluator: {EVALUATOR_MODEL}")
    print(f"{'='*60}\n")

    models = get_available_ollama_models()
    if not models:
        return {"error": "No models found from Ollama"}

    # Top-level stability directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stability_dir = os.path.join("trials", f"stability_{timestamp}_{n}runs")
    os.makedirs(stability_dir, exist_ok=True)

    # Per-model accumulators
    model_scores: Dict[str, List[float]] = defaultdict(list)    # composite scores per run
    model_ranks: Dict[str, List[int]] = defaultdict(list)       # rank position per run
    rank_frequency: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))  # rank -> count

    all_run_summaries = []

    for run_idx in range(1, n + 1):
        print(f"\n--- Run {run_idx}/{n} ---")
        
        # First run may regenerate if requested, subsequent runs always use cache
        force = force_regenerate_first and run_idx == 1
        results = run_full_benchmark(force_regenerate=force)

        # Record scores and ranks for this run
        successful = [r for r in results if r.get("status") == "success"]
        for rank_pos, res in enumerate(successful, start=1):
            model = res["model"]
            score = res.get("overall_quality_score", 0)
            model_scores[model].append(score)
            model_ranks[model].append(rank_pos)
            rank_frequency[model][rank_pos] += 1

        all_run_summaries.append({
            "run": run_idx,
            "ranking": [{"model": r["model"], "score": r.get("overall_quality_score", 0)}
                        for r in successful]
        })

    # Build final aggregated stability report
    def median(lst):
        if not lst: return 0
        s = sorted(lst)
        n = len(s)
        return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

    def stdev(lst):
        if len(lst) < 2: return 0
        mean = sum(lst) / len(lst)
        return round((sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5, 3)

    stability_report = []
    for model in models:
        scores = model_scores.get(model, [])
        ranks = model_ranks.get(model, [])
        freq = rank_frequency.get(model, {})
        if not scores:
            continue
        stability_report.append({
            "model": model,
            "runs_completed": len(scores),
            "median_score": round(median(scores), 2),
            "score_stdev": stdev(scores),             # 0 = perfectly stable
            "mean_rank": round(sum(ranks) / len(ranks), 2),
            "best_rank": min(ranks),
            "worst_rank": max(ranks),
            "rank_frequency": {                       # how many times ranked #1, #2, etc.
                f"#{k}": v for k, v in sorted(freq.items())
            }
        })

    # Sort by median score descending
    stability_report.sort(key=lambda x: -x["median_score"])

    final_output = {
        "meta": {
            "total_runs": n,
            "evaluator_model": EVALUATOR_MODEL,
            "models_tested": models,
            "timestamp": timestamp
        },
        "stability_report": stability_report,
        "all_run_summaries": all_run_summaries
    }

    report_path = os.path.join(stability_dir, "stability_report.json")
    with open(report_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"STABILITY ANALYSIS COMPLETE — {n} runs finished")
    print(f"Report: {report_path}")
    print(f"\nFINAL RANKING STABILITY:")
    for i, r in enumerate(stability_report, 1):
        print(f"  #{i} {r['model']}: median={r['median_score']}, stdev={r['score_stdev']}, mean_rank={r['mean_rank']}")
    print(f"{'='*60}\n")

    return final_output

import os
import time
import json
import requests
from typing import List, Dict, Any

# --- Configuration ---
# The EVALUATOR model is fixed. It should NOT be the model being tested.
# It grades every other model's output. Using a lightweight, reliable model.
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "qwen3-coder-next:cloud")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Sample context used for benchmarking (same for all models for fair comparison)
BENCHMARK_CONTEXT = """
Machine learning (ML) is a subset of artificial intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed. 
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
Return ONLY: {{ "faithfulness_score": 0.9, "reasoning": "..." }}
"""

DISTRACTOR_PROMPT = """
Rate the quality of these MCQ distractors (incorrect options) from 1-10.
Criteria: Are they plausible but clearly wrong? Avoid "all of the above".
Question: {question}
Correct: {correct}
Distractors: {distractors}
Return ONLY: {{ "score": 7, "reasoning": "..." }}
"""

QUEST_PROMPT = """
Evaluate this question using the QUEST framework. Return scores 1-5 for each dimension.
Context snippet: {context}
Question: {question}
Return ONLY this JSON:
{{
  "quality": {{ "score": 1-5, "feedback": "..." }},
  "uniqueness": {{ "score": 1-5, "feedback": "..." }},
  "effort": {{ "score": 1-5, "feedback": "..." }},
  "structure": {{ "score": 1-5, "feedback": "..." }},
  "transparency": {{ "score": 1-5, "feedback": "..." }},
  "overall_pedagogy_score": 1-5
}}
"""


def _call_ollama(prompt: str, model: str) -> str:
    """Raw Ollama call with specific model."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "")


def _safe_json(text: str):
    """Robustly parse JSON from LLM output."""
    if not text:
        return None
    text = text.strip()
    # Strip markdown code blocks
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:]).strip()

    # Find JSON Object or Array
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


def _evaluate_question(question: Dict, context: str) -> Dict:
    """Evaluates a single question using the FIXED evaluator model."""
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

    # 1. Faithfulness
    faith_prompt = FAITHFULNESS_PROMPT.format(
        context=context[:2000], question=q_text, answer=", ".join(correct_texts)
    )
    faith_raw = _call_ollama(faith_prompt, EVALUATOR_MODEL)
    faith_data = _safe_json(faith_raw) or {}
    scores["ragas_faithfulness"] = faith_data.get("faithfulness_score", 0.0)
    scores["ragas_reasoning"] = faith_data.get("reasoning", "")

    # 2. Distractor quality (MCQ only)
    if is_mcq and distractor_texts:
        dist_prompt = DISTRACTOR_PROMPT.format(
            question=q_text, correct=correct_texts, distractors=distractor_texts
        )
        dist_raw = _call_ollama(dist_prompt, EVALUATOR_MODEL)
        dist_data = _safe_json(dist_raw) or {}
        scores["geval_score"] = dist_data.get("score", 0)
        scores["geval_reasoning"] = dist_data.get("reasoning", "")
    else:
        scores["geval_score"] = None
        scores["geval_reasoning"] = "N/A for essay"

    # 3. QUEST
    quest_prompt = QUEST_PROMPT.format(context=context[:1500], question=q_text)
    quest_raw = _call_ollama(quest_prompt, EVALUATOR_MODEL)
    quest_data = _safe_json(quest_raw) or {}
    scores["quest"] = quest_data

    return scores


def get_available_ollama_models() -> List[str]:
    """Retrieves available models from the local Ollama instance."""
    try:
        url = OLLAMA_URL.replace("/generate", "/tags")
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception as e:
        print(f"Could not fetch models from Ollama: {e}")
        return []


def run_model_benchmark(model_name: str) -> Dict[str, Any]:
    """Runs a single trial for a specific model, using a fixed evaluator."""
    print(f"\nBenchmarking Generator: {model_name}  |  Evaluator: {EVALUATOR_MODEL}")

    start_time = time.time()
    try:
        # Step 1: Generate questions with the model under test
        prompt = GENERATION_PROMPT.format(context=BENCHMARK_CONTEXT)
        raw = _call_ollama(prompt, model_name)
        latency = time.time() - start_time

        questions = _safe_json(raw)
        if not questions or not isinstance(questions, list) or len(questions) == 0:
            return {
                "model": model_name,
                "status": "failed",
                "reason": "Could not generate valid JSON questions",
                "latency_sec": round(latency, 2),
                "overall_quality_score": 0,
                "avg_faithfulness": 0,
                "avg_distractor_quality": 0,
                "avg_pedagogy_quest": 0,
                "raw_questions": []
            }

        # Step 2: Evaluate each question with the FIXED evaluator
        evaluated = []
        faithfulness_scores = []
        geval_scores = []
        quest_scores = []

        for q in questions:
            try:
                eval_res = _evaluate_question(q, BENCHMARK_CONTEXT)
                q["eval_results"] = eval_res
                faithfulness_scores.append(eval_res.get("ragas_faithfulness", 0))
                if eval_res.get("geval_score") is not None:
                    geval_scores.append(eval_res.get("geval_score", 0))
                quest_overall = eval_res.get("quest", {}).get("overall_pedagogy_score", 0)
                if quest_overall:
                    quest_scores.append(quest_overall)
                evaluated.append(q)
            except Exception as e:
                print(f"  Evaluation error for a question: {e}")
                continue

        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        avg_geval = sum(geval_scores) / len(geval_scores) if geval_scores else 0
        avg_quest = sum(quest_scores) / len(quest_scores) if quest_scores else 0

        # Composite score: weighted average (Faith 40%, GEval 40%, QUEST 20%)
        composite = round((avg_faithfulness * 10 * 0.4) + (avg_geval * 0.4) + (avg_quest * 2 * 0.2), 2)

        return {
            "model": model_name,
            "status": "success",
            "latency_sec": round(latency, 2),
            "avg_faithfulness": round(avg_faithfulness, 2),
            "avg_distractor_quality": round(avg_geval, 2),
            "avg_pedagogy_quest": round(avg_quest, 2),
            "overall_quality_score": composite,
            "raw_questions": evaluated
        }

    except Exception as e:
        return {
            "model": model_name,
            "status": "failed",
            "reason": str(e),
            "latency_sec": round(time.time() - start_time, 2),
            "overall_quality_score": 0,
            "avg_faithfulness": 0,
            "avg_distractor_quality": 0,
            "avg_pedagogy_quest": 0,
            "raw_questions": []
        }


def run_full_benchmark() -> List[Dict[str, Any]]:
    """Runs benchmarks for all discovered models and saves a log."""
    models = get_available_ollama_models()
    results = []

    print(f"Starting benchmark for {len(models)} models. Evaluator: {EVALUATOR_MODEL}")

    os.makedirs("trials", exist_ok=True)

    for model in models:
        res = run_model_benchmark(model)
        results.append(res)

        # Save incremental log per model
        model_safe = model.replace(":", "_").replace("/", "-")
        with open(f"trials/{model_safe}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"  -> {model}: score={res.get('overall_quality_score')}, status={res.get('status')}")

    # Sort: successes first by score, then failures
    results.sort(key=lambda x: (x.get("status") != "success", -x.get("overall_quality_score", 0)))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = f"trials/benchmark_leaderboard_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark complete. Leaderboard saved to {log_path}")
    return results


if __name__ == "__main__":
    run_full_benchmark()

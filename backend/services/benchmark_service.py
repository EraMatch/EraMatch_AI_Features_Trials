import os
import time
import json
import requests
from typing import List, Dict, Any
from services.ai_service import generate_questions_ai, OLLAMA_URL
from services.eval_service import run_full_evaluation

# Sample context for benchmarking
BENCHMARK_CONTEXT = """
Machine learning (ML) is a subset of artificial intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed. 
In the 1950s, early ideas like a computer playing checkers showed that machines could learn simple tasks. 
By the 1990s, ML began showing up in practical tools like email spam filters. 
The 2000s saw companies like Netflix and Amazon use ML to suggest movies or products. 
Today, ML powers voice assistants, self-driving cars, and healthcare tools.
Key considerations include Bias and Fairness, where model effectiveness is tied to training data quality; 
and Explainability and Transparency, which is essential in sectors like healthcare and criminal justice.
"""

def get_available_ollama_models() -> List[str]:
    """Retrieves a list of available models from the local Ollama instance."""
    try:
        # Ollama API tags endpoint (typically http://localhost:11434/api/tags)
        url = OLLAMA_URL.replace("/generate", "/tags")
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        return models
    except Exception as e:
        print(f"Error fetching models: {e}")
        # Secondary source: check common names if API fails (hardcoded fallback based on git logs)
        return ["llama3.2:latest", "deepseek-v3.1:671b-cloud"]

def run_model_benchmark(model_name: str) -> Dict[str, Any]:
    """Runs a single trial for a specific model."""
    print(f"\nBenchmarking Model: {model_name}")
    
    # Temporarily override the environment model for the duration of the test
    original_model = os.getenv("OLLAMA_MODEL")
    os.environ["OLLAMA_MODEL"] = model_name
    
    start_time = time.time()
    try:
        # 1. Generate Questions
        questions = generate_questions_ai(BENCHMARK_CONTEXT, mcq_count=2, essay_count=1, difficulty="Medium")
        latency = time.time() - start_time
        
        # 2. Extract Average Scores
        avg_faithfulness = 0.0
        avg_geval = 0.0
        avg_quest = 0.0
        
        if questions:
            avg_faithfulness = sum(q.get('eval_results', {}).get('ragas_faithfulness', 0.0) for q in questions) / len(questions)
            avg_geval = sum(q.get('eval_results', {}).get('geval_score', 0.0) for q in questions) / len(questions)
            # Quest overall pedagogy score (avg across all questions)
            quest_scores = [q.get('eval_results', {}).get('quest', {}).get('overall_pedagogy_score', 0) for q in questions]
            avg_quest = sum(quest_scores) / len(questions) if quest_scores else 0

        return {
            "model": model_name,
            "status": "success",
            "latency_sec": round(latency, 2),
            "avg_faithfulness": round(avg_faithfulness, 2),
            "avg_distractor_quality": round(avg_geval, 2),
            "avg_pedagogy_quest": round(avg_quest, 2),
            "overall_quality_score": round((avg_faithfulness * 10 + avg_geval + avg_quest * 2) / 3, 2),
            "raw_questions": questions
        }
    except Exception as e:
        return {
            "model": model_name,
            "status": "failed",
            "error": str(e)
        }
    finally:
        # Restore original model
        if original_model:
            os.environ["OLLAMA_MODEL"] = original_model

def run_full_benchmark() -> List[Dict[str, Any]]:
    """Runs benchmarks for all discovered models and saves a log."""
    models = get_available_ollama_models()
    results = []
    
    print(f"System: Starting benchmark for {len(models)} models.")
    
    # Ensure trials directory exists
    os.makedirs("trials", exist_ok=True)
    
    for model in models:
        res = run_model_benchmark(model)
        results.append(res)
    
    # Sort by quality score
    results.sort(key=lambda x: x.get("overall_quality_score", 0), reverse=True)
    
    # Save the leaderboard log
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = f"trials/benchmark_leaderboard_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark completed. Log saved to {log_path}")
    return results

if __name__ == "__main__":
    run_full_benchmark()

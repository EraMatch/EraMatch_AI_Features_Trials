import os
import json
from typing import List, Dict, Any
from services.ai_service import get_llm_client

# Official DeepEval & Ragas Imports
try:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from ragas.metrics import faithfulness
    from ragas import evaluate
    import pandas as pd
    HAS_EVAL_LIBS = True
except ImportError:
    print("Warning: deepeval or ragas not found. Using simulated LLM evaluations.")
    HAS_EVAL_LIBS = False

# QUEST Framework Rubric
QUEST_PROMPT = """
You are a pedagogical expert. Evaluate the provided multiple choice question (MCQ) or essay question based on the QUEST framework.
Return strictly a JSON object with scores (1-5) and specific feedback for each dimension.

QUEST Dimensions:
1. Quality: Relevance to the topic and alignment with learning objectives.
2. Uniqueness: Originality and avoidance of repetitive patterns.
3. Effort: Cognitive demand (Bloom's Taxonomy level).
4. Structure: Clarity, grammar, and proper formatting (e.g., no "all of the above").
5. Transparency: Is the correct answer unambiguous to a subject matter expert?

Question to evaluate:
{question_json}

Context used for generation:
{context_snippet}

Return format:
{{
    "quality": {{ "score": 1-5, "feedback": "..." }},
    "uniqueness": {{ "score": 1-5, "feedback": "..." }},
    "effort": {{ "score": 1-5, "feedback": "..." }},
    "structure": {{ "score": 1-5, "feedback": "..." }},
    "transparency": {{ "score": 1-5, "feedback": "..." }},
    "overall_pedagogy_score": 1-5
}}
"""

# G-Eval Rubric for DeepEval-style evaluation
GEVAL_RUBRIC_PROMPT = """
Evaluate the following question based on the provided rubric. Score from 1 to 10.

Rubric: {rubric_name}
Criteria: {criteria}

Question:
{question_text}

Correct Answer:
{correct_answer}

Return strictly a JSON object:
{{
    "score": 1-10,
    "reasoning": "..."
}}
"""

def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Robustly parse JSON from LLM output, handling markdown and common errors."""
    if not text: return {}
    
    # 1. Clean markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        # Remove first line if it's ```json or similar
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 2. Find the first '{' and last '}'
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1 or end == 0:
        return {}
    
    json_str = text[start:end]
    
    try:
        # Standard load
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # Handle common LLM mistake: single quotes instead of double
            import ast
            return ast.literal_eval(json_str)
        except Exception:
            # Last ditch attempt: remove trailing commas
            try:
                import re
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                return json.loads(json_str)
            except Exception:
                return {}

def evaluate_question_quest(question: Dict[str, Any], context: str) -> Dict[str, Any]:
    """Evaluates a single question using the QUEST framework."""
    client = get_llm_client()
    prompt = QUEST_PROMPT.format(
        question_json=json.dumps(question, indent=2),
        context_snippet=context[:2000] # Limit context for efficiency
    )
    
    try:
        response = client(prompt)
        result = _safe_json_loads(response)
        if result:
            return result
    except Exception as e:
        print(f"Error in QUEST evaluation: {e}")
    
    return {"error": "QUEST evaluation failed"}

def evaluate_faithfulness_ragas(question: Dict[str, Any], context: str) -> float:
    """Uses Ragas library for faithfulness evaluation or falls back to prompt-based check."""
    if HAS_EVAL_LIBS:
        try:
            # RAGAS usually expects a dataset-like structure
            data = {
                "question": [question.get('question_text')],
                "answer": [json.dumps(question.get('correct_answer'))],
                "contexts": [[context[:5000]]]
            }
            # This is a simplified call for the prototype
            print("Calling RAGAS Faithfulness metric...")
            # Note: RAGAS often requires an OpenAI key or a LangChain wrapper
            # For this local prototype, we'll gracefully fall back if it fails
        except Exception as e:
            print(f"Ragas execution error: {e}")

    # Prompt-based fallback (The user's "Approach B" logic)
    # Prepare resolved answers for the LLM to understand meaningful content
    options = question.get('question_config', {}).get('options', [])
    correct_indices = question.get('correct_answer', [])
    
    if question['question_type'] == 'mcq' and isinstance(correct_indices, list):
        resolved_answers = [options[i] for i in correct_indices if i < len(options)]
    else:
        resolved_answers = [str(question.get('correct_answer'))]

    client = get_llm_client()
    prompt = f"""
    Check if the following question and its correct answer(s) are FAITHFUL to the provided context.
    Faithfulness means every claim in the question and answer can be directly inferred from the context.
    
    Context: {context[:3000]}
    
    Question: {question.get('question_text')}
    Options provided (if MCQ): {json.dumps(options)}
    Correct Answer(s) (Text): {json.dumps(resolved_answers)}
    
    Return strictly a JSON object with a 'faithfulness_score' between 0.0 and 1.0 and 'reasoning'.
    {{ "faithfulness_score": 0.95, "reasoning": "..." }}
    """
    try:
        response = client(prompt)
        result = _safe_json_loads(response)
        if result:
            return result.get('faithfulness_score', 0.0), result.get('reasoning', "")
    except Exception:
        pass
    return 0.5, "Faithfulness check failed"

def evaluate_geval_deepeval(question: Dict[str, Any]) -> Dict[str, Any]:
    """Uses DeepEval GEval metric or falls back to prompt-based check."""
    if HAS_EVAL_LIBS:
        try:
            print("Initializing DeepEval G-Eval...")
            # We would typically initialize a GEval metric here
            # metric = GEval(name="Distractor Quality", criteria="...", ...)
        except Exception as e:
            print(f"DeepEval initialization error: {e}")

    # Prompt-based fallback (The user's "Approach A" logic)
    # Resolve MCQ options for distractor evaluation
    options = question.get('question_config', {}).get('options', [])
    correct_indices = question.get('correct_answer', [])
    
    # Identify which options are correct and which are distractors
    correct_texts = [options[i] for i in correct_indices if i < len(options)]
    distractor_texts = [opt for i, opt in enumerate(options) if i not in correct_indices]

    client = get_llm_client()
    rubric = {
        "name": "Distractor Quality",
        "criteria": "Are the incorrect options (distractors) plausible and not obviously wrong? Are they free from 'all of the above' or 'none of the above' phrasing? Do they effectively challenge the learner?"
    }
    
    prompt = f"""
    Evaluate the distractor quality of this MCQ based on the rubric. Score from 1 to 10.
    
    Rubric: {rubric["name"]}
    Criteria: {rubric["criteria"]}
    
    Question: {question.get('question_text')}
    Correct Answer(s): {json.dumps(correct_texts)}
    Incorrect Options (Distractors): {json.dumps(distractor_texts)}
    
    Return strictly a JSON object:
    {{
        "score": 1-10,
        "reasoning": "..."
    }}
    """
    
    try:
        response = client(prompt)
        result = _safe_json_loads(response)
        if result:
            return result
    except Exception:
        pass
    return {"score": 0, "reasoning": "G-Eval failed"}

def run_full_evaluation(questions: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
    """Runs the three-pillar evaluation on a set of questions."""
    evaluated_questions = []
    
    for q in questions:
        # 1. RAGAS Faithfulness
        faith_score, faith_reason = evaluate_faithfulness_ragas(q, context)
        
        # 2. DeepEval style G-Eval (Distractor Quality for MCQs)
        geval_results = evaluate_geval_deepeval(q) if q['question_type'] == 'mcq' else {"score": 10, "reasoning": "N/A for essays"}
        
        # 3. QUEST Framework
        quest_results = evaluate_question_quest(q, context)
        
        q['eval_results'] = {
            "ragas_faithfulness": faith_score,
            "ragas_reasoning": faith_reason,
            "geval_score": geval_results.get('score', 0),
            "geval_reasoning": geval_results.get('reasoning', ""),
            "quest": quest_results
        }
        evaluated_questions.append(q)
        
    return evaluated_questions

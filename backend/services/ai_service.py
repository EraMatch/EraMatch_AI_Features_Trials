import os
import json
import requests
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Service configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama") # or "openai"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_llm_client():
    """Returns a function that can be used to call the configured LLM."""
    if AI_PROVIDER == "openai" and OPENAI_API_KEY:
        return _call_openai
    return _call_ollama

def call_ai(prompt: str) -> str:
    """Helper to call the configured AI provider."""
    client = get_llm_client()
    return client(prompt)

print(f"AI Service initialized: provider={AI_PROVIDER}, model={OLLAMA_MODEL}")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

PROMPT_GENERATE = """
- tags: An array of 2-3 relevant topic strings (e.g. ["Python", "Loops"])
- points: Default to 10

Strict Grounding Rules:
1. ONLY generate questions based on the factual content provided.
2. Ignore irrelevant administrative sections, headers, footers, or mission statements unless they are the primary subject.
3. If a question cannot be fully justified by the context, do not generate it.
4. For MCQs, distractors must be plausible but clearly incorrect based on the text.

Schema Example:
[
  {{
    "question_type": "mcq",
    "question_text": "What is X?",
    "question_config": {{"options": ["A", "B", "C"], "evidence": "Snippet X is used here..."}},
    "correct_answer": [0, 2],
    "difficulty": {difficulty_score},
    "tags": ["TopicA", "TopicB"],
    "points": 10
  }}
]
"""

PROMPT_EXTRACT = """
You are a parsing assistant for EraMatch. Extract existing questions from the following text into the EraMatch schema.

Fields:
- question_type: 'mcq' or 'essay'
- question_text: The full text of the question
- question_config: {{"options": [...], "evidence": "snippet"}} for mcq, else {{"evidence": "snippet"}}
- correct_answer: Array of indices for mcq if found, else reference answer for essay
- difficulty: infer integer 1-5
- tags: relevant topics
- points: 10
"""

def _call_ollama(prompt: str) -> str:
    current_model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    print(f"Calling Ollama with model: {current_model}...")
    payload = {
        "model": current_model,
        "prompt": prompt,
        "stream": False,
        # "format": "json" # Commented out for better compatibility
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json().get("response", "")
        print(f"Ollama Success! Response length: {len(res_json)}")
        return res_json
    except Exception as e:
        print(f"Ollama Error: {e}")
        return ""

def _call_openai(prompt: str) -> str:
    print(f"Calling OpenAI with model: gpt-4o-mini...")
    if not client:
        print("OpenAI Error: Client not initialized (missing API key?)")
        return ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        res_text = response.choices[0].message.content
        print(f"OpenAI Success! Response length: {len(res_text)}")
        return res_text
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return ""

def _get_mock_fallback(mode: str) -> List[Dict]:
    """Fallback mock data when AI is unavailable."""
    if mode == "generate":
        return [
            {"id": "m1", "type": "mcq", "title": "Fallback: Could not reach AI. Is Ollama running?", "options": ["Yes", "No"], "correct": 0, "difficulty": "Easy"},
            {"id": "m2", "type": "essay", "title": "Fallback: Explain how to configure AI_PROVIDER in .env", "difficulty": "Medium"}
        ]
    return [
        {"id": "e1", "type": "mcq", "title": "Fallback: Extracted question mock", "options": ["Opt A", "Opt B"], "correct": 0, "difficulty": "Medium"}
    ]

def _parse_ai_json(raw_text: str, fallback_mode: str) -> List[Dict]:
    """Safely parse AI JSON response, handling various formats."""
    print(f"RAW AI RESPONSE: {raw_text[:500]}..." if len(raw_text) > 500 else f"RAW AI RESPONSE: {raw_text}")
    
    if not raw_text or not raw_text.strip():
        print(f"AI response was empty. Falling back to {fallback_mode} mocks.")
        return _get_mock_fallback(fallback_mode)
    
    try:
        # Pre-process text to find JSON if model included conversational garbage
        start = raw_text.find('[')
        end = raw_text.rfind(']') + 1
        if start != -1 and end != -1:
            json_str = raw_text[start:end]
            data = json.loads(json_str)
            return data if isinstance(data, list) else [data]
        
        # Try finding a dictionary
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = raw_text[start:end]
            data = json.loads(json_str)
            if isinstance(data, dict):
                for key in ["questions", "data", "list"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
        
        print("Could not find JSON structure in response. Falling back.")
        return _get_mock_fallback(fallback_mode)
    except Exception as e:
        print(f"JSON Parse Error: {e}. Raw was: {raw_text[:200]}")
        return _get_mock_fallback(fallback_mode)

def _difficulty_to_score(difficulty: str) -> int:
    mapping = {"Easy": 2, "Medium": 3, "Hard": 5}
    return mapping.get(difficulty, 3)

def generate_questions_ai(context: str, mcq_count: int, essay_count: int, difficulty: str) -> List[Dict]:
    difficulty_score = _difficulty_to_score(difficulty)
    prompt = PROMPT_GENERATE.format(
        mcq_count=mcq_count,
        essay_count=essay_count,
        difficulty_score=difficulty_score
    ) + f"\n\nContext:\n{context}"
    
    raw_response = call_ai(prompt)
    questions = _parse_ai_json(raw_response, "generate")
    
    # Run Evaluation (DeepEval, RAGAS, QUEST)
    try:
        from services.eval_service import run_full_evaluation
        print("Running AI Evaluation Pillar...")
        return run_full_evaluation(questions, context)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return questions

def extract_questions_ai(context: str) -> List[Dict]:
    prompt = PROMPT_EXTRACT + f"\n\nContext:\n{context}"
    raw_response = call_ai(prompt)
    questions = _parse_ai_json(raw_response, "extract")
    
    # Run Evaluation for extraction too
    try:
        from services.eval_service import run_full_evaluation
        print("Running AI Evaluation Pillar for extraction...")
        return run_full_evaluation(questions, context)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return questions

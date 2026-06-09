import os
import json
import torch
from tqdm import tqdm
import evaluate
from unsloth import FastLanguageModel
from datasets import Dataset
import glob
from openai import OpenAI

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_FOLDER = "/kaggle/input/datasets/adhamashraf202200953/technical-parsed-questions"
MODEL_PATH = "/kaggle/working/UNSLOTH_MISTRAL_7B_LORA" # Path to your fine-tuned model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here") # Or DeepSeek API Key
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") # Change for DeepSeek if needed

# Initialize LLM Client for LLM-as-a-Judge
llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ==============================================================================
# 1. DATA LOADING HELPER
# ==============================================================================
def load_and_parse_custom_jsonl(folder_path):
    all_records = []
    search_pattern = os.path.join(folder_path, "*.jsonl")
    jsonl_files = glob.glob(search_pattern)
    
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    if all(k in record for k in ["context", "scenario_context", "question", "model_answer"]):
                        all_records.append(record)
                except:
                    continue
    return all_records

# ==============================================================================
# 2. HIERARCHICAL LLM-AS-A-JUDGE EVALUATION
# ==============================================================================
LLJ_SYSTEM_PROMPT = """You are an expert evaluator for an AI Proctoring & Interview system (EraMatch).
You are evaluating the quality of an automatically generated interview question suite.
You must use a strict hierarchical decomposition evaluation style.

Respond to the following Yes/No questions based on the Textbook Context and the Generated Output.
You MUST output valid JSON only, using the exact keys provided. Use true for 'Yes' and false for 'No'.

EVALUATION CRITERIA:
1. "has_scenario": Does the output include a clearly defined scenario?
2. "has_question": Does the output include a clearly defined question?
3. "has_answer": Does the output include a model answer?
4. "scenario_relevant": Is the scenario accurately derived from the textbook context without hallucinatory external facts?
5. "question_analytical": Does the question require problem-solving or analytical thinking (rather than simple memorization)?
6. "answer_correct": Is the model answer technically correct and aligned with the textbook context?
7. "answer_comprehensive": Does the model answer fully resolve the question?

OUTPUT FORMAT:
{
    "has_scenario": true/false,
    "has_question": true/false,
    "has_answer": true/false,
    "scenario_relevant": true/false,
    "question_analytical": true/false,
    "answer_correct": true/false,
    "answer_comprehensive": true/false
}
"""

def evaluate_with_llj(context, generation):
    prompt = f"Textbook Context:\n{context}\n\nGenerated Output:\n{generation}"
    
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o", # Or 'deepseek-chat' / 'deepseek-reasoner'
            messages=[
                {"role": "system", "content": LLJ_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        
        result = json.loads(response.choices[0].message.content)
        
        score = sum(1 for v in result.values() if v is True)
        total = len(result)
        
        return {
            "llj_score": score,
            "llj_total": total,
            "llj_breakdown": result
        }
    except Exception as e:
        print(f"LLJ Error: {e}")
        return None

# ==============================================================================
# 3. MAIN EVALUATION SCRIPT
# ==============================================================================
def main():
    print("Loading datasets...")
    raw_parsed_data = load_and_parse_custom_jsonl(DATA_FOLDER)
    dataset = Dataset.from_list(raw_parsed_data)
    
    # Re-create the same test split
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    test_dataset = split_dataset["test"]
    
    # We evaluate on a small subset for time considerations (e.g. 50 samples)
    eval_dataset = test_dataset.select(range(min(50, len(test_dataset))))
    
    print(f"Loading Model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 384,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    print("Loading metrics...")
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    bertscore = evaluate.load("bertscore")
    
    generated_texts = []
    reference_texts = []
    llj_scores = []
    llj_breakdowns = {
        "has_scenario": 0, "has_question": 0, "has_answer": 0,
        "scenario_relevant": 0, "question_analytical": 0, 
        "answer_correct": 0, "answer_comprehensive": 0
    }
    
    print("Running Inference & Evaluation...")
    for item in tqdm(eval_dataset):
        context = item['context']
        
        # Format the inference prompt
        prompt = f"<s>[INST] You are an expert technical interviewer. Based on the given textbook context, generate a professional workplace scenario context, an analytical interview question, and a detailed model answer.\n\nTextbook Context:\n{context} [/INST]\n"
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        generation = generated_text.split("[/INST]")[-1].strip()
        
        # Build the reference string
        reference = f"Scenario: {item['scenario_context']}\nQuestion: {item['question']}\nModel Answer: {item['model_answer']}"
        
        generated_texts.append(generation)
        reference_texts.append(reference)
        
        # Run Hierarchical LLM-as-a-Judge
        llj_result = evaluate_with_llj(context, generation)
        if llj_result:
            llj_scores.append(llj_result["llj_score"] / llj_result["llj_total"])
            for key, val in llj_result["llj_breakdown"].items():
                if val:
                    llj_breakdowns[key] += 1
                    
    # ==============================================================================
    # 4. COMPUTE FINAL METRICS
    # ==============================================================================
    print("\n--- NLP METRICS ---")
    rouge_res = rouge.compute(predictions=generated_texts, references=reference_texts)
    print(f"ROUGE: {rouge_res}")
    
    bleu_res = bleu.compute(predictions=generated_texts, references=reference_texts)
    print(f"BLEU: {bleu_res}")
    
    bert_res = bertscore.compute(predictions=generated_texts, references=reference_texts, lang="en")
    mean_bertscore = sum(bert_res['f1']) / len(bert_res['f1'])
    print(f"BERTScore (F1 mean): {mean_bertscore:.4f}")
    
    print("\n--- HIERARCHICAL LLM-AS-A-JUDGE METRICS ---")
    if llj_scores:
        avg_llj = sum(llj_scores) / len(llj_scores)
        print(f"Overall LLJ Accuracy (Yes ratio): {avg_llj:.2%}")
        print("Breakdown (% of generations that passed each check):")
        total_evals = len(llj_scores)
        for key, count in llj_breakdowns.items():
            print(f"  - {key}: {count/total_evals:.2%}")
    else:
        print("LLJ Evaluation failed or was skipped.")

if __name__ == "__main__":
    main()

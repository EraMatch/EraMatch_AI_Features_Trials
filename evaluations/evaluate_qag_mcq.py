import os
import torch
import pandas as pd
from tqdm import tqdm
import evaluate
from unsloth import FastLanguageModel
from datasets import Dataset
import glob
from openai import OpenAI
import json
import gc

DATA_FOLDER = "/root/.cache/kagglehub/datasets/adhamashraf202200953/technical-parsed-questions/versions/5"
MODEL_PATH = "./outputs/STABLE_LLAMA_MCQ"
OLLAMA_API_KEY = "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
JUDGE_MODELS = ["llama3", "mistral"]
llm_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)


def load_mcq_data(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*_mcq.csv"))
    if not csv_files:
        print(f"No MCQ CSV files found in {folder_path}")
        return None
    required_columns = [
        "context",
        "scenario_context",
        "question",
        "choices/0",
        "choices/1",
        "choices/2",
        "choices/3",
        "correct_choice",
    ]
    all_dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, usecols=required_columns)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    if not all_dfs:
        return None
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=["context", "question", "correct_choice"])
    return combined_df


LLJ_SYSTEM_PROMPT = """You are an expert technical interviewer evaluating an automatically generated Multiple Choice Question (MCQ).
You must use a strict hierarchical decomposition evaluation style.
Respond to the following Yes/No questions based on the Textbook Context and the Generated MCQ Output.
You MUST output valid JSON only, using the exact keys provided. Use true for 'Yes' and false for 'No'.
EVALUATION CRITERIA:
1. "has_scenario": Does the output include a clearly defined Scenario?
2. "has_question": Does the output include a clearly defined Question?
3. "has_4_choices": Does the output include exactly 4 distinct choices labeled A, B, C, and D?
4. "has_answer": Does the output specify the correct answer at the end?
5. "question_relevant": Is the generated question highly relevant to the provided textbook context?
6. "answer_correct": Is the specified correct answer technically accurate based on the context?
7. "distractors_plausible": Are the incorrect choices (distractors) plausible yet definitively incorrect (no ambiguity)?
OUTPUT FORMAT:
{
    "has_scenario": true/false,
    "has_question": true/false,
    "has_4_choices": true/false,
    "has_answer": true/false,
    "question_relevant": true/false,
    "answer_correct": true/false,
    "distractors_plausible": true/false
}
"""


def evaluate_with_llj(context, generation, judge_model):
    prompt = f"Textbook Context:\n{context}\n\nGenerated MCQ Output:\n{generation}"
    try:
        response = llm_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": LLJ_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        score = sum(1 for v in result.values() if v is True)
        return {"llj_score": score, "llj_total": len(result), "llj_breakdown": result}
    except Exception as e:
        print(f"LLJ Error: {e}")
        return None


def main():
    print("Loading datasets...")
    df = load_mcq_data(DATA_FOLDER)
    if df is None:
        return
    dataset = Dataset.from_pandas(df)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = split_dataset["test"]
    eval_dataset = test_dataset.select(range(min(50, len(test_dataset))))
    print(f"Loading Model from {MODEL_PATH}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    print("Loading metrics...")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    generated_texts = []
    reference_texts = []
    llj_scores = {m: [] for m in JUDGE_MODELS}
    llj_breakdowns = {
        m: {
            "has_scenario": 0,
            "has_question": 0,
            "has_4_choices": 0,
            "has_answer": 0,
            "question_relevant": 0,
            "answer_correct": 0,
            "distractors_plausible": 0,
        }
        for m in JUDGE_MODELS
    }
    print("Running Inference & Evaluation...")
    for item in tqdm(eval_dataset):
        context = item["context"]
        prompt = f"Context: {context.strip()}\n\nScenario:"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        generation = generated_text[len(prompt) :].strip()
        reference = (
            f"Scenario: {str(item['scenario_context']).strip()}\n"
            f"Question: {str(item['question']).strip()}\n"
            f"Choices:\n"
            f"A) {str(item['choices/0']).strip()}\n"
            f"B) {str(item['choices/1']).strip()}\n"
            f"C) {str(item['choices/2']).strip()}\n"
            f"D) {str(item['choices/3']).strip()}\n"
            f"Answer: {str(item['correct_choice']).strip()}"
        )
        generated_texts.append(generation)
        reference_texts.append(reference)
        for judge_model in JUDGE_MODELS:
            llj_result = evaluate_with_llj(context, generation, judge_model)
            if llj_result:
                llj_scores[judge_model].append(
                    llj_result["llj_score"] / llj_result["llj_total"]
                )
                for key, val in llj_result["llj_breakdown"].items():
                    if val:
                        llj_breakdowns[judge_model][key] += 1
    print("\n--- NLP METRICS ---")
    rouge_res = rouge.compute(predictions=generated_texts, references=reference_texts)
    print(f"ROUGE: {rouge_res}")
    bleu_res = bleu.compute(predictions=generated_texts, references=reference_texts)
    print(f"BLEU: {bleu_res}")
    bert_res = bertscore.compute(
        predictions=generated_texts, references=reference_texts, lang="en"
    )
    mean_bertscore = sum(bert_res["f1"]) / len(bert_res["f1"])
    print(f"BERTScore (F1 mean): {mean_bertscore:.4f}")
    print("\n--- HIERARCHICAL LLM-AS-A-JUDGE METRICS ---")
    for judge_model in JUDGE_MODELS:
        print(f"\n--- RESULTS FOR JUDGE: {judge_model} ---")
        scores = llj_scores[judge_model]
        if scores:
            avg_llj = sum(scores) / len(scores)
            print(f"Overall LLJ Accuracy (Yes ratio): {avg_llj:.2%}")
            print("Breakdown (% of generations that passed each check):")
            total_evals = len(scores)
            for key, count in llj_breakdowns[judge_model].items():
                print(f"  - {key}: {count/total_evals:.2%}")
        else:
            print("LLJ Evaluation failed or was skipped.")


if __name__ == "__main__":
    main()

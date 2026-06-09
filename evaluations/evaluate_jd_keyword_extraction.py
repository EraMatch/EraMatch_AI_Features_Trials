import os
import json
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import Dataset
import glob
from openai import OpenAI
import re

DATA_FOLDER = "/kaggle/input/datasets/adhamashraf202200953/jd-keywords"
MODEL_PATH = "/kaggle/working/UNSLOTH_KEYWORD_EXTRACTION_LORA"
OLLAMA_API_KEY = "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
JUDGE_MODELS = ["llama3", "mistral"]
llm_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)


def load_and_parse_custom_jsonl(folder_path):
    all_records = []
    search_pattern = os.path.join(folder_path, "*.jsonl")
    jsonl_files = glob.glob(search_pattern)
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if all(k in record for k in ["job_description", "keywords"]):
                        all_records.append(record)
                except:
                    continue
    return all_records


LLJ_SYSTEM_PROMPT = """You are an expert HR Data Scientist evaluating an automated Keyword Extraction system.
You are evaluating the quality of keywords extracted from a Job Description.
You must use a strict hierarchical decomposition evaluation style.
Respond to the following Yes/No questions based on the Job Description and the Extracted Keywords.
You MUST output valid JSON only, using the exact keys provided. Use true for 'Yes' and false for 'No'.
EVALUATION CRITERIA:
1. "has_keywords_list": Did the output successfully produce a recognizable list or comma-separated sequence of keywords?
2. "keywords_relevant": Are the extracted keywords highly relevant to the role described in the Job Description?
3. "captured_core_skills": Did the extraction successfully capture the primary hard skills mentioned in the JD?
4. "no_hallucinations": Are you confident that NO completely fabricated or completely irrelevant keywords were hallucinated?
OUTPUT FORMAT:
{
    "has_keywords_list": true/false,
    "keywords_relevant": true/false,
    "captured_core_skills": true/false,
    "no_hallucinations": true/false
}
"""


def evaluate_with_llj(jd, keywords, judge_model):
    prompt = f"Job Description:\n{jd}\n\nExtracted Keywords:\n{keywords}"
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


def compute_keyword_metrics(pred_str, true_str):
    pred_kws = set(
        k.strip().lower() for k in re.split(r"[,;|\n]", pred_str) if k.strip()
    )
    true_kws = set(
        k.strip().lower() for k in re.split(r"[,;|\n]", true_str) if k.strip()
    )
    if not pred_kws:
        return 0.0, 0.0, 0.0
    overlap = len(pred_kws.intersection(true_kws))
    precision = overlap / len(pred_kws) if pred_kws else 0
    recall = overlap / len(true_kws) if true_kws else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


def main():
    print("Loading datasets...")
    raw_parsed_data = load_and_parse_custom_jsonl(DATA_FOLDER)
    if not raw_parsed_data:
        print(f"No data found in {DATA_FOLDER}")
        return
    dataset = Dataset.from_list(raw_parsed_data)
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    test_dataset = split_dataset["test"]
    eval_dataset = test_dataset.select(range(min(50, len(test_dataset))))
    print(f"Loading Model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=384,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    metrics_sum = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    llj_scores = {m: [] for m in JUDGE_MODELS}
    llj_breakdowns = {
        m: {
            "has_keywords_list": 0,
            "keywords_relevant": 0,
            "captured_core_skills": 0,
            "no_hallucinations": 0,
        }
        for m in JUDGE_MODELS
    }
    print("Running Inference & Evaluation...")
    for item in tqdm(eval_dataset):
        jd = item["job_description"]
        ground_truth = item["keywords"]
        prompt = f"<s>[INST] Extract the core technical and soft skills from the following job description as a comma-separated list.\n\nJob Description:\n{jd} [/INST]\n"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        generation = generated_text.split("[/INST]")[-1].strip()
        p, r, f1 = compute_keyword_metrics(generation, ground_truth)
        metrics_sum["precision"] += p
        metrics_sum["recall"] += r
        metrics_sum["f1"] += f1
        for judge_model in JUDGE_MODELS:
            llj_result = evaluate_with_llj(jd, generation, judge_model)
            if llj_result:
                llj_scores[judge_model].append(
                    llj_result["llj_score"] / llj_result["llj_total"]
                )
                for key, val in llj_result["llj_breakdown"].items():
                    if val:
                        llj_breakdowns[judge_model][key] += 1
    n_evals = len(eval_dataset)
    print("\n--- KEYWORD MATCHING METRICS (Exact/Partial Match) ---")
    print(f"Average Precision: {metrics_sum['precision']/n_evals:.4f}")
    print(f"Average Recall:    {metrics_sum['recall']/n_evals:.4f}")
    print(f"Average F1-Score:  {metrics_sum['f1']/n_evals:.4f}")
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

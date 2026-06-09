import os
import torch
import pandas as pd
from tqdm import tqdm
import evaluate
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
import glob
from openai import OpenAI
import json

DATA_FOLDER = "/kaggle/input/datasets/adhamashraf202200953/technical-parsed-questions"
MODEL_PATH = "/kaggle/working/UNSLOTH_QA_RUBRIC_LORA"
OLLAMA_API_KEY = "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
JUDGE_MODELS = ["llama3", "mistral"]
llm_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)


def load_rubric_data(folder_path):
    file_pattern = os.path.join(folder_path, "*_essay_variants.csv")
    csv_files = glob.glob(file_pattern)
    if not csv_files:
        print(f"No essay variant CSV files found in {folder_path}")
        return None
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


LLJ_SYSTEM_PROMPT = """You are an expert technical grading coordinator.
You are evaluating an automatically generated grading Rubric and Yes/No verification questions for an essay response.
You must use a strict hierarchical decomposition evaluation style.
Respond to the following Yes/No questions based on the provided inputs and Generated Output.
You MUST output valid JSON only, using the exact keys provided. Use true for 'Yes' and false for 'No'.
EVALUATION CRITERIA:
1. "has_rubric_description": Does the output include a '### RUBRIC DESCRIPTION' section?
2. "has_yes_no_questions": Does the output include a '### YES/NO QUESTIONS' section with a list?
3. "rubric_relevant": Is the rubric description accurately aligned with the Model Answer?
4. "questions_objective": Are the generated Yes/No questions truly binary and objectively gradable?
5. "questions_comprehensive": Do the Yes/No questions comprehensively cover the critical technical elements of the Model Answer?
OUTPUT FORMAT:
{
    "has_rubric_description": true/false,
    "has_yes_no_questions": true/false,
    "rubric_relevant": true/false,
    "questions_objective": true/false,
    "questions_comprehensive": true/false
}
"""


def evaluate_with_llj(context, question, model_answer, generation, judge_model):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nModel Answer:\n{model_answer}\n\nGenerated Rubric:\n{generation}"
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
    df = load_rubric_data(DATA_FOLDER)
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
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3",
            mapping={
                "role": "role",
                "content": "content",
                "user": "user",
                "assistant": "assistant",
            },
        )
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
            "has_rubric_description": 0,
            "has_yes_no_questions": 0,
            "rubric_relevant": 0,
            "questions_objective": 0,
            "questions_comprehensive": 0,
        }
        for m in JUDGE_MODELS
    }
    print("Running Inference & Evaluation...")
    for item in tqdm(eval_dataset):
        context = item.get("context", "N/A")
        question = item.get("question", "N/A")
        model_answer = item.get("model_answer", "N/A")
        messages = [
            {
                "role": "system",
                "content": "You are an automated technical assessment expert. Analyze the provided Question and Model Answer to produce a comprehensive Rubric Description and list out exact binary Yes/No questions to grade candidate submissions.",
            },
            {
                "role": "user",
                "content": f"Context: {context}\nQuestion: {question}\nModel Answer: {model_answer}",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        user_content = (
            f"Context: {context}\nQuestion: {question}\nModel Answer: {model_answer}"
        )
        if user_content in generated_text:
            generation = generated_text.split(user_content)[-1].strip()
        else:
            generation = generated_text
        yn_cols = [
            str(item[c]).strip()
            for c in item.keys()
            if str(c).startswith("yes_no_questions/")
            and pd.notna(item[c])
            and str(item[c]).strip() != ""
        ]
        formatted_questions = "\n".join([f"- {q}" for q in yn_cols])
        reference = f"### RUBRIC DESCRIPTION\n{item.get('rubric_description', 'N/A')}\n\n### YES/NO QUESTIONS\n{formatted_questions}"
        generated_texts.append(generation)
        reference_texts.append(reference)
        for judge_model in JUDGE_MODELS:
            llj_result = evaluate_with_llj(
                context, question, model_answer, generation, judge_model
            )
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

import os
import random
import pandas as pd
from tqdm import tqdm
import ollama
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_POJ = os.path.join(SCRIPT_DIR, "data/poj104_train.csv")
OUTPUT_POJ = os.path.join(SCRIPT_DIR, "data/synthetic_poj104.csv")

MODEL_ID = "qwen2.5-coder"

print(f"Using local Ollama with {MODEL_ID} model...")

def clean_bpe_artifacts(raw_text):
    if not isinstance(raw_text, str):
        return raw_text
    clean_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text.strip()

def generate_code_variant(prompt_text, language="cpp"):
    """
    Calls the local Ollama instance to generate the code variant.
    """
    try:
        response = ollama.chat(model=MODEL_ID, messages=[
            {
                'role': 'user',
                'content': prompt_text
            }
        ], options={
            'temperature': 0.85,
            'top_p': 0.95,
            'num_predict': 512 
        })
        
        generated_text = response['message']['content']
        generated_text = clean_bpe_artifacts(generated_text)

        # Extract code from markdown blocks
        extracted_code = generated_text
        if "```" in generated_text:
            try:
                extracted_code = generated_text.split(f"```{language}")[1].split("```")[0].strip()
            except IndexError:
                try:
                    extracted_code = generated_text.split("```")[1].split("```")[0].strip()
                except IndexError:
                    pass

        return extracted_code.strip()
        
    except Exception as e:
        print(f"\n[!] Error generating code with Ollama: {e}")
        return ""

# ==========================================
# Main Processing Functions
# ==========================================
def synthesize_poj104_local(num_variants_per_problem=500):
    print("\n--- Running Local POJ-104 Track ---")

    if not os.path.exists(INPUT_POJ):
        print(f"🔴 Error: Cannot find input file at {INPUT_POJ}.")
        print("Please ensure the CSV file is placed in the './Code_Datasets/' folder.")
        return

    df_poj = pd.read_csv(INPUT_POJ)
    grouped = df_poj.groupby("label")

    if os.path.exists(OUTPUT_POJ):
        print("Found existing checkpoint! Resuming...")
        df_existing = pd.read_csv(OUTPUT_POJ)
        synthetic_records = df_existing.to_dict("records")
        counts = df_existing["label"].value_counts()
        completed_labels = set(str(x) for x in counts[counts >= num_variants_per_problem].index)
        print(f"Skipping {len(completed_labels)} already finished categories.")
    else:
        print("Starting fresh task.")
        synthetic_records = []
        completed_labels = set()

    outer_pbar = tqdm(grouped, desc="Overall POJ Problems", total=len(grouped))

    for label, group in outer_pbar:
        str_label = str(label)

        if str_label in completed_labels:
            continue

        existing_count = sum(1 for r in synthetic_records if str(r["label"]) == str_label)
        variants_to_generate = num_variants_per_problem - existing_count

        if variants_to_generate <= 0:
            continue

        all_human_codes = group["code"].dropna().tolist()

        print(f"\n[+] Problem ID: {label} | Found: {existing_count} existing | Remaining to generate: {variants_to_generate}")

        inner_pbar = tqdm(total=variants_to_generate, desc=f"Problem {label} Variants", leave=False)

        for i in range(variants_to_generate):
            inner_pbar.set_description(f"Problem {label} | Generating {i+1}/{variants_to_generate}")

            selected_samples = random.sample(all_human_codes, k=min(3, len(all_human_codes)))

            concatenated_block = ""
            for idx, code_sample in enumerate(selected_samples):
                safe_sample = str(code_sample)[:1000]
                concatenated_block += f"\n--- Sample {idx+1} ---\n{safe_sample}\n"

            prompt = f"""
            You are an advanced AI Code Generator. Analyze the following cluster of C++ source codes written to solve the EXACT SAME algorithmic problem.

            Cluster of Human Reference Solutions:
            {concatenated_block}

            Write a brand new, fully functional C++ source code that solves this identical problem using a completely distinct programming architecture, naming convention, and code layout.

            Strict Operational Rules:
            1. DO NOT reuse specific variable names or line-by-line formatting patterns from the references.
            2. Fully refactor the logic.
            3. Ensure the code is self-contained and clean.
            4. Respond with NOTHING else except the code inside a single ```cpp ... ``` block.
            """

            ai_code = generate_code_variant(prompt, language="cpp")
            
            if ai_code: # Only append if code was successfully generated
                synthetic_records.append({"code": ai_code, "label": label})

            # Save checkpoint every 10 generations
            if len(synthetic_records) % 10 == 0:
                pd.DataFrame(synthetic_records).to_csv(OUTPUT_POJ, index=False)

            inner_pbar.update(1)

        inner_pbar.close()

        # Save at the end of each problem group
        pd.DataFrame(synthetic_records).to_csv(OUTPUT_POJ, index=False)
        print(f"\n[Secured] Problem {label} completely generated.")


if __name__ == "__main__":
    # Ensure Ollama service is running in the background before executing this script!
    synthesize_poj104_local(num_variants_per_problem=500)
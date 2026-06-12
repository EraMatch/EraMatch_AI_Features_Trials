import os
import random
import re
import pandas as pd
from tqdm import tqdm
import ollama

# =====================================================
# Configuration & Paths
# =====================================================
LOCAL_DIR = "./Code_Datasets/"
os.makedirs(LOCAL_DIR, exist_ok=True)

INPUT_BCB = os.path.join(LOCAL_DIR, "bigclonebench_train.parquet")
OUTPUT_BCB = os.path.join(LOCAL_DIR, "synthetic_bcb_files.parquet")

MODEL_ID = "qwen2.5-coder" 

print(f"Initializing local Ollama with {MODEL_ID} model...")

# =====================================================
# Helper Functions
# =====================================================
def clean_bpe_artifacts(raw_text):
    if not isinstance(raw_text, str):
        return raw_text
    clean_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text.strip()

def generate_code_variant(prompt_text, language="java"):
    try:
        response = ollama.chat(model=MODEL_ID, messages=[
            {
                'role': 'user',
                'content': prompt_text
            }
        ], options={
            'temperature': 0.85,
            'top_p': 0.95,
            'num_predict': 512 # equivalent to max_new_tokens
        })
        
        generated_text = response['message']['content']
        generated_text = clean_bpe_artifacts(generated_text)

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

# =====================================================
# Main Processing Function
# =====================================================
def synthesize_bcb_local(num_files_needed=25000):
    print("\n--- Running Local BigCloneBench (Java) Track ---")

    if not os.path.exists(INPUT_BCB):
        print(f"🔴 Error: Cannot find input file at {INPUT_BCB}.")
        print("Please ensure 'bigclonebench_train.parquet' is inside the './Code_Datasets/' folder.")
        return

    df_bcb = pd.read_parquet(INPUT_BCB)

    if os.path.exists(OUTPUT_BCB):
        print("Found existing Parquet checkpoint! Resuming...")
        df_existing = pd.read_parquet(OUTPUT_BCB)
        synthetic_records = df_existing.to_dict("records")
        start_index = len(synthetic_records)
        print(f"Resuming from index {start_index}...")
    else:
        print("Starting fresh task.")
        synthetic_records = []
        start_index = 0

    # Extracting all unique Java functions from the dataset
    all_java_functions = df_bcb["func1"].dropna().unique().tolist()
    counter = start_index

    pbar = tqdm(total=num_files_needed, initial=start_index, desc="Synthesizing Java Codes")
    
    while counter < num_files_needed:
        selected_samples = random.sample(all_java_functions, k=min(3, len(all_java_functions)))

        concatenated_block = ""
        for idx, func_sample in enumerate(selected_samples):
            safe_sample = str(func_sample)[:1000] # Protection against massive strings
            concatenated_block += f"\n--- Sample {idx+1} ---\n{safe_sample}\n"

        prompt = f"""
        You are an Enterprise Java Code Synthesis Engine. Examine this cluster of production Java methods.

        Cluster of Reference Java Implementations:
        {concatenated_block}

        Extract the core backend business logic and write a completely brand new, highly robust, and distinct Java method that satisfies that identical functional objective.

        Strict Operational Rules:
        1. Fully rename all local parameters and logic routines.
        2. Do not mirror any single reference layout. Heavy refactoring is strictly required.
        3. Respond with NOTHING else except the code block inside a single ```java ... ``` block.
        """

        ai_code = generate_code_variant(prompt, language="java")

        if ai_code: # Only append if the model returned valid code
            synthetic_records.append({
                "func": ai_code,
                "functional_group": counter // 25
            })
            counter += 1
            pbar.update(1)

            # Batch Saving every 100 files
            if len(synthetic_records) % 100 == 0:
                pd.DataFrame(synthetic_records).to_parquet(OUTPUT_BCB, index=False)

    # Final save after loop completion
    pd.DataFrame(synthetic_records).to_parquet(OUTPUT_BCB, index=False)
    pbar.close()
    print("BigCloneBench File Track Fully Completed and Secured!")


if __name__ == "__main__":
    synthesize_bcb_local(num_files_needed=25000)
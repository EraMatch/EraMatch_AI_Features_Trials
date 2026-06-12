# pip install datasets pyarrow fastparquet

import os
import random
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import ollama

LOCAL_DIR = "./Code_Datasets/"
os.makedirs(LOCAL_DIR, exist_ok=True)
OUTPUT_PYTHON = os.path.join(LOCAL_DIR, "synthetic_codenet_python.parquet")
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

def generate_code_variant(prompt_text, language="python"):
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
def synthesize_python_codenet(num_files_needed=25000):
    print("\n--- Running Local Python (CodeNet) Track ---")
    print("Downloading/Loading Project CodeNet Python split from HuggingFace...")
    
    # This will cache locally on your machine automatically
    dataset = load_dataset("claudios/code_search_net", "python", split="train")
    all_python_functions = dataset['func_code_string']

    if os.path.exists(OUTPUT_PYTHON):
        print("Found existing Python Parquet checkpoint! Resuming...")
        df_existing = pd.read_parquet(OUTPUT_PYTHON)
        synthetic_records = df_existing.to_dict("records")
        start_index = len(synthetic_records)
        print(f"Resuming from index {start_index}...")
    else:
        print("Starting fresh Python task.")
        synthetic_records = []
        start_index = 0

    counter = start_index
    pbar = tqdm(total=num_files_needed, initial=start_index, desc="Synthesizing Python Codes")

    while counter < num_files_needed:
        selected_samples = random.sample(all_python_functions, k=3)

        concatenated_block = ""
        for idx, func_sample in enumerate(selected_samples):
            safe_sample = str(func_sample)[:1000] # OOM Protection
            concatenated_block += f"\n--- Python Sample {idx+1} ---\n{safe_sample}\n"

        prompt = f"""
        You are an Expert Python Developer. Examine this cluster of Python functions written to solve similar tasks.

        Cluster of Reference Python Implementations:
        {concatenated_block}

        Write a completely brand new, highly robust, and distinct Python function that satisfies the same functional objective.

        Strict Operational Rules:
        1. Use Pythonic conventions (list comprehensions, generators, type hints if applicable).
        2. Fully rename all local parameters and logic routines.
        3. Do not mirror any single reference layout. Heavy refactoring is strictly required.
        4. Respond with NOTHING else except the code block inside a single ```python ... ``` block.
        """

        ai_code = generate_code_variant(prompt, language="python")

        if ai_code: # Only append if a valid response was generated
            synthetic_records.append({
                "code": ai_code,
                "language": "python",
                "id": counter
            })
            counter += 1
            pbar.update(1)

        # Batch Saving every 50 files
        if len(synthetic_records) % 50 == 0:
            pd.DataFrame(synthetic_records).to_parquet(OUTPUT_PYTHON, index=False)

    # Final save
    pd.DataFrame(synthetic_records).to_parquet(OUTPUT_PYTHON, index=False)
    pbar.close()
    print("Python Track Fully Completed and Secured!")


if __name__ == "__main__":
    synthesize_python_codenet(num_files_needed=25000)
# =========================================================================================
# OLLAMA ENVIRONMENT SETUP STEPS (DO THIS BEFORE RUNNING THE SCRIPT)
# =========================================================================================
# 1. Open a fresh PowerShell terminal.
# 2. Configure Ollama for maximum parallel processing on your RTX 5080:
#    $env:OLLAMA_NUM_PARALLEL="4"
# 3. Start the model in this terminal so the background server is awake and ready:
#    ollama run qwen2.5-coder
# 4. Leave that PowerShell window open, open a SEPARATE terminal, and launch this script:
#    python generate_java_dataset.py
# =========================================================================================

import os
import random
import re
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
from ollama import AsyncClient

# =====================================================
# Configuration & Paths
# =====================================================
LOCAL_DIR = "./Code_Datasets/"
os.makedirs(LOCAL_DIR, exist_ok=True)

INPUT_BCB = os.path.join(LOCAL_DIR, "bigclonebench_train.parquet")
OUTPUT_BCB = os.path.join(LOCAL_DIR, "synthetic_bcb_files.parquet")

MODEL_ID = "qwen2.5-coder" 
CONCURRENCY_LIMIT = 4  

print(f"Initializing local Async Ollama with {MODEL_ID} model...")

# =====================================================
# Helper Functions
# =====================================================
def clean_bpe_artifacts(raw_text):
    if not isinstance(raw_text, str):
        return raw_text
    clean_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text.strip()

async def generate_code_variant(client, prompt_text, language="java"):
    try:
        response = await client.chat(model=MODEL_ID, messages=[
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
        return ""

# =====================================================
# Worker Task Definition
# =====================================================
async def worker(client, all_java_functions, semaphore, state, pbar):
    while True:
        async with state['lock']:
            if state['counter'] >= state['target']:
                break
            current_idx = state['counter']
            state['counter'] += 1

        selected_samples = random.sample(all_java_functions, k=min(3, len(all_java_functions)))

        concatenated_block = ""
        for idx, func_sample in enumerate(selected_samples):
            safe_sample = str(func_sample)[:1000]  
            concatenated_block += f"\n--- Sample {idx+1} ---\n{safe_sample}\n"

        prompt = f"""You are an Enterprise Java Code Synthesis Engine. Examine this cluster of production Java methods.

Cluster of Reference Java Implementations:
{concatenated_block}

Extract the core backend business logic and write a completely brand new, highly robust, and distinct Java method that satisfies that identical functional objective.

Strict Operational Rules:
1. Fully rename all local parameters and logic routines.
2. Do not mirror any single reference layout. Heavy refactoring is strictly required.
3. Respond with NOTHING else except the code block inside a single ```java ... ``` block."""

        async with semaphore:
            ai_code = await generate_code_variant(client, prompt, language="java")

        if ai_code:
            async with state['lock']:
                state['records'].append({
                    "func": ai_code,
                    "functional_group": current_idx // 25
                })
                
                if len(state['records']) % 100 == 0:
                    pd.DataFrame(state['records']).to_parquet(OUTPUT_BCB, index=False)
            
            pbar.update(1)
        else:
            async with state['lock']:
                state['counter'] -= 1

# =====================================================
# Main Coroutine Entry Point
# =====================================================
async def main(num_files_needed=25000):
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

    all_java_functions = df_bcb["func1"].dropna().unique().tolist()

    shared_state = {
        'counter': start_index,
        'target': num_files_needed,
        'records': synthetic_records,
        'lock': asyncio.Lock()
    }

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    client = AsyncClient()

    pbar = tqdm_asyncio(total=num_files_needed, initial=start_index, desc="Synthesizing Java Codes")

    tasks = [
        worker(client, all_java_functions, semaphore, shared_state, pbar)
        for _ in range(CONCURRENCY_LIMIT * 2)
    ]

    await asyncio.gather(*tasks)
    pbar.close()

    async with shared_state['lock']:
        pd.DataFrame(shared_state['records']).to_parquet(OUTPUT_BCB, index=False)
    print("BigCloneBench File Track Fully Completed and Secured!")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main(num_files_needed=25000))

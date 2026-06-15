# pip install datasets pyarrow fastparquet ollama tqdm pandas
import os
import random
import re
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from ollama import AsyncClient

LOCAL_DIR = "./Code_Datasets/"
os.makedirs(LOCAL_DIR, exist_ok=True)
OUTPUT_PYTHON = os.path.join(LOCAL_DIR, "synthetic_codenet_python.parquet")
MODEL_ID = "qwen2.5-coder" 
CONCURRENCY_LIMIT = 4  # Matches OLLAMA_NUM_PARALLEL

print(f"Initializing local Async Ollama with {MODEL_ID} model...")

def clean_bpe_artifacts(raw_text):
    if not isinstance(raw_text, str):
        return raw_text
    clean_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text.strip()

async def generate_code_variant(client, prompt_text, language="python"):
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
        # Silently absorb or log errors to avoid breaking the async loop
        return ""

async def worker(client, all_python_functions, semaphore, state, pbar, num_files_needed):
    """Worker task that continuously generates code until the target is met."""
    while True:
        # Check if target met under lock protection
        async with state['lock']:
            if state['counter'] >= num_files_needed:
                break
            # Reserve a slot index
            current_id = state['counter']
            state['counter'] += 1

        selected_samples = random.sample(all_python_functions, k=3)
        concatenated_block = ""
        for idx, func_sample in enumerate(selected_samples):
            safe_sample = str(func_sample)[:1000] # OOM Protection
            concatenated_block += f"\n--- Python Sample {idx+1} ---\n{safe_sample}\n"

        prompt = f"""You are an Expert Python Developer. Examine this cluster of Python functions written to solve similar tasks.

Cluster of Reference Python Implementations:
{concatenated_block}

Write a completely brand new, highly robust, and distinct Python function that satisfies the same functional objective.

Strict Operational Rules:
1. Use Pythonic conventions (list comprehensions, generators, type hints if applicable).
2. Fully rename all local parameters and logic routines.
3. Do not mirror any single reference layout. Heavy refactoring is strictly required.
4. Respond with NOTHING else except the code block inside a single ```python ... ``` block."""

        # Throttle concurrent API calls to Ollama
        async with semaphore:
            ai_code = await generate_code_variant(client, prompt, language="python")

        if ai_code:
            async with state['lock']:
                state['records'].append({
                    "code": ai_code,
                    "language": "python",
                    "id": current_id
                })
                # Checkpoint saving every 50 records
                if len(state['records']) % 50 == 0:
                    pd.DataFrame(state['records']).to_parquet(OUTPUT_PYTHON, index=False)
            
            pbar.update(1)
        else:
            # If failed, revert the counter so another worker attempts it
            async with state['lock']:
                state['counter'] -= 1

async def main(num_files_needed=25000):
    print("\n--- Running Local Python (CodeNet) Track ---")
    print("Downloading/Loading Project CodeNet Python split from HuggingFace...")
    
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

    # Thread-safe/Async-safe shared state object
    state = {
        'counter': start_index,
        'records': synthetic_records,
        'lock': asyncio.Lock()
    }

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    client = AsyncClient()
    
    pbar = tqdm_asyncio(total=num_files_needed, initial=start_index, desc="Synthesizing Python Codes")

    # Fire off multiple concurrent workers matching your concurrency limit
    tasks = [
        worker(client, all_python_functions, semaphore, state, pbar, num_files_needed)
        for _ in range(CONCURRENCY_LIMIT * 2) # Over-provision tasks slightly to prevent GPU downtime
    ]
    
    await asyncio.gather(*tasks)

    # Final save
    pd.DataFrame(state['records']).to_parquet(OUTPUT_PYTHON, index=False)
    pbar.close()
    print("Python Track Fully Completed and Secured!")

if __name__ == "__main__":
    # Fix for Windows loop error management if applicable
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main(num_files_needed=25000))

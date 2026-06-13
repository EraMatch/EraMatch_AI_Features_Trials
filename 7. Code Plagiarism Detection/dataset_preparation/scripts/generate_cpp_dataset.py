import os
import random
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
from ollama import AsyncClient
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_POJ = os.path.join(SCRIPT_DIR, "data/poj104_train.csv")
OUTPUT_POJ = os.path.join(SCRIPT_DIR, "data/synthetic_poj104.csv")

MODEL_ID = "qwen2.5-coder"
CONCURRENCY_LIMIT = 4  # Matches OLLAMA_NUM_PARALLEL configuration

print(f"Using local Async Ollama with {MODEL_ID} model...")

def clean_bpe_artifacts(raw_text):
    if not isinstance(raw_text, str):
        return raw_text
    clean_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text.strip()

async def generate_code_variant(client, prompt_text, language="cpp"):
    """
    Calls the local Ollama instance asynchronously to generate the code variant.
    """
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
        # Suppress individual runtime exceptions to prevent crashing the worker pool
        return ""

async def worker(client, all_human_codes, label, semaphore, state, inner_pbar):
    """
    Worker task that continuously grabs jobs for a specific problem label until target met.
    """
    while True:
        async with state['lock']:
            if state['current_generated'] >= state['target_variants']:
                break
            state['current_generated'] += 1

        selected_samples = random.sample(all_human_codes, k=min(3, len(all_human_codes)))

        concatenated_block = ""
        for idx, code_sample in enumerate(selected_samples):
            safe_sample = str(code_sample)[:1000]
            concatenated_block += f"\n--- Sample {idx+1} ---\n{safe_sample}\n"

        prompt = f"""You are an advanced AI Code Generator. Analyze the following cluster of C++ source codes written to solve the EXACT SAME algorithmic problem.

Cluster of Human Reference Solutions:
{concatenated_block}

Write a brand new, fully functional C++ source code that solves this identical problem using a completely distinct programming architecture, naming convention, and code layout.

Strict Operational Rules:
1. DO NOT reuse specific variable names or line-by-line formatting patterns from the references.
2. Fully refactor the logic.
3. Ensure the code is self-contained and clean.
4. Respond with NOTHING else except the code inside a single ```cpp ... ``` block."""

        # Throttle Ollama interactions to prevent server timeouts
        async with semaphore:
            ai_code = await generate_code_variant(client, prompt, language="cpp")

        if ai_code:
            async with state['lock']:
                state['records'].append({"code": ai_code, "label": label})
                
                # Checkpoint saving every 10 generations globally
                if len(state['records']) % 10 == 0:
                    pd.DataFrame(state['records']).to_csv(OUTPUT_POJ, index=False)
            
            inner_pbar.update(1)
        else:
            # If generation failed, revert the target counter
            async with state['lock']:
                state['current_generated'] -= 1

async def main(num_variants_per_problem=500):
    print("\n--- Running Local POJ-104 Track ---")

    if not os.path.exists(INPUT_POJ):
        print(f"🔴 Error: Cannot find input file at {INPUT_POJ}.")
        return

    df_poj = pd.read_csv(INPUT_POJ)
    grouped = list(df_poj.groupby("label"))

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

    # Share mutable data securely with async Locks
    shared_state = {
        'records': synthetic_records,
        'lock': asyncio.Lock()
    }

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    client = AsyncClient()

    for label, group in grouped:
        str_label = str(label)

        if str_label in completed_labels:
            continue

        existing_count = sum(1 for r in shared_state['records'] if str(r["label"]) == str_label)
        variants_to_generate = num_variants_per_problem - existing_count

        if variants_to_generate <= 0:
            continue

        all_human_codes = group["code"].dropna().tolist()
        print(f"\n[+] Problem ID: {label} | Found: {existing_count} existing | Remaining to generate: {variants_to_generate}")

        inner_pbar = tqdm_asyncio(total=variants_to_generate, desc=f"Problem {label}", leave=True)

        # Isolated tracking dict for this specific cluster loop
        problem_state = {
            'current_generated': 0,
            'target_variants': variants_to_generate,
            'records': shared_state['records'],
            'lock': shared_state['lock']
        }

        # Build execution tasks 
        tasks = [
            worker(client, all_human_codes, label, semaphore, problem_state, inner_pbar)
            for _ in range(CONCURRENCY_LIMIT * 2) # Slightly over-provision tasks to maximize batch streams
        ]

        await asyncio.gather(*tasks)
        inner_pbar.close()

        # Save at the end of each problem group
        async with shared_state['lock']:
            pd.DataFrame(shared_state['records']).to_csv(OUTPUT_POJ, index=False)
        print(f"[Secured] Problem {label} completely generated.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main(num_variants_per_problem=500))

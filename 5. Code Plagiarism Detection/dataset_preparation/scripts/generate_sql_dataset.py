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
OUTPUT_SQL_PLAGIARISM = os.path.join(LOCAL_DIR, "synthetic_sql_plagiarism_pairs.parquet")
MODEL_ID = "qwen2.5-coder" 
CONCURRENCY_LIMIT = 4  

print(f"Initializing local Async Ollama with {MODEL_ID} model...")

# =====================================================
# Mutation Strategies for Plagiarism
# =====================================================
MUTATION_STRATEGIES = [
    {
        "type": "alias_and_casing",
        "instruction": "Rename all table aliases and column aliases to entirely different, arbitrary names (e.g., 'data_tbl', 'x1'). Radically change the capitalization of SQL keywords (e.g., mix uppercase and lowercase inconsistently)."
    },
    {
        "type": "structural_rewrite",
        "instruction": "Change the structure drastically without altering the output logic. Convert any subqueries into Common Table Expressions (CTEs) or vice versa. Swap explicit JOINs (e.g., INNER JOIN x ON y) with implicit joins (e.g., FROM a, b WHERE a.id = b.id), or vice versa."
    },
    {
        "type": "logic_and_noise",
        "instruction": "Reorder the conditions in the WHERE, GROUP BY, or ORDER BY clauses. Inject harmless, redundant dummy logic that does not affect the output (e.g., adding 'AND 1=1' or 'AND id = id')."
    }
]

# =====================================================
# Helper Functions
# =====================================================
def clean_bpe_artifacts(raw_text):
    if not isinstance(raw_text, str):
        return raw_text
    clean_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text.strip()

async def generate_code_variant(client, prompt_text, language="sql"):
    try:
        response = await client.chat(model=MODEL_ID, messages=[
            {
                'role': 'user',
                'content': prompt_text
            }
        ], options={
            'temperature': 0.85, # Keep creativity reasonably high for diverse plagiarism
            'top_p': 0.95,
            'num_predict': 512
        })
        
        generated_text = clean_bpe_artifacts(response['message']['content'])
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
    except Exception:
        return ""

async def worker(client, all_sql_queries, semaphore, state, pbar, num_files_needed):
    while True:
        async with state['lock']:
            if state['counter'] >= num_files_needed:
                break
            current_id = state['counter']
            state['counter'] += 1

        # Select ONE original query to plagiarize
        original_sql = random.choice(all_sql_queries)
        
        # Select a random plagiarism strategy
        strategy = random.choice(MUTATION_STRATEGIES)

        prompt = f"""You are a clever software engineering candidate trying to copy a SQL solution but evade automated plagiarism detection. 
                Original Source Query:
                ```sql
                {original_sql}
                ```
                Your Task:
                Rewrite the query to satisfy the EXACT same functional requirements, but apply the following obfuscation strategy:
                STRATEGY: {strategy['instruction']}

                Strict Rules:

                The new query MUST remain valid SQL and return the exact same data as the original.

                Apply the requested obfuscation strategy heavily.

                Respond with NOTHING else except the modified code inside a single sql ...  block. No explanations."""

        async with semaphore:
            plagiarized_code = await generate_code_variant(client, prompt, language="sql")

        if plagiarized_code and plagiarized_code != original_sql:
            async with state['lock']:
                state['records'].append({
                    "id": current_id,
                    "original_sql": original_sql,
                    "plagiarized_sql": plagiarized_code,
                    "mutation_type": strategy['type']
                })

                if len(state['records']) % 50 == 0:
                    pd.DataFrame(state['records']).to_parquet(OUTPUT_SQL_PLAGIARISM, index=False)
            pbar.update(1)
        else:
            async with state['lock']:
                state['counter'] -= 1

async def main(num_files_needed=25000):
        print("\n--- Running Plagiarism Data Synthesis Track ---")
        dataset = load_dataset("b-mc2/sql-create-context", split="train")
        all_sql_queries = dataset['answer']

        if os.path.exists(OUTPUT_SQL_PLAGIARISM):
            print("Found existing Plagiarism Parquet checkpoint! Resuming...")
            df_existing = pd.read_parquet(OUTPUT_SQL_PLAGIARISM)
            synthetic_records = df_existing.to_dict("records")
            start_index = len(synthetic_records)
        else:
            print("Starting fresh Plagiarism task.")
            synthetic_records = []
            start_index = 0

        state = {
            'counter': start_index,
            'records': synthetic_records,
            'lock': asyncio.Lock()
        }

        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        client = AsyncClient()
        pbar = tqdm_asyncio(total=num_files_needed, initial=start_index, desc="Synthesizing Plagiarism Pairs")

        tasks = [
            worker(client, all_sql_queries, semaphore, state, pbar, num_files_needed)
            for _ in range(CONCURRENCY_LIMIT * 2) 
        ]

        await asyncio.gather(*tasks)

        pd.DataFrame(state['records']).to_parquet(OUTPUT_SQL_PLAGIARISM, index=False)
        pbar.close()

print("Plagiarism Track Fully Completed and Secured!")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main(num_files_needed=10000)) # Adjust target volume as needed

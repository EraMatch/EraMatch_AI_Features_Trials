# =========================================================================================
# OLLAMA ENVIRONMENT SETUP STEPS
# =========================================================================================
# 1. Open a fresh PowerShell terminal.
# 2. Configure Ollama for maximum parallel processing on your RTX 5080:
#    $env:OLLAMA_NUM_PARALLEL="4"
# 3. Start the model in this terminal:
#    ollama run qwen2.5-coder
# 4. Open a SEPARATE terminal, ensure dependencies are installed:
#    pip install datasets pyarrow fastparquet ollama tqdm pandas
# 5. Launch this script:
#    python master_synthesis_pipeline.py
# =========================================================================================

import os
import re
import asyncio
import pandas as pd

from datasets import load_dataset
from ollama import AsyncClient
from tqdm.asyncio import tqdm_asyncio

# =====================================================
# Configuration & Paths
# =====================================================

LOCAL_DIR = "./Code_Datasets/"
os.makedirs(LOCAL_DIR, exist_ok=True)

OUT_MBPP = os.path.join(LOCAL_DIR, "synthetic_mbpp_python.parquet")
OUT_SPIDER = os.path.join(LOCAL_DIR, "synthetic_spider_sql.parquet")
OUT_CC = os.path.join(LOCAL_DIR, "synthetic_codecontests_python.parquet")

MODEL_ID = "qwen2.5-coder"
CONCURRENCY_LIMIT = 4
CHECKPOINT_INTERVAL = 50
MAX_RETRIES = 3

print(f"Initializing Master Async Pipeline with {MODEL_ID}...")

# =====================================================
# Helper Functions
# =====================================================

def clean_bpe_artifacts(raw_text):
    if not isinstance(raw_text, str):
        return ""

    clean_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text.strip()


async def generate_code_variant(client, prompt_text, language):
    try:
        response = await client.chat(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            options={
                "temperature": 0.85,
                "top_p": 0.95,
                "num_predict": 512
            }
        )

        generated_text = clean_bpe_artifacts(
            response["message"]["content"]
        )

        extracted_code = generated_text

        if "```" in generated_text:
            lang_marker = f"```{language}"

            if lang_marker in generated_text:
                try:
                    extracted_code = (
                        generated_text
                        .split(lang_marker, 1)[1]
                        .split("```", 1)[0]
                        .strip()
                    )
                except Exception:
                    pass
            else:
                try:
                    extracted_code = (
                        generated_text
                        .split("```", 1)[1]
                        .split("```", 1)[0]
                        .strip()
                    )
                except Exception:
                    pass

        return extracted_code.strip()

    except Exception:
        return ""


# =====================================================
# Prompt Factories
# =====================================================

def make_mbpp_prompt(row):
    # Use 'prompt' instead of 'text' to match the sanitized MBPP dataset schema
    problem_description = row.get('prompt', row.get('text', ''))
    
    return f"""
You are an Expert Python Developer.

A candidate was given this coding task:

"{problem_description}"

Here is the standard baseline solution:

Python
{row['code']}

Write a highly obfuscated, semantically identical Python function that solves the exact same task but looks completely written by a different human.

Strict Rules:
- Change the algorithm approach if possible.
- Rename all variables and functions creatively.
- Respond ONLY with the code inside a single python code block.
"""


def make_spider_prompt(row):
    return f"""
You are an Expert Database Engineer.

A candidate was given this SQL task:

"{row['question']}"

Here is the standard baseline query:

SQL
{row['query']}

Write a semantically identical SQL query that returns the exact same results while using a substantially different structure.

Strict Rules:
- Use alternative joins, EXISTS, IN, CTEs, or subqueries where appropriate.
- Use different aliases.
- Respond ONLY with the query inside a single sql code block.
"""


def make_code_contests_prompt(row):
    description = str(row["description"])[:1500]

    solutions = (
        row.get("solutions", {})
        .get("solution", [])[:2]
    )

    sol_text = ""

    for idx, sol in enumerate(solutions):
        sol_text += (
            f"\n--- Reference Solution {idx + 1} ---\n"
            f"{str(sol)[:800]}\n"
        )

    return f"""
You are a Competitive Programming Expert.

Problem Description:

"{description}"

Reference solutions:

{sol_text}

Write a distinct Python 3 solution that solves the same problem.

Strict Rules:
- Change control flow where possible.
- Use different variable names.
- Use different data structures when appropriate.
- Respond ONLY with the code inside a single python code block.
"""


# =====================================================
# Dataset Processing Track
# =====================================================

async def run_track(
    track_name,
    dataset,
    prompt_func,
    language,
    output_file,
    max_records=None
):
    print(f"\n--- Starting Track: {track_name} ---")

    dataset_rows = list(dataset)

    target_count = (
        min(max_records, len(dataset_rows))
        if max_records
        else len(dataset_rows)
    )

    if os.path.exists(output_file):
        df_existing = pd.read_parquet(output_file)
        synthetic_records = df_existing.to_dict("records")

        start_idx = len(synthetic_records)

        print(
            f"Found existing Parquet! "
            f"Resuming from index {start_idx} "
            f"out of {target_count}..."
        )
    else:
        synthetic_records = []
        start_idx = 0

    if start_idx >= target_count:
        print(f"✅ {track_name} already completed.")
        return

    queue = asyncio.Queue()

    for idx, row in enumerate(dataset_rows[start_idx:target_count]):
        row_copy = dict(row)

        row_copy["internal_idx"] = start_idx + idx
        row_copy["retry_count"] = 0

        queue.put_nowait(row_copy)

    shared_state = {
        "records": synthetic_records,
        "lock": asyncio.Lock()
    }

    client = AsyncClient()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    pbar = tqdm_asyncio(
        total=target_count,
        initial=start_idx,
        desc=f"Synthesizing {track_name}"
    )

    async def track_worker():
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            prompt = prompt_func(row)

            async with semaphore:
                ai_code = await generate_code_variant(
                    client,
                    prompt,
                    language
                )

            if ai_code:
                async with shared_state["lock"]:

                    original_code = row.get(
                        "code",
                        row.get(
                            "query",
                            "Competitive Programming Solution"
                        )
                    )

                    shared_state["records"].append(
                        {
                            "id": row["internal_idx"],
                            "original_code": str(original_code),
                            "ai_variant_code": ai_code,
                            "language": language
                        }
                    )

                    if (
                        len(shared_state["records"])
                        % CHECKPOINT_INTERVAL
                        == 0
                    ):
                        pd.DataFrame(
                            shared_state["records"]
                        ).to_parquet(
                            output_file,
                            index=False
                        )

                pbar.update(1)

            else:
                row["retry_count"] += 1

                if row["retry_count"] < MAX_RETRIES:
                    queue.put_nowait(row)

                else:
                    print(
                        f"\nFailed permanently: "
                        f"{row['internal_idx']}"
                    )

                    pbar.update(1)

    tasks = [
        asyncio.create_task(track_worker())
        for _ in range(CONCURRENCY_LIMIT * 2)
    ]

    await asyncio.gather(*tasks)

    pd.DataFrame(
        shared_state["records"]
    ).to_parquet(
        output_file,
        index=False
    )

    pbar.close()

    print(f"Track {track_name} Fully Completed!")


# =====================================================
# Main Execution Flow
# =====================================================

async def main():
    print("\n[1/3] Loading Datasets from Hugging Face...")

    # MBPP supports split="all" safely to merge its small splits
    ds_mbpp = load_dataset(
        "google-research-datasets/mbpp",
        "sanitized",
        split="all"
    )

    # For Spider, manually combine train and validation splits
    from datasets import concatenate_datasets
    spider_tr = load_dataset("xlangai/spider", split="train")
    spider_va = load_dataset("xlangai/spider", split="validation")
    ds_spider = concatenate_datasets([spider_tr, spider_va])

    ds_cc = load_dataset(
        "deepmind/code_contests",
        split="train"
    )

    # MBPP
    await run_track(
        track_name="MBPP (Basic Python)",
        dataset=ds_mbpp,
        prompt_func=make_mbpp_prompt,
        language="python",
        output_file=OUT_MBPP
    )

    # Spider
    await run_track(
        track_name="Spider (SQL)",
        dataset=ds_spider,
        prompt_func=make_spider_prompt,
        language="sql",
        output_file=OUT_SPIDER
    )

    # Code Contests
    await run_track(
        track_name="Code Contests (Algorithms)",
        dataset=ds_cc,
        prompt_func=make_code_contests_prompt,
        language="python",
        output_file=OUT_CC,
        max_records=10000
    )


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )

    asyncio.run(main())

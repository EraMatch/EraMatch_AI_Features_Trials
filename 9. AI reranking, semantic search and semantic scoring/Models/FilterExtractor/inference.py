"""
Run inference with the fine-tuned filter extraction model.

Usage:
    # Interactive mode
    python inference.py --model_path ./qwen_filter_extractor

    # Single query
    python inference.py --model_path ./qwen_filter_extractor --query "need sr python dev 5yrs remote"

    # Batch from a text file (one query per line)
    python inference.py --model_path ./qwen_filter_extractor --input queries.txt --output results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert HR filter extractor. "
    "Given a natural language job requirement query, extract structured filters as a valid JSON object.\n\n"
    "Output ONLY valid JSON. Include only fields that are clearly present in the query. Common fields:\n"
    "  role, skills, preferred_skills, min_experience_years, remote_ok, timezone, location,\n"
    "  employment_type, seniority_level, industry, start_date, visa_sponsorship,\n"
    "  certifications, education_level, no_c2c, citizenship_required, contract_duration,\n"
    "  direct_hire, experience_domain, security_clearance, startup_experience"
)

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 512


def load_model(model_path: str):
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig
    import torch

    print(f"Loading native Hugging Face model from {model_path} ...")
    
    # Load the tokenizer from your saved directory
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Wrap the 4-bit requirement in a dedicated config object
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    
    # AutoPeftModel reads your adapter_config.json, automatically fetches 
    # the base Qwen2.5 model, and applies your fine-tuned LoRA weights.
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        quantization_config=bnb_config,  # <--- The fix
    )
    
    print("Model ready.\n")
    return model, tokenizer


def extract_filters(query: str, model, tokenizer) -> dict:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, # Pure greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON block if model added surrounding text
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"_raw_output": raw, "_parse_error": True}


def interactive_mode(model, tokenizer):
    print("Interactive mode — type a query and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue
        filters = extract_filters(query, model, tokenizer)
        print(json.dumps(filters, indent=2, ensure_ascii=False))
        print()


def batch_mode(input_file: str, output_file: str, model, tokenizer):
    queries = Path(input_file).read_text(encoding="utf-8").splitlines()
    queries = [q.strip() for q in queries if q.strip()]
    print(f"Processing {len(queries)} queries ...")

    with open(output_file, "w", encoding="utf-8") as out:
        for i, query in enumerate(queries, 1):
            filters = extract_filters(query, model, tokenizer)
            record = {"query": query, "filters": filters}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(queries)}] {query[:60]}...")

    print(f"Results written to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Filter extraction inference")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--query", default=None, help="Single query to process")
    parser.add_argument("--input", default=None, help="Input file with queries (one per line)")
    parser.add_argument("--output", default="results.jsonl", help="Output JSONL file for batch mode")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model_path)

    if args.query:
        filters = extract_filters(args.query, model, tokenizer)
        print(json.dumps(filters, indent=2, ensure_ascii=False))
    elif args.input:
        batch_mode(args.input, args.output, model, tokenizer)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
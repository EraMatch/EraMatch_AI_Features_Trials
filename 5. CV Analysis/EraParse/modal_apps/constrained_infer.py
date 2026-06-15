"""Track 1 / RQ3 — Constrained decoding inference with Outlines + JSON grammar.

Tests whether forcing JSON validity (constrained decoding) helps or hurts
content accuracy and faithfulness vs unconstrained generation from the same
fine-tuned adapter.

Two modes:
  --constrained     grammar-guided generation (JSON schema constraint)
  --unconstrained   standard greedy generation (for direct A/B comparison)

Both run the same adapter so the only variable is decoding strategy.

Output: same ingest-mapper compatible JSONL as other inference apps, with
an extra `constrained: true/false` field per row.

Run:
    modal run modal_apps/constrained_infer.py \\
        --requests-path artifacts/trials/router/validation.gemma-ft.requests.jsonl \\
        --output-path artifacts/trials/ft/validation.gemma-constrained.jsonl \\
        --adapter-name gemma3-1b-reduced \\
        --constrained
"""
import json
import os
import time
from pathlib import Path

import modal

GPU = "L4"
TIMEOUT = 2 * 60 * 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "outlines>=0.2.0",
        "transformers>=4.50",
        "peft>=0.14",
        "accelerate>=1.0",
        "bitsandbytes>=0.43",
        "torch==2.5.1",
        "safetensors",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("constrained-infer", image=image)
vol = modal.Volume.from_name("eraparse-adapters")

# JSON schema for Outlines to constrain against — matches REDUCED_SCHEMA_TEMPLATE
REDUCED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "full_name":    {"type": "string"},
        "email":        {"type": "string"},
        "location":     {"type": "string"},
        "phone":        {"type": "string"},
        "linkedin_url": {"type": ["string", "null"]},
        "github_url":   {"type": ["string", "null"]},
        "summary":      {"type": "string"},
        "skills":       {"type": "array", "items": {"type": "string"}},
        "work_experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "job_title": {"type": "string"},
                    "company":   {"type": "string"},
                    "start_date":{"type": "string"},
                    "end_date":  {"type": "string"},
                    "duration":  {"type": "string"},
                },
                "required": ["job_title", "company", "start_date", "end_date", "duration"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree":          {"type": "string"},
                    "field_of_study":  {"type": "string"},
                    "institution":     {"type": "string"},
                    "graduation_date": {"type": "string"},
                },
                "required": ["degree", "field_of_study", "institution", "graduation_date"],
            },
        },
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":         {"type": "string"},
                    "technologies": {"type": "array", "items": {"type": "string"}},
                    "url":          {"type": "string"},
                },
                "required": ["name", "technologies", "url"],
            },
        },
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":   {"type": "string"},
                    "issuer": {"type": "string"},
                    "date":   {"type": "string"},
                },
                "required": ["name", "issuer", "date"],
            },
        },
    },
    "required": ["full_name", "email", "location", "phone", "summary", "skills",
                 "work_experience", "education", "projects", "certifications"],
}


@app.cls(gpu=GPU, volumes={"/adapters": vol}, timeout=TIMEOUT)
class Worker:
    base_model: str = modal.parameter(default="unsloth/gemma-3-1b-it")
    adapter_name: str = modal.parameter(default="gemma3-1b-reduced")
    use_constrained: bool = modal.parameter(default=True)
    is_nuextract: bool = modal.parameter(default=False)

    @modal.enter()
    def load(self):
        import torch
        import outlines
        from transformers import AutoTokenizer
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.use_constrained = self.use_constrained

        self.tok = AutoTokenizer.from_pretrained(self.base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        adapter_path = f"/adapters/{self.adapter_name}"
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        merged = PeftModel.from_pretrained(base, adapter_path)

        if self.use_constrained:
            # Outlines wraps the model for grammar-guided generation
            self.generator = outlines.generate.json(
                outlines.models.Transformers(merged, self.tok),
                REDUCED_JSON_SCHEMA,
            )
        else:
            self.model = merged.eval()

        print(f"loaded; constrained={self.use_constrained}")

    @modal.method()
    def predict(self, requests: list[dict], max_new_tokens: int = 1200) -> list[dict]:
        out = []
        for item in requests:
            if self.is_nuextract:
                prompt = item.get("prompt") or item.get("text")
            else:
                msgs = [
                    {"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["text"]},
                ]
                prompt = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )

            t0 = time.perf_counter()
            if self.use_constrained:
                result = self.generator(prompt, max_tokens=max_new_tokens)
                raw = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
            else:
                enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
                n_in = int(enc.input_ids.shape[-1])
                with self.torch.inference_mode():
                    gen = self.model.generate(
                        **enc, do_sample=False, max_new_tokens=max_new_tokens,
                        pad_token_id=self.tok.pad_token_id,
                    )
                raw = self.tok.batch_decode(gen[:, n_in:], skip_special_tokens=True)[0].strip()

            out.append({
                "cv_id": item["cv_id"],
                "raw_output": raw,
                "latency_seconds": time.perf_counter() - t0,
                "model_id": f"eraparse/{self.adapter_name}",
                "revision": "ft-v1",
                "parser_id": "pymupdf4llm_markdown",
                "constrained": self.use_constrained,
            })
        return out


@app.local_entrypoint()
def main(
    requests_path: str,
    output_path: str,
    base_model: str = "unsloth/gemma-3-1b-it",
    adapter_name: str = "gemma3-1b-reduced",
    constrained: bool = True,
    is_nuextract: bool = False,
    chunk_size: int = 20,
):
    reqs = [json.loads(x) for x in Path(requests_path).read_text().splitlines() if x.strip()]

    done = set()
    if Path(output_path).exists():
        for line in Path(output_path).read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["cv_id"])
    reqs = [r for r in reqs if r["cv_id"] not in done]
    print(f"running {len(reqs)} CVs (constrained={constrained})")

    if not reqs:
        print("all done")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    w = Worker(
        base_model=base_model,
        adapter_name=adapter_name,
        use_constrained=constrained,
        is_nuextract=is_nuextract,
    )

    with open(output_path, "a") as fh:
        for i in range(0, len(reqs), chunk_size):
            chunk = reqs[i: i + chunk_size]
            print(f"chunk {i // chunk_size + 1} | {len(done) + i + 1}–{len(done) + i + len(chunk)}")
            results = w.predict.remote(chunk)
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    total = sum(1 for _ in Path(output_path).read_text().splitlines() if _.strip())
    print(f"done: {total} responses -> {output_path}")

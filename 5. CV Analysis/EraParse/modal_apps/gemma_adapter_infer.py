"""Run a fine-tuned Gemma-3 LoRA adapter as a mapper; emit ingest-mapper rows.

Output format: one JSON line per CV with cv_id, raw_output, latency_seconds,
model_id, revision, parser_id — compatible with `eraparse trials ingest-mapper`.

Supports resume: already-written cv_ids are skipped so the retry-loop in the
launch script can recover from Modal heartbeat disconnects.

Run (after training):
    bash scripts/run_gemma_ft_infer.sh
"""
import json
import os
import time
from pathlib import Path

import modal

GPU = "L4"
TIMEOUT = 2 * 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.50",
        "peft>=0.14",
        "accelerate>=1.0",
        "bitsandbytes>=0.43",
        "torch==2.5.1",
        "safetensors",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("gemma-adapter-infer", image=image)
vol = modal.Volume.from_name("eraparse-adapters")


@app.cls(gpu=GPU, volumes={"/adapters": vol}, timeout=TIMEOUT)
class Worker:
    base_model: str = modal.parameter(default="unsloth/gemma-3-1b-it")
    adapter_name: str = modal.parameter(default="gemma3-1b-reduced")

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.torch = torch
        print(f"loading tokenizer from {self.base_model}")
        self.tok = AutoTokenizer.from_pretrained(self.base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        adapter_path = f"/adapters/{self.adapter_name}"
        print(f"loading base model {self.base_model} then adapter {adapter_path}")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, adapter_path).eval()
        print("model ready")

    @modal.method()
    def predict(self, requests: list[dict], max_new_tokens: int = 1200) -> list[dict]:
        out = []
        for item in requests:
            msgs = [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["text"]},
            ]
            prompt = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            enc = self.tok(
                prompt, return_tensors="pt", truncation=True, max_length=8192
            ).to("cuda")
            n_in = int(enc.input_ids.shape[-1])
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                gen = self.model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
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
            })
        return out


@app.local_entrypoint()
def main(
    requests_path: str,
    output_path: str,
    base_model: str = "unsloth/gemma-3-1b-it",
    adapter_name: str = "gemma3-1b-reduced",
    chunk_size: int = 25,
):
    reqs = [json.loads(x) for x in Path(requests_path).read_text().splitlines() if x.strip()]

    done = set()
    if Path(output_path).exists():
        for line in Path(output_path).read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["cv_id"])
    reqs = [r for r in reqs if r["cv_id"] not in done]
    print(f"running {len(reqs)} CVs (skipped {len(done)} already done)")

    if not reqs:
        print("all done")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    w = Worker(base_model=base_model, adapter_name=adapter_name)

    with open(output_path, "a") as fh:
        for i in range(0, len(reqs), chunk_size):
            chunk = reqs[i : i + chunk_size]
            done_so_far = len(done) + i
            print(f"chunk {i//chunk_size + 1} | CVs {done_so_far+1}–{done_so_far+len(chunk)}")
            results = w.predict.remote(chunk)
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    total = sum(1 for _ in Path(output_path).read_text().splitlines() if _.strip())
    print(f"done: {total} responses written to {output_path}")

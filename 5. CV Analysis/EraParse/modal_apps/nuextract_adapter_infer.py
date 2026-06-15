"""Inference with a fine-tuned NuExtract-1.5-tiny or NuExtract-1.5 LoRA adapter.

Uses NuExtract's <|input|>...<|output|> format (not chat), matching the SFT format.
Works for both tiny (0.5B) and full (3.8B) adapters — pass adapter_name to switch.

Output: one JSON line per CV, compatible with `eraparse trials ingest-mapper`.

Run:
    modal run modal_apps/nuextract_adapter_infer.py \\
        --requests-path artifacts/trials/router/validation.nuextract.requests.jsonl \\
        --output-path artifacts/trials/ft/validation.nuextract-tiny-ft.jsonl \\
        --adapter-name nuextract-tiny-reduced
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

app = modal.App("nuextract-adapter-infer", image=image)
vol = modal.Volume.from_name("eraparse-adapters")


@app.cls(gpu=GPU, volumes={"/adapters": vol}, timeout=TIMEOUT)
class Worker:
    base_model: str = modal.parameter(default="numind/NuExtract-1.5-tiny")
    adapter_name: str = modal.parameter(default="nuextract-tiny-reduced")

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        adapter_path = f"/adapters/{self.adapter_name}"
        print(f"loading {self.base_model} + adapter {adapter_path}")
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
            # NuExtract format: the "prompt" field already contains <|input|>...<|output|>
            prompt = item.get("prompt") or item.get("text")
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
            raw = self.tok.batch_decode(
                gen[:, n_in:], skip_special_tokens=False
            )[0]
            # strip <|end-output|> if present
            raw = raw.split("<|end-output|>")[0].strip()
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
    base_model: str = "numind/NuExtract-1.5-tiny",
    adapter_name: str = "nuextract-tiny-reduced",
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
            chunk = reqs[i: i + chunk_size]
            done_so_far = len(done) + i
            print(f"chunk {i // chunk_size + 1} | CVs {done_so_far + 1}–{done_so_far + len(chunk)}")
            results = w.predict.remote(chunk)
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    total = sum(1 for _ in Path(output_path).read_text().splitlines() if _.strip())
    print(f"done: {total} responses written to {output_path}")

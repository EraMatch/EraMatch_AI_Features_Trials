import json
import os
import time
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_CACHE = "/models"

app = modal.App("eraparse-qwen3-trial")
model_volume = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "safetensors==0.6.2",
    )
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

EXTRACTION_INSTRUCTION = """Extract the CV into exactly the JSON schema shown below.
Return only one valid JSON object. Use empty strings or empty arrays for missing values.
Do not add fields or explanations.

Schema:
{schema}

CV text:
{text}
"""


@app.cls(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class Qwen3Worker:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                dtype=torch.bfloat16,
            )
            .to("cuda")
            .eval()
        )
        model_volume.commit()

    @modal.method()
    def predict(self, requests: list[dict], max_new_tokens: int = 1_600) -> list[dict]:
        outputs: list[dict] = []
        for index, item in enumerate(requests, start=1):
            instruction = EXTRACTION_INSTRUCTION.format(
                schema=json.dumps(item["schema"], indent=2),
                text=item["text"],
            )
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=10_000,
            ).to("cuda")
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            self.torch.cuda.synchronize()
            response_ids = generated[:, encoded.input_ids.shape[1] :]
            outputs.append(
                {
                    "cv_id": item["cv_id"],
                    "raw_output": self.tokenizer.batch_decode(
                        response_ids, skip_special_tokens=True
                    )[0],
                    "latency_seconds": time.perf_counter() - started,
                    "input_tokens": int(encoded.input_ids.shape[-1]),
                    "output_tokens": int(response_ids.shape[-1]),
                }
            )
            print(f"processing {index}/{len(requests)}")
        return outputs


@app.local_entrypoint()
def main(
    requests_path: str,
    output_path: str,
    max_new_tokens: int = 1_600,
    chunk_size: int = 50,
    parallel_chunks: int = 4,
) -> None:
    from eraparse.trials import chunk_pending_requests

    requests = [
        json.loads(line)
        for line in Path(requests_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schema = {
        "full_name": "",
        "email": "",
        "location": "",
        "phone": "",
        "linkedin_url": "",
        "github_url": "",
        "summary": "",
        "skills": [],
        "work_experience": [
            {
                "job_title": "",
                "company": "",
                "start_date": "",
                "end_date": "",
                "duration": "",
            }
        ],
        "education": [
            {
                "degree": "",
                "field_of_study": "",
                "institution": "",
                "graduation_date": "",
            }
        ],
        "projects": [{"name": "", "technologies": [], "url": ""}],
        "certifications": [{"name": "", "issuer": "", "date": ""}],
    }
    remote_requests = [
        {"cv_id": row["cv_id"], "text": row["text"], "schema": schema} for row in requests
    ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()
    if destination.is_file():
        completed_ids = {
            str(json.loads(line)["cv_id"])
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    chunks = chunk_pending_requests(remote_requests, completed_ids, chunk_size=chunk_size)
    worker = Qwen3Worker()
    for wave_start in range(0, len(chunks), parallel_chunks):
        wave = chunks[wave_start : wave_start + parallel_chunks]
        calls = []
        for offset, chunk in enumerate(wave, start=wave_start + 1):
            print(f"spawning remote chunk {offset}/{len(chunks)} ({len(chunk)} requests)")
            calls.append(worker.predict.spawn(chunk, max_new_tokens=max_new_tokens))
        for call in calls:
            responses = call.get()
            with destination.open("a", encoding="utf-8") as handle:
                for response in responses:
                    handle.write(json.dumps(response) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

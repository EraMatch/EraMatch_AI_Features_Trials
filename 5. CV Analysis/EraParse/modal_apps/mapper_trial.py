import json
import os
import re
import time
from pathlib import Path
from typing import Any

import modal

MODEL_CACHE = "/models"
SUPPORTED_MODELS = {
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "microsoft/Phi-4-mini-instruct",
}
COMMIT_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
REDUCED_SCHEMA_TEMPLATE = {
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

app = modal.App("eraparse-mapper-trial")
model_volume = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
    )
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

phi_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.49.0",
        "accelerate==1.3.0",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
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
Do not add fields, explanations, or unsupported values.

Schema:
{schema}

CV text:
{text}
"""


def validate_model_configuration(model_id: str, revision: str) -> None:
    if model_id not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"unsupported model_id {model_id!r}; choose one of: {supported}")
    if not COMMIT_REVISION.fullmatch(revision.strip()):
        raise ValueError("revision must be an immutable 40-character Hugging Face commit SHA")


def build_instruction(request: dict[str, Any]) -> str:
    return EXTRACTION_INSTRUCTION.format(
        schema=json.dumps(
            request.get("schema", REDUCED_SCHEMA_TEMPLATE),
            indent=2,
            ensure_ascii=False,
        ),
        text=request["text"],
    )


def chat_template_kwargs(model_id: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_id.startswith("Qwen/"):
        kwargs["enable_thinking"] = False
    return kwargs


def model_dtype_keyword(model_id: str) -> str:
    return "torch_dtype" if model_id == "microsoft/Phi-4-mini-instruct" else "dtype"


def load_pending_chunks(
    requests_path: Path,
    output_path: Path,
    *,
    model_id: str,
    revision: str,
    chunk_size: int,
    max_records: int,
) -> list[list[dict[str, Any]]]:
    from eraparse.trials import chunk_pending_requests

    requests = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_records:
        requests = requests[:max_records]

    completed_ids: set[str] = set()
    if output_path.is_file():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model_id") == model_id and row.get("revision") == revision:
                completed_ids.add(str(row["cv_id"]))
    return chunk_pending_requests(requests, completed_ids, chunk_size=chunk_size)


class _MapperWorker:
    model_id: str = modal.parameter()
    revision: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        validate_model_configuration(self.model_id, self.revision)
        trust_remote_code = self.model_id == "microsoft/Phi-4-mini-instruct"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=trust_remote_code,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=trust_remote_code,
                **{model_dtype_keyword(self.model_id): torch.bfloat16},
            )
            .to("cuda")
            .eval()
        )
        model_volume.commit()

    @modal.method()
    def predict(
        self,
        requests: list[dict[str, Any]],
        max_new_tokens: int = 1_600,
        max_input_tokens: int = 10_000,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for index, item in enumerate(requests, start=1):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": build_instruction(item)}],
                **chat_template_kwargs(self.model_id),
            )
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            ).to("cuda")
            input_tokens = int(encoded.input_ids.shape[-1])
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            self.torch.cuda.synchronize()
            response_ids = generated[:, input_tokens:]
            outputs.append(
                {
                    "cv_id": item["cv_id"],
                    "raw_output": self.tokenizer.batch_decode(
                        response_ids, skip_special_tokens=True
                    )[0].strip(),
                    "latency_seconds": time.perf_counter() - started,
                    "input_tokens": input_tokens,
                    "output_tokens": int(response_ids.shape[-1]),
                    "input_truncated": input_tokens >= max_input_tokens,
                    "model_id": self.model_id,
                    "revision": self.revision,
                    "parser_id": item.get("parser_id", item.get("representation")),
                    "generation": {
                        "do_sample": False,
                        "max_new_tokens": max_new_tokens,
                        "max_input_tokens": max_input_tokens,
                        "thinking_enabled": False
                        if self.model_id.startswith("Qwen/")
                        else None,
                    },
                }
            )
            print(f"processing {index}/{len(requests)}")
        return outputs


@app.cls(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class QwenMapperWorker(_MapperWorker):
    model_id: str = modal.parameter()
    revision: str = modal.parameter()


@app.cls(
    image=phi_image,
    gpu="A10",
    volumes={MODEL_CACHE: model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class PhiMapperWorker(_MapperWorker):
    model_id: str = modal.parameter()
    revision: str = modal.parameter()


@app.local_entrypoint()
def main(
    requests_path: str,
    output_path: str,
    model_id: str,
    revision: str,
    max_new_tokens: int = 1_600,
    max_input_tokens: int = 10_000,
    chunk_size: int = 25,
    max_records: int = 0,
    parallel_chunks: int = 2,
) -> None:
    validate_model_configuration(model_id, revision)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks = load_pending_chunks(
        Path(requests_path),
        destination,
        model_id=model_id,
        revision=revision,
        chunk_size=chunk_size,
        max_records=max_records,
    )

    worker_class = (
        PhiMapperWorker if model_id == "microsoft/Phi-4-mini-instruct" else QwenMapperWorker
    )
    worker = worker_class(model_id=model_id, revision=revision)
    if parallel_chunks == 1:
        for index, chunk in enumerate(chunks, start=1):
            print(f"running remote chunk {index}/{len(chunks)} ({len(chunk)} requests)")
            responses = worker.predict.remote(
                chunk,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
            )
            with destination.open("a", encoding="utf-8") as handle:
                for response in responses:
                    handle.write(json.dumps(response, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return

    for wave_start in range(0, len(chunks), parallel_chunks):
        wave = chunks[wave_start : wave_start + parallel_chunks]
        calls = []
        for offset, chunk in enumerate(wave, start=wave_start + 1):
            print(f"spawning remote chunk {offset}/{len(chunks)} ({len(chunk)} requests)")
            calls.append(
                worker.predict.spawn(
                    chunk,
                    max_new_tokens=max_new_tokens,
                    max_input_tokens=max_input_tokens,
                )
            )
        for call in calls:
            responses = call.get()
            with destination.open("a", encoding="utf-8") as handle:
                for response in responses:
                    handle.write(json.dumps(response, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

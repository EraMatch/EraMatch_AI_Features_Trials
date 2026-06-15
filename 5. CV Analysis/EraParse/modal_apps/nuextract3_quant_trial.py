import json
import os
import time
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any

import modal

MODELS = {
    "w8a8": {
        "model_id": "numind/NuExtract3-W8A8",
        "revision": "e9ffaea6c5cbf2bed066dcc6b193fb608b8bdcf7",
    },
    "w4a16": {
        "model_id": "numind/NuExtract3-W4A16",
        "revision": "b5028670152c8130a3f362b66981eee16612b7f6",
    },
}
VLLM_VERSION = "0.21.0"
MODEL_CACHE = "/models"
MAX_MODEL_LEN = 16_384
MAX_IMAGES_PER_PROMPT = 6
GPU_MEMORY_UTILIZATION = 0.90
COMPLETE_SCHEMA_KEYS = {
    "full_name",
    "email",
    "location",
    "phone",
    "summary",
    "linkedin_url",
    "github_url",
    "skills",
    "work_experience",
    "education",
    "projects",
    "certifications",
}
RUNTIME_ENV = {
    "HF_HUB_CACHE": MODEL_CACHE,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    # FlashInfer sampling has caused model-specific failures; use the stable
    # PyTorch sampler so quantization is the only inference variable.
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}

app = modal.App("eraparse-nuextract3-quant-trial")
model_volume = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "pillow==11.3.0")
    .env(RUNTIME_ENV)
)


def model_spec(variant: str) -> dict[str, str]:
    try:
        return MODELS[variant]
    except KeyError as error:
        supported = ", ".join(sorted(MODELS))
        raise ValueError(
            f"unsupported quantization variant {variant!r}; choose one of: {supported}"
        ) from error


def require_complete_schema(schema_template: dict[str, Any]) -> None:
    if set(schema_template) != COMPLETE_SCHEMA_KEYS:
        raise ValueError("quantization trials require the complete reduced schema")


def engine_config(variant: str) -> dict[str, Any]:
    spec = model_spec(variant)
    return {
        "model": spec["model_id"],
        "revision": spec["revision"],
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "limit_mm_per_prompt": {"image": MAX_IMAGES_PER_PROMPT, "video": 0},
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "generation_config": "vllm",
    }


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "runtime-provided"


def json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def serializable_reason(reason: Any) -> str | int | None:
    if reason is None:
        return None
    if isinstance(reason, int):
        return int(reason)
    return str(reason)


def request_latency_or_chunk_share(
    request_latency: float | None, chunk_latency: float, request_count: int
) -> float:
    return request_latency if request_latency is not None else chunk_latency / request_count


def run_config(*, variant: str, max_new_tokens: int, chunk_size: int) -> dict[str, Any]:
    spec = model_spec(variant)
    return {
        "runner": "nuextract3_quant_trial",
        "variant": variant,
        "model_id": spec["model_id"],
        "model_revision": spec["revision"],
        "vllm_version": VLLM_VERSION,
        "gpu": "A10",
        "native_template": True,
        "complete_schema": True,
        "enable_thinking": False,
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "chunk_size": chunk_size,
        "sampler_backend": "pytorch",
        "mtp": False,
        "engine": engine_config(variant),
    }


def metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.metadata.json")


def ensure_resume_config(output_path: Path, config: dict[str, Any]) -> None:
    path = metadata_path(output_path)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["configuration"] != config:
            raise RuntimeError(f"refusing to resume {output_path} with a different configuration")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            {"configuration": config, "created_unix_seconds": time.time()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def completed_ids(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    return {
        str(json.loads(line)["cv_id"])
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


class _NuExtract3QuantWorker:
    variant = ""

    @modal.enter()
    def load(self) -> None:
        from transformers import AutoProcessor
        from vllm import LLM

        spec = model_spec(self.variant)
        config = engine_config(self.variant)
        self.processor = AutoProcessor.from_pretrained(
            spec["model_id"],
            revision=spec["revision"],
            trust_remote_code=True,
        )
        self.llm = LLM(**config)
        self.runtime_metadata = {
            "model_id": spec["model_id"],
            "model_revision": spec["revision"],
            "variant": self.variant,
            "engine": config,
            "packages": {
                package: package_version(package)
                for package in ("vllm", "torch", "transformers", "modal")
            },
        }
        model_volume.commit()

    def _prepare_request(self, item: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        require_complete_schema(item["schema_template"])
        images = [Image.open(BytesIO(page)).convert("RGB") for page in item["page_images"]]
        if not images:
            raise ValueError(f"{item['cv_id']} has no page images")
        if len(images) > MAX_IMAGES_PER_PROMPT:
            raise ValueError(
                f"{item['cv_id']} has {len(images)} pages; maximum is {MAX_IMAGES_PER_PROMPT}"
            )
        content = [{"type": "image", "image": page} for page in images]
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
            template=json.dumps(item["schema_template"], ensure_ascii=False),
            enable_thinking=False,
        )
        return {"prompt": prompt, "multi_modal_data": {"image": images}}

    @modal.method()
    def predict(
        self, requests: list[dict[str, Any]], max_new_tokens: int = 2_048
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        prepared = [self._prepare_request(item) for item in requests]
        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        started = time.perf_counter()
        generated = self.llm.generate(prepared, sampling_params=sampling, use_tqdm=False)
        chunk_latency = time.perf_counter() - started
        outputs: list[dict[str, Any]] = []
        for item, result in zip(requests, generated, strict=True):
            candidate = result.outputs[0]
            metrics = result.metrics
            first_token_time = getattr(metrics, "first_token_time", None)
            finished_time = getattr(metrics, "finished_time", None)
            request_latency = (
                float(finished_time - metrics.arrival_time)
                if finished_time is not None
                else None
            )
            request_latency = request_latency_or_chunk_share(
                request_latency, chunk_latency, len(requests)
            )
            outputs.append(
                {
                    "cv_id": item["cv_id"],
                    "raw_output": candidate.text.strip(),
                    "model_id": self.runtime_metadata["model_id"],
                    "model_revision": self.runtime_metadata["model_revision"],
                    "variant": self.variant,
                    "input_tokens": len(result.prompt_token_ids),
                    "output_tokens": len(candidate.token_ids),
                    "finish_reason": serializable_reason(candidate.finish_reason),
                    "stop_reason": serializable_reason(candidate.stop_reason),
                    "latency_seconds": request_latency,
                    "request_latency_seconds": request_latency,
                    "time_to_first_token_seconds": (
                        float(first_token_time - metrics.arrival_time)
                        if first_token_time is not None
                        else None
                    ),
                    "chunk_latency_seconds": chunk_latency,
                    "chunk_size": len(requests),
                }
            )
        return json_roundtrip(
            {
                "outputs": outputs,
                "chunk": {
                    "latency_seconds": chunk_latency,
                    "request_count": len(requests),
                    "input_tokens": sum(row["input_tokens"] for row in outputs),
                    "output_tokens": sum(row["output_tokens"] for row in outputs),
                },
                "runtime": self.runtime_metadata,
            }
        )


@app.cls(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class NuExtract3W8A8Worker(_NuExtract3QuantWorker):
    variant = "w8a8"


@app.cls(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class NuExtract3W4A16Worker(_NuExtract3QuantWorker):
    variant = "w4a16"


@app.local_entrypoint()
def main(
    requests_path: str,
    dataset_root: str,
    output_path: str,
    variant: str,
    max_new_tokens: int = 2_048,
    chunk_size: int = 4,
    max_records: int = 0,
) -> None:
    from eraparse.trials import chunk_pending_requests

    model_spec(variant)
    requests = [
        json.loads(line)
        for line in Path(requests_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in requests:
        require_complete_schema(row["schema_template"])
    root = Path(dataset_root)
    remote_requests = [
        {
            "cv_id": row["cv_id"],
            "schema_template": row["schema_template"],
            "page_images": [
                (root / relative_path).read_bytes() for relative_path in row["page_images"]
            ],
        }
        for row in requests
    ]
    if max_records:
        remote_requests = remote_requests[:max_records]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_resume_config(
        destination,
        run_config(variant=variant, max_new_tokens=max_new_tokens, chunk_size=chunk_size),
    )
    chunks = chunk_pending_requests(
        remote_requests, completed_ids(destination), chunk_size=chunk_size
    )
    worker = NuExtract3W8A8Worker() if variant == "w8a8" else NuExtract3W4A16Worker()
    for index, chunk in enumerate(chunks, start=1):
        print(f"running remote chunk {index}/{len(chunks)} ({len(chunk)} requests)")
        response = worker.predict.remote(chunk, max_new_tokens=max_new_tokens)
        with destination.open("a", encoding="utf-8") as handle:
            for output in response["outputs"]:
                output["runtime"] = response["runtime"]
                handle.write(json.dumps(json_roundtrip(output)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

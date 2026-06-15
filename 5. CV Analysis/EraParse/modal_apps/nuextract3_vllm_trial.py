import json
import os
import time
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "numind/NuExtract3"
MODEL_REVISION = "acaf70ecff9c3dbbfcbae651b82b66a0d8dbd0c6"
MODEL_CACHE = "/models"
# vLLM 0.14.0 predates the Qwen3.5 architecture used by NuExtract3.
VLLM_VERSION = "0.21.0"
MAX_MODEL_LEN = 16_384
MAX_IMAGES_PER_PROMPT = 6
GPU_MEMORY_UTILIZATION = 0.90
MTP_SPECULATIVE_CONFIG = {
    "method": "qwen3_next_mtp",
    "num_speculative_tokens": 2,
}

app = modal.App("eraparse-nuextract3-vllm-trial")
model_volume = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        "pillow==11.3.0",
    )
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)


def engine_config(*, use_mtp: bool) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "limit_mm_per_prompt": {"image": MAX_IMAGES_PER_PROMPT, "video": 0},
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "generation_config": "vllm",
        "speculative_config": MTP_SPECULATIVE_CONFIG if use_mtp else None,
    }


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "runtime-provided"


def request_latency_or_chunk_share(
    request_latency: float | None, chunk_latency: float, request_count: int
) -> float:
    return request_latency if request_latency is not None else chunk_latency / request_count


def serializable_reason(reason: Any) -> str | int | None:
    if reason is None:
        return None
    if isinstance(reason, int):
        return int(reason)
    return str(reason)


def run_config(
    *,
    use_mtp: bool,
    max_new_tokens: int,
    include_evidence_text: bool,
    chunk_size: int,
) -> dict[str, Any]:
    return {
        "runner": "nuextract3_vllm_trial",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vllm_version": VLLM_VERSION,
        "variant": "mtp" if use_mtp else "baseline",
        "native_template": True,
        "enable_thinking": False,
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "include_evidence_text": include_evidence_text,
        "chunk_size": chunk_size,
        "sampler_backend": "pytorch",
        "engine": engine_config(use_mtp=use_mtp),
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
            {
                "configuration": config,
                "created_unix_seconds": time.time(),
            },
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


class _NuExtract3VLLMWorker:
    use_mtp = False

    @modal.enter()
    def load(self) -> None:
        from transformers import AutoProcessor
        from vllm import LLM

        config = engine_config(use_mtp=self.use_mtp)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        self.llm = LLM(**config)
        self.runtime_metadata = {
            "engine": config,
            "packages": {
                package: package_version(package)
                for package in ("vllm", "torch", "transformers", "modal")
            },
        }
        model_volume.commit()

    def _prepare_request(
        self,
        item: dict[str, Any],
        *,
        include_evidence_text: bool,
    ) -> dict[str, Any]:
        from PIL import Image

        images = [Image.open(BytesIO(page)).convert("RGB") for page in item["page_images"]]
        if not images:
            raise ValueError(f"{item['cv_id']} has no page images")
        if len(images) > MAX_IMAGES_PER_PROMPT:
            raise ValueError(
                f"{item['cv_id']} has {len(images)} pages; maximum is {MAX_IMAGES_PER_PROMPT}"
            )
        content: list[dict[str, Any]] = [{"type": "image", "image": page} for page in images]
        if include_evidence_text and item.get("evidence_text"):
            content.append({"type": "text", "text": item["evidence_text"]})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
            template=json.dumps(item["schema_template"], ensure_ascii=False),
            enable_thinking=False,
        )
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        }

    @modal.method()
    def predict(
        self,
        requests: list[dict[str, Any]],
        max_new_tokens: int = 2_048,
        include_evidence_text: bool = False,
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        prepared = [
            self._prepare_request(item, include_evidence_text=include_evidence_text)
            for item in requests
        ]
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
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
                float(finished_time - result.metrics.arrival_time)
                if finished_time is not None
                else None
            )
            request_latency = request_latency_or_chunk_share(
                request_latency, chunk_latency, len(requests)
            )
            time_to_first_token = (
                float(first_token_time - result.metrics.arrival_time)
                if first_token_time is not None
                else None
            )
            outputs.append(
                {
                    "cv_id": item["cv_id"],
                    "raw_output": candidate.text.strip(),
                    "input_tokens": len(result.prompt_token_ids),
                    "output_tokens": len(candidate.token_ids),
                    "finish_reason": serializable_reason(candidate.finish_reason),
                    "stop_reason": serializable_reason(candidate.stop_reason),
                    "latency_seconds": request_latency,
                    "request_latency_seconds": request_latency,
                    "time_to_first_token_seconds": time_to_first_token,
                    "chunk_latency_seconds": chunk_latency,
                    "chunk_size": len(requests),
                    "variant": "mtp" if self.use_mtp else "baseline",
                }
            )
        response = {
            "outputs": outputs,
            "chunk": {
                "latency_seconds": chunk_latency,
                "request_count": len(requests),
                "input_tokens": sum(row["input_tokens"] for row in outputs),
                "output_tokens": sum(row["output_tokens"] for row in outputs),
            },
            "runtime": self.runtime_metadata,
        }
        return json.loads(json.dumps(response, default=str))


@app.cls(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class NuExtract3VLLMBaselineWorker(_NuExtract3VLLMWorker):
    use_mtp = False


@app.cls(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class NuExtract3VLLMMTPWorker(_NuExtract3VLLMWorker):
    use_mtp = True


@app.local_entrypoint()
def main(
    requests_path: str,
    dataset_root: str,
    output_path: str,
    use_mtp: bool = False,
    max_new_tokens: int = 2_048,
    chunk_size: int = 4,
    max_records: int = 0,
    include_evidence_text: bool = False,
) -> None:
    from eraparse.trials import chunk_pending_requests

    requests = [
        json.loads(line)
        for line in Path(requests_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root = Path(dataset_root)
    remote_requests = [
        {
            "cv_id": row["cv_id"],
            "schema_template": row["schema_template"],
            "evidence_text": row.get("evidence_text"),
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
    configuration = run_config(
        use_mtp=use_mtp,
        max_new_tokens=max_new_tokens,
        include_evidence_text=include_evidence_text,
        chunk_size=chunk_size,
    )
    ensure_resume_config(destination, configuration)
    chunks = chunk_pending_requests(
        remote_requests,
        completed_ids(destination),
        chunk_size=chunk_size,
    )
    worker = NuExtract3VLLMMTPWorker() if use_mtp else NuExtract3VLLMBaselineWorker()
    for index, chunk in enumerate(chunks, start=1):
        print(f"running remote chunk {index}/{len(chunks)} ({len(chunk)} requests)")
        response = worker.predict.remote(
            chunk,
            max_new_tokens=max_new_tokens,
            include_evidence_text=include_evidence_text,
        )
        with destination.open("a", encoding="utf-8") as handle:
            for output in response["outputs"]:
                output["runtime"] = response["runtime"]
                handle.write(json.dumps(output) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "numind/NuExtract3"
MODEL_REVISION = "acaf70ecff9c3dbbfcbae651b82b66a0d8dbd0c6"
MODEL_CACHE = "/models"

app = modal.App("eraparse-nuextract3-trial")
model_volume = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers==5.8.1",
        "accelerate==1.12.0",
        "safetensors==0.6.2",
        "pillow==11.3.0",
    )
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

EXTRACTION_INSTRUCTION = """Extract the CV into exactly one JSON object that matches this schema.
Return only valid JSON with the same keys and nested structure.
Use empty strings or empty arrays when evidence is missing.
Do not use null for required string fields such as dates, durations, titles,
companies, institutions, summaries, names, email, phone, or location.
Only optional URL fields may be null.

Schema:
{schema}
"""


def _build_instruction(schema_template: dict[str, Any], evidence_text: str | None) -> str:
    instruction = EXTRACTION_INSTRUCTION.format(
        schema=json.dumps(schema_template, indent=2, ensure_ascii=False)
    )
    if evidence_text:
        instruction += f"\nAuxiliary text extracted from the PDF:\n{evidence_text}\n"
    return instruction


@app.cls(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class NuExtract3Worker:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                torch_dtype=torch.bfloat16,
            )
            .to("cuda")
            .eval()
        )
        model_volume.commit()

    @modal.method()
    def predict(
        self,
        requests: list[dict[str, Any]],
        max_new_tokens: int = 2_048,
        include_evidence_text: bool = False,
    ) -> list[dict[str, Any]]:
        from PIL import Image

        outputs: list[dict[str, Any]] = []
        for index, item in enumerate(requests, start=1):
            content = [
                {"type": "image", "image": Image.open(BytesIO(page)).convert("RGB")}
                for page in item["page_images"]
            ]
            content.append(
                {
                    "type": "text",
                    "text": _build_instruction(
                        item["schema_template"],
                        item.get("evidence_text") if include_evidence_text else None,
                    ),
                }
            )
            messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            self.torch.cuda.synchronize()
            response_ids = generated[:, inputs["input_ids"].shape[-1] :]
            raw_output = self.processor.batch_decode(response_ids, skip_special_tokens=True)[0]
            outputs.append(
                {
                    "cv_id": item["cv_id"],
                    "raw_output": raw_output.strip(),
                    "latency_seconds": time.perf_counter() - started,
                    "input_tokens": int(inputs["input_ids"].shape[-1]),
                    "output_tokens": int(response_ids.shape[-1]),
                }
            )
            print(f"processing {index}/{len(requests)}")
        return outputs


@app.local_entrypoint()
def main(
    requests_path: str,
    dataset_root: str,
    output_path: str,
    max_new_tokens: int = 2_048,
    chunk_size: int = 10,
    max_records: int = 0,
    parallel_chunks: int = 2,
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
    completed_ids = set()
    if destination.is_file():
        completed_ids = {
            str(json.loads(line)["cv_id"])
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    chunks = chunk_pending_requests(remote_requests, completed_ids, chunk_size=chunk_size)
    worker = NuExtract3Worker()
    for wave_start in range(0, len(chunks), parallel_chunks):
        wave = chunks[wave_start : wave_start + parallel_chunks]
        calls = []
        for offset, chunk in enumerate(wave, start=wave_start + 1):
            print(f"spawning remote chunk {offset}/{len(chunks)} ({len(chunk)} requests)")
            calls.append(
                worker.predict.spawn(
                    chunk,
                    max_new_tokens=max_new_tokens,
                    include_evidence_text=include_evidence_text,
                )
            )
        for call in calls:
            responses = call.get()
            with destination.open("a", encoding="utf-8") as handle:
                for response in responses:
                    handle.write(json.dumps(response) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

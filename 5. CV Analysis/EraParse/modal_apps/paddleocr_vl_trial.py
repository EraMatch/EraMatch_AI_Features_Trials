import json
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path

import modal

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"

app = modal.App("eraparse-paddleocr-vl-trial")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/",
        "python -m pip install -U 'paddleocr[doc-parser]>=3.6.0' pillow==11.3.0",
    )
)


@app.cls(
    image=image,
    gpu="A10",
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class PaddleOCRVLWorker:
    @modal.enter()
    def load(self) -> None:
        from paddleocr import PaddleOCRVL

        self.pipeline = PaddleOCRVL(pipeline_version="v1.6")

    def _page_json(self, result) -> dict | list | None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result.save_to_json(save_path=temporary_directory)
            json_paths = sorted(Path(temporary_directory).rglob("*.json"))
            if not json_paths:
                return None
            return json.loads(json_paths[0].read_text(encoding="utf-8"))

    @modal.method()
    def predict(self, requests: list[dict]) -> list[dict]:
        from PIL import Image

        outputs: list[dict] = []
        for index, item in enumerate(requests, start=1):
            started = time.perf_counter()
            try:
                markdown_pages = []
                page_json = []
                for page_index, page_bytes in enumerate(item["page_images"], start=1):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                        Image.open(BytesIO(page_bytes)).convert("RGB").save(handle.name)
                        image_path = handle.name
                    try:
                        page_outputs = list(self.pipeline.predict(image_path))
                    finally:
                        Path(image_path).unlink(missing_ok=True)
                    for result in page_outputs:
                        markdown_info = getattr(result, "markdown", None)
                        if markdown_info is not None:
                            markdown_pages.append(markdown_info)
                        page_json.append(
                            {
                                "page": page_index,
                                "result": self._page_json(result),
                            }
                        )
                markdown = (
                    self.pipeline.concatenate_markdown_pages(markdown_pages)
                    if markdown_pages
                    else ""
                )
                outputs.append(
                    {
                        "cv_id": item["cv_id"],
                        "markdown": markdown,
                        "page_json": page_json,
                        "latency_seconds": time.perf_counter() - started,
                        "model_id": MODEL_ID,
                    }
                )
            except Exception as error:
                outputs.append({"cv_id": item["cv_id"], "error": str(error)})
            print(f"processing {index}/{len(requests)}")
        return outputs


@app.local_entrypoint()
def main(
    requests_path: str,
    dataset_root: str,
    output_path: str,
    chunk_size: int = 10,
    max_records: int = 0,
    parallel_chunks: int = 2,
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
    worker = PaddleOCRVLWorker()
    for wave_start in range(0, len(chunks), parallel_chunks):
        wave = chunks[wave_start : wave_start + parallel_chunks]
        calls = []
        for offset, chunk in enumerate(wave, start=wave_start + 1):
            print(f"spawning remote chunk {offset}/{len(chunks)} ({len(chunk)} requests)")
            calls.append(worker.predict.spawn(chunk))
        for call in calls:
            responses = call.get()
            with destination.open("a", encoding="utf-8") as handle:
                for response in responses:
                    handle.write(json.dumps(response) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

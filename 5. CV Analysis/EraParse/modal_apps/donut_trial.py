import json
import os
import time
from pathlib import Path

import modal

TASK_PROMPT = "<s_eraparse>"
MODEL_CACHE = "/models"
DATA_ROOT = Path("/data/eraparse")
CHECKPOINT_ROOT = Path("/checkpoints")

app = modal.App("eraparse-donut-trial")
dataset_volume = modal.Volume.from_name("eraparse-dataset", create_if_missing=True)
model_volume = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("eraparse-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "safetensors==0.6.2",
        "pillow==11.3.0",
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


@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={
        "/data": dataset_volume,
        MODEL_CACHE: model_volume,
        str(CHECKPOINT_ROOT): checkpoint_volume,
    },
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class DonutWorker:
    checkpoint_name: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        checkpoint = CHECKPOINT_ROOT / self.checkpoint_name
        self.torch = torch
        self.processor = DonutProcessor.from_pretrained(checkpoint, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(checkpoint).to("cuda").eval()
        self.task_ids = self.processor.tokenizer(
            TASK_PROMPT,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to("cuda")

    def _compose_pages(self, relative_paths: list[str]):
        from PIL import Image

        pages = [Image.open(DATA_ROOT / path).convert("RGB") for path in relative_paths]
        width = max(page.width for page in pages)
        resized = [
            page
            if page.width == width
            else page.resize(
                (width, round(page.height * width / page.width)),
                Image.Resampling.LANCZOS,
            )
            for page in pages
        ]
        canvas = Image.new(
            "RGB",
            (width, sum(page.height for page in resized) + 16 * (len(resized) - 1)),
            "white",
        )
        top = 0
        for page in resized:
            canvas.paste(page, (0, top))
            top += page.height + 16
        return canvas

    @modal.method()
    def predict(
        self,
        requests: list[dict],
        max_length: int = 1536,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
    ) -> list[dict]:
        outputs = []
        for index, item in enumerate(requests, start=1):
            image = self._compose_pages(item["page_images"])
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to("cuda")
            self.torch.cuda.synchronize()
            total_started = time.perf_counter()
            with self.torch.inference_mode():
                encoder_started = time.perf_counter()
                encoder_outputs = self.model.encoder(pixel_values)
                self.torch.cuda.synchronize()
                encoder_latency = time.perf_counter() - encoder_started
                decoder_started = time.perf_counter()
                generated = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=self.task_ids,
                    max_length=max_length,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,
                    use_cache=True,
                )
                self.torch.cuda.synchronize()
                decoder_latency = time.perf_counter() - decoder_started
            generated_sequence = self.processor.batch_decode(generated, skip_special_tokens=False)[
                0
            ]
            raw_output = generated_sequence
            native_decode_error = None
            cleaned = generated_sequence.replace(self.processor.tokenizer.eos_token, "")
            cleaned = cleaned.replace(self.processor.tokenizer.pad_token, "")
            cleaned = cleaned.replace(TASK_PROMPT, "", 1).strip()
            if "<s_" in cleaned:
                try:
                    raw_output = json.dumps(self.processor.token2json(cleaned), ensure_ascii=False)
                except Exception as error:  # Preserve native decoding failures for evaluation.
                    native_decode_error = str(error)
            outputs.append(
                {
                    "cv_id": item["cv_id"],
                    "raw_output": raw_output,
                    "generated_sequence": generated_sequence,
                    "native_decode_error": native_decode_error,
                    "latency_seconds": time.perf_counter() - total_started,
                    "encoder_latency_seconds": encoder_latency,
                    "decoder_latency_seconds": decoder_latency,
                    "visual_tokens": int(encoder_outputs.last_hidden_state.shape[1]),
                    "output_tokens": int(generated.shape[-1]),
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                    "repetition_penalty": repetition_penalty,
                }
            )
            print(f"processing {index}/{len(requests)}")
        return outputs


@app.local_entrypoint()
def main(
    records_path: str,
    output_path: str,
    checkpoint_name: str,
    max_length: int = 1536,
    chunk_size: int = 10,
    max_records: int = 0,
    parallel_chunks: int = 4,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
) -> None:
    records = [
        json.loads(line)
        for line in Path(records_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()
    if destination.is_file():
        completed_ids = {
            str(json.loads(line)["cv_id"])
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line
        }
    pending = [record for record in records if record["cv_id"] not in completed_ids]
    if max_records:
        pending = pending[:max_records]
    worker = DonutWorker(checkpoint_name=checkpoint_name)
    chunks = [pending[start : start + chunk_size] for start in range(0, len(pending), chunk_size)]
    for wave_start in range(0, len(chunks), parallel_chunks):
        calls = []
        for chunk in chunks[wave_start : wave_start + parallel_chunks]:
            requests = [
                {"cv_id": record["cv_id"], "page_images": record["page_images"]} for record in chunk
            ]
            calls.append(
                worker.predict.spawn(
                    requests,
                    max_length=max_length,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,
                )
            )
        for call in calls:
            responses = call.get()
            with destination.open("a", encoding="utf-8") as handle:
                for response in responses:
                    handle.write(json.dumps(response) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

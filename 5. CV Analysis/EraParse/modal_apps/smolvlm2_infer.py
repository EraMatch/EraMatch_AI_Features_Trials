"""Run fine-tuned SmolVLM2 LoRA adapter on validation CVs; emit ingest-mapper rows.

Output: one JSON line per CV with cv_id, raw_output, latency_seconds,
model_id, revision, parser_id — compatible with `eraparse trials ingest-mapper`.

Resume support: already-written cv_ids are skipped.

Run:
    modal run modal_apps/smolvlm2_infer.py
"""
import json
import time
from pathlib import Path

import modal

GPU = "A100"
TIMEOUT = 4 * 60 * 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "libgl1")
    .pip_install(
        "unsloth[huggingface]",
        "unsloth_zoo",
        "datasets>=3.0",
        "Pillow",
        "sentencepiece",
        "protobuf",
        "torchvision",
        "num2words",
        "peft>=0.14",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("smolvlm2-infer", image=image)
vol = modal.Volume.from_name("eraparse-adapters")

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ADAPTER_NAME = "smolvlm2-cv-reduced"


@app.cls(gpu=GPU, volumes={"/adapters": vol}, timeout=TIMEOUT)
class Worker:
    @modal.enter()
    def load(self):
        import torch
        from PIL import Image as PILImage
        from transformers import AutoProcessor
        from unsloth import FastVisionModel
        from peft import PeftModel

        self.torch = torch
        self.PILImage = PILImage

        print(f"loading base model {MODEL_ID}")
        model, tokenizer = FastVisionModel.from_pretrained(
            MODEL_ID,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        adapter_path = f"/adapters/{ADAPTER_NAME}"
        print(f"loading adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(model, adapter_path).eval()
        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        if not hasattr(self.processor, "eos_token"):
            self.processor.eos_token = self.processor.tokenizer.eos_token
        print("model ready")

    @modal.method()
    def predict(self, requests: list[dict], max_new_tokens: int = 1200) -> list[dict]:
        out = []
        images_dir = Path("/adapters/page_images")

        for item in requests:
            # Load images
            imgs = []
            for p in item.get("image_paths", []):
                full = images_dir / Path(p).name
                if full.exists():
                    imgs.append(self.PILImage.open(full).convert("RGB"))

            convs = item["conversations"]
            if imgs:
                user_content = [{"type": "image"} for _ in imgs] + [
                    {"type": "text", "text": convs[1]["content"]}
                ]
            else:
                user_content = [{"type": "text", "text": convs[1]["content"]}]

            messages = [
                {"role": "system", "content": [{"type": "text", "text": convs[0]["content"]}]},
                {"role": "user", "content": user_content},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[imgs] if imgs else None,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
            ).to("cuda")

            t0 = time.perf_counter()
            with self.torch.inference_mode():
                gen = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            n_in = inputs["input_ids"].shape[-1]
            raw = self.processor.tokenizer.decode(
                gen[0][n_in:], skip_special_tokens=True
            ).strip()

            out.append({
                "cv_id": item["cv_id"],
                "raw_output": raw,
                "latency_seconds": time.perf_counter() - t0,
                "model_id": f"eraparse/{ADAPTER_NAME}",
                "revision": "ft-v1",
                "parser_id": "vision_smolvlm2",
            })
        return out


@app.local_entrypoint()
def main(
    val_sft: str = "artifacts/sft/validation.vision.sft.jsonl",
    output_path: str = "artifacts/trials/ft/validation.smolvlm2-ft.jsonl",
    chunk_size: int = 10,
):
    import os

    # Load val SFT — one example per cv_id (pick the first augmentation per cv)
    rows_raw = [json.loads(l) for l in Path(val_sft).read_text().splitlines() if l.strip()]
    # Deduplicate: keep first occurrence per cv_id
    seen = set()
    rows = []
    for r in rows_raw:
        cv_id = r["id"]
        if cv_id not in seen:
            r["cv_id"] = cv_id
            rows.append(r)
            seen.add(cv_id)
    print(f"unique CVs to infer: {len(rows)}")

    # Resume
    done = set()
    if Path(output_path).exists():
        for line in Path(output_path).read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["cv_id"])
    rows = [r for r in rows if r["cv_id"] not in done]
    print(f"remaining after resume: {len(rows)} (skipped {len(done)})")
    if not rows:
        print("all done")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    w = Worker()

    with open(output_path, "a") as fh:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i: i + chunk_size]
            done_so_far = len(done) + i
            print(f"chunk {i // chunk_size + 1} | CVs {done_so_far + 1}–{done_so_far + len(chunk)}")
            results = w.predict.remote(chunk)
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    total = sum(1 for _ in Path(output_path).read_text().splitlines() if _.strip())
    print(f"done: {total} responses written to {output_path}")

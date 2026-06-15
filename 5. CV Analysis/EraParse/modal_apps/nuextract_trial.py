import json
import time
from pathlib import Path

import modal

MODEL_ID = "numind/NuExtract-1.5-tiny"
MODEL_REVISION = "63e2e80c804d9c97f3f19a4aa25613e7beca83c9"
MODEL_CACHE = "/models"

app = modal.App("eraparse-nuextract-trial")
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


@app.cls(
    image=image,
    gpu="A10",
    volumes={MODEL_CACHE: model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class NuExtractWorker:
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
    def predict(
        self,
        requests: list[dict],
        batch_size: int = 1,
        max_new_tokens: int = 1_024,
    ) -> list[dict]:
        outputs: list[dict] = []
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            print(f"processing {start + 1}-{start + len(batch)} of {len(requests)}")
            prompts = [str(item["prompt"]) for item in batch]
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
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
            elapsed = time.perf_counter() - started
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for item, output, input_ids, generated_ids in zip(
                batch, decoded, encoded.input_ids, generated, strict=True
            ):
                outputs.append(
                    {
                        "cv_id": item["cv_id"],
                        "raw_output": output.split("<|output|>", maxsplit=1)[-1],
                        "latency_seconds": elapsed / len(batch),
                        "input_tokens": int(input_ids.ne(self.tokenizer.pad_token_id).sum().item()),
                        "output_tokens": int(generated_ids.shape[-1] - input_ids.shape[-1]),
                    }
                )
        return outputs


@app.local_entrypoint()
def main(
    requests_path: str,
    output_path: str,
    batch_size: int = 1,
    max_new_tokens: int = 1_024,
) -> None:
    requests = [
        json.loads(line)
        for line in Path(requests_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    remote_requests = [{"cv_id": row["cv_id"], "prompt": row["prompt"]} for row in requests]
    responses = NuExtractWorker().predict.remote(
        remote_requests,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        "".join(json.dumps(response) + "\n" for response in responses),
        encoding="utf-8",
    )

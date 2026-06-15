"""Benchmark Gemma-3-1B vLLM latency vs HF-generate baseline (~35s/CV on L4).

Uses the LoRA adapter stored in Modal volume 'eraparse-adapters' at
/adapters/gemma3-1b-reduced (r=16, alpha=16, safetensors format).

IMPORTANT — adapter compatibility note:
    The adapter was trained with Unsloth against 'unsloth/gemma-3-1b-it-unsloth-bnb-4bit'
    (a 4-bit quantised base). vLLM LoRA requires the adapter to be applied on top of a
    full-precision (fp16/bf16) base model. Using a different base may produce degraded
    outputs, but latency numbers are unaffected — which is what this benchmark measures.

    Two modes:
      --use-lora  : loads adapter on top of unsloth/gemma-3-1b-it (bf16)
                    → valid latency measure; output quality slightly off vs training base
      --no-use-lora: base model only → strict speed floor, no quality claim

Run (adapter mode):
    uv run modal run modal_apps/gemma_vllm_latency.py \\
        --requests-path artifacts/trials/router/validation.gemma-ft.requests.jsonl \\
        --output-path artifacts/benchmarks/gemma_vllm_latency.json \\
        --n-cvs 50

Base-only speed floor:
    uv run modal run modal_apps/gemma_vllm_latency.py \\
        --requests-path artifacts/trials/router/validation.gemma-ft.requests.jsonl \\
        --output-path artifacts/benchmarks/gemma_vllm_latency.json \\
        --n-cvs 50 --no-use-lora
"""
import json
import os
import statistics
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL = "unsloth/gemma-3-1b-it"
ADAPTER_NAME = "gemma3-1b-reduced"
ADAPTER_PATH = f"/adapters/{ADAPTER_NAME}"
LORA_RANK = 16
MAX_TOKENS = 1200
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.88

# vLLM 0.8.x has stable LoRA support for Gemma-3.
# 0.21.0 is what NuExtract3 uses; we use a slightly older stable release
# that is known to work well with Gemma architecture + LoRA.
VLLM_VERSION = "0.8.5"

HF_BASELINE_SECONDS = 35.0  # measured on L4, HF generate, adapter merged

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("eraparse-gemma-vllm-latency")

adapters_vol = modal.Volume.from_name("eraparse-adapters")
model_cache_vol = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)

MODEL_CACHE = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        "transformers>=4.50",
        "huggingface_hub>=0.24",
    )
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE,
            "TOKENIZERS_PARALLELISM": "false",
            # vLLM multiproc must use spawn on Modal
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            # Disable FlashInfer sampler — avoids a compile step on first run
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)


# ---------------------------------------------------------------------------
# Worker class
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="L4",  # same GPU as HF-generate baseline for fair comparison
    volumes={
        MODEL_CACHE: model_cache_vol,
        "/adapters": adapters_vol,
    },
    timeout=2 * 60 * 60,
    scaledown_window=5 * 60,
)
class GemmaVLLMWorker:
    use_lora: bool = modal.parameter(default=True)

    @modal.enter()
    def load(self) -> None:
        from vllm import LLM
        from transformers import AutoTokenizer

        print(f"loading vLLM engine: {BASE_MODEL} | use_lora={self.use_lora}")

        # Load tokenizer for apply_chat_template (same template used in training)
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        engine_kwargs: dict = {
            "model": BASE_MODEL,
            "dtype": "bfloat16",
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "trust_remote_code": True,
            "download_dir": MODEL_CACHE,
        }

        if self.use_lora:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_lora_rank"] = LORA_RANK

        self.llm = LLM(**engine_kwargs)
        self._lora_request = None

        if self.use_lora:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest(
                lora_name="gemma-cv-adapter",
                lora_int_id=1,
                lora_path=ADAPTER_PATH,
            )
            print(f"LoRA adapter registered from {ADAPTER_PATH}")
        else:
            print("running base model only (no LoRA)")

        print("engine ready")

    def _build_prompt(self, item: dict) -> str:
        """Build the chat-template prompt matching the training format."""
        msgs = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["text"]},
        ]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    @modal.method()
    def benchmark(
        self,
        requests: list[dict],
        max_new_tokens: int = MAX_TOKENS,
    ) -> dict:
        """Run requests one by one to get individual per-CV latencies.

        We do NOT batch here because the HF-generate baseline processes one
        CV at a time, so sequential throughput is the fair comparison unit.
        For a throughput comparison we also run a batched pass at the end.
        """
        from vllm import SamplingParams

        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

        results = []
        # ---- Sequential pass: per-CV latency --------------------------------
        for item in requests:
            prompt = self._build_prompt(item)
            t0 = time.perf_counter()
            outputs = self.llm.generate(
                [prompt],
                sampling_params=sampling,
                lora_request=self._lora_request,
                use_tqdm=False,
            )
            elapsed = time.perf_counter() - t0
            out = outputs[0]
            candidate = out.outputs[0]
            results.append(
                {
                    "cv_id": item["cv_id"],
                    "latency_seconds": elapsed,
                    "input_tokens": len(out.prompt_token_ids),
                    "output_tokens": len(candidate.token_ids),
                    "finish_reason": str(candidate.finish_reason),
                }
            )

        # ---- Batched pass: total throughput ---------------------------------
        all_prompts = [self._build_prompt(r) for r in requests]
        t_batch_start = time.perf_counter()
        batch_outputs = self.llm.generate(
            all_prompts,
            sampling_params=sampling,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        batch_elapsed = time.perf_counter() - t_batch_start

        total_output_tokens = sum(
            len(o.outputs[0].token_ids) for o in batch_outputs
        )

        return {
            "sequential_results": results,
            "batch": {
                "n_cvs": len(requests),
                "total_seconds": batch_elapsed,
                "total_output_tokens": total_output_tokens,
                "tokens_per_second": total_output_tokens / batch_elapsed if batch_elapsed > 0 else 0,
                "mean_latency_seconds": batch_elapsed / len(requests),
            },
            "use_lora": self.use_lora,
            "base_model": BASE_MODEL,
            "adapter_path": ADAPTER_PATH if self.use_lora else None,
            "vllm_version": VLLM_VERSION,
            "gpu": "L4",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_stats(latencies: list[float]) -> dict:
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    p50_idx = int(n * 0.50)
    p95_idx = int(n * 0.95)
    return {
        "n": n,
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p50": sorted_lat[min(p50_idx, n - 1)],
        "p95": sorted_lat[min(p95_idx, n - 1)],
        "min": min(latencies),
        "max": max(latencies),
        "stdev": statistics.stdev(latencies) if n > 1 else 0.0,
    }


def _print_summary(stats: dict, batch: dict, use_lora: bool) -> None:
    print()
    print("=" * 60)
    mode = "Gemma-3-1B + LoRA adapter" if use_lora else "Gemma-3-1B base (no adapter)"
    print(f"=== vLLM {mode} Latency Benchmark ({stats['n']} CVs) ===")
    print("=" * 60)
    print(f"  Sequential (1 CV at a time):")
    print(f"    P50:  {stats['p50']:.2f}s")
    print(f"    P95:  {stats['p95']:.2f}s")
    print(f"    Mean: {stats['mean']:.2f}s")
    print(f"    Min:  {stats['min']:.2f}s")
    print(f"    Max:  {stats['max']:.2f}s")
    print(f"    Stdev:{stats['stdev']:.2f}s")
    print()
    print(f"  Batched ({stats['n']} CVs together):")
    print(f"    Total wall time: {batch['total_seconds']:.2f}s")
    print(f"    Mean/CV:         {batch['mean_latency_seconds']:.2f}s")
    print(f"    Throughput:      {batch['tokens_per_second']:.1f} tok/s")
    print()
    speedup_seq = HF_BASELINE_SECONDS / stats["mean"]
    speedup_batch = HF_BASELINE_SECONDS / batch["mean_latency_seconds"]
    print(f"  vs HF-generate baseline ({HF_BASELINE_SECONDS}s/CV):")
    print(f"    Sequential speedup: {speedup_seq:.1f}x")
    print(f"    Batched speedup:    {speedup_batch:.1f}x")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    requests_path: str = "artifacts/trials/router/validation.gemma-ft.requests.jsonl",
    output_path: str = "artifacts/benchmarks/gemma_vllm_latency.json",
    n_cvs: int = 50,
    use_lora: bool = True,
) -> None:
    # Load requests
    all_reqs = [
        json.loads(line)
        for line in Path(requests_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reqs = all_reqs[:n_cvs]
    print(f"loaded {len(reqs)} CV requests from {requests_path}")
    print(f"use_lora={use_lora}, vLLM version={VLLM_VERSION}, GPU=L4")
    print("sending to Modal... (this will cold-start the engine on first run)")

    worker = GemmaVLLMWorker(use_lora=use_lora)
    result = worker.benchmark.remote(reqs, max_new_tokens=MAX_TOKENS)

    # Compute stats from sequential results
    latencies = [r["latency_seconds"] for r in result["sequential_results"]]
    stats = _compute_stats(latencies)
    batch = result["batch"]

    _print_summary(stats, batch, use_lora=result["use_lora"])

    # Build output payload
    speedup_seq = HF_BASELINE_SECONDS / stats["mean"]
    speedup_batch = HF_BASELINE_SECONDS / batch["mean_latency_seconds"]
    output = {
        "benchmark": "gemma_vllm_latency",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "base_model": result["base_model"],
            "adapter": result["adapter_path"],
            "vllm_version": result["vllm_version"],
            "gpu": result["gpu"],
            "max_new_tokens": MAX_TOKENS,
            "n_cvs": len(reqs),
            "use_lora": result["use_lora"],
        },
        "hf_baseline_seconds": HF_BASELINE_SECONDS,
        "sequential_stats": stats,
        "batch_stats": batch,
        "speedup": {
            "sequential_mean": round(speedup_seq, 2),
            "batched_mean": round(speedup_batch, 2),
        },
        "per_cv_results": result["sequential_results"],
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"results saved to {out_path}")

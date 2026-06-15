# Modal Execution Guide

Checked against Context7, official Modal documentation, and the local Modal
skill on 2026-06-09. Re-check version-sensitive syntax before implementation.

No remote GPU job, deployment, secret creation, or large upload may run unless
the user explicitly requests it.

## Intended Resource Split

- Local/CPU: dataset audit, manifests, evaluator, precomputed parsers, reports.
- Modal CPU: scalable parser work or data preparation when justified.
- Modal GPU: mapper inference, Donut fine-tuning/inference, SG-VTC,
  LayoutLMv3, and selected VLM upper bounds.

Start with measured needs. Current official GPU names include `T4`, `L4`,
`A10`, `L40S`, `A100`, `A100-40GB`, `A100-80GB`, `H100`, `H200`, and `B200`.
Use `A10`, not the historical `A10G` name, and verify availability/pricing.

## Compatibility-Lane Images

Use separate Modal images for:

- CPU parsers/evaluation;
- core Transformers 4 models;
- modern Transformers 5 VLMs;
- PaddleOCR/PaddlePaddle.

Do not build one monolithic image. Put stable dependencies before frequently
changed source layers for cache reuse.

The current local machine is an Apple M1 Pro with 16 GB unified memory. The
default repository development environment currently runs under Rosetta, but
an ignored native ARM64 Python 3.11 environment has verified Torch 2.7.1 MPS,
Transformers 4.57.3, and PyMuPDF4LLM 1.27.2.2.

Use local ARM64/MPS for evidence generation, smart OCR, architecture debugging,
and tiny-overfit trials. Use Modal only for promoted throughput/debug work that
passes local gates. Do not weaken pinned versions or replace the default
development environment merely to obtain native ML wheels.

## Current Trial Budget

As of June 12, 2026, the user authorized up to `$15-20` for the next controlled
trial stage. Repository budget guards use an `$18.00` stage hard stop, leaving
approximately `$2.00` of the maximum authorization as a reserve. Every paid
run must still use a measured runtime projection with the `1.30` safety factor;
the larger allowance is for a broader documented comparison matrix, not blind
full-data reruns.

## Base Pattern

```python
import modal

app = modal.App("eraparse")

core_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("transformers==4.57.3")
)

dataset_volume = modal.Volume.from_name(
    "eraparse-dataset", create_if_missing=True
)
model_volume = modal.Volume.from_name(
    "eraparse-model-cache", create_if_missing=True
)
result_volume = modal.Volume.from_name(
    "eraparse-results", create_if_missing=True
)
```

Actual implementation must add tested dependencies and explicit CPU, memory,
GPU, timeout, volumes, and secrets.

## Model Cache

Mount a named Volume and set Hugging Face cache variables:

```python
HF_CACHE_PATH = "/models/huggingface"
core_image = core_image.env(
    {
        "HF_HUB_CACHE": HF_CACHE_PATH,
        "HF_XET_HIGH_PERFORMANCE": "1",
    }
)
```

Use a named `modal.Secret` for `HF_TOKEN` only when required. Never embed or
print token values.

## Stateful Model Worker

```python
@app.cls(
    image=core_image,
    gpu="A10",
    volumes={
        "/models": model_volume,
        "/data": dataset_volume,
        "/results": result_volume,
    },
    timeout=60 * 60,
)
class MapperWorker:
    @modal.enter()
    def load(self):
        self.model = load_pinned_model()

    @modal.method()
    def predict(self, batch_ids: list[str]) -> list[dict]:
        rows = run_batch(self.model, batch_ids)
        write_results(rows)
        result_volume.commit()
        return summarize(rows)
```

Pass small IDs/configs to remote functions. Put datasets, weights, and large
outputs in Volumes or object storage.

## Volume Semantics

- Fresh containers mount the latest Volume state.
- Call `commit()` after writes that other jobs must see.
- Call `reload()` only when a long-lived container needs commits made by
  another container.
- Do not reload mechanically before every read.
- Avoid many concurrent commits and concurrent modifications to the same file.
- Make jobs idempotent because calls may be retried.

Write many small temporary files to container-local storage first, then copy
completed batches to the Volume and commit once.

For local-entrypoint batch inference, divide real-size runs into remote chunks
that each fit comfortably below the function timeout. Persist each completed
chunk locally before launching or waiting on later chunks. A single remote call
that returns an entire validation/test split is prohibited because timeout or
client interruption would discard all completed outputs.

## Run Modes

- `modal run`: one-off jobs and development.
- `modal serve`: iterative endpoint development.
- `modal deploy`: persistent endpoints or repeated call-many services.

Before execution:

1. verify local syntax and tests;
2. run `modal --version` and `modal <command> --help`;
3. validate app discovery and a cheap CPU smoke test;
4. confirm volumes, secret names, GPU profile, timeout, and expected cost;
5. require explicit approval for paid/remote work.

For fine-tuning, full-data execution additionally requires a tiny-set overfit
test with valid generated outputs. Falling training or validation loss alone is
not a promotion signal. Distinguish smoke, debug, selection, and final App names
or billing tags, and record actual cost after paid work with
`modal billing report`.

## Result Contract

Every remote run writes immutable, content-addressed outputs containing:

- resolved config and run ID;
- code, manifest, model, and environment revisions;
- sample-level predictions and errors;
- timing and resource metrics;
- batch completion markers;
- summary suitable for later local ingestion.

## Trial 3 Named Volumes

- `eraparse-dataset`: exact audited page images and generated Donut records;
- `eraparse-model-cache`: pinned Hugging Face base-model cache;
- `eraparse-checkpoints`: immutable Donut checkpoints and training summaries.

Do not substitute the older `cv-parsing-data` volume. A matching image filename
was verified to contain different bytes from the audited local source.

The Donut training app uploads only images referenced by the selected
manifests. It trains on `train.jsonl`, reads validation only for loss
measurement, and never accepts the locked manifest through the project record
builder.

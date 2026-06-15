import importlib.util
import json
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).parents[1] / "modal_apps" / "nuextract3_quant_trial.py"
SPEC = importlib.util.spec_from_file_location("nuextract3_quant_trial", RUNNER_PATH)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_only_official_quantized_checkpoints_are_supported() -> None:
    assert runner.MODELS == {
        "w8a8": {
            "model_id": "numind/NuExtract3-W8A8",
            "revision": "e9ffaea6c5cbf2bed066dcc6b193fb608b8bdcf7",
        },
        "w4a16": {
            "model_id": "numind/NuExtract3-W4A16",
            "revision": "b5028670152c8130a3f362b66981eee16612b7f6",
        },
    }
    with pytest.raises(ValueError, match="unsupported quantization variant"):
        runner.model_spec("bf16")


@pytest.mark.parametrize("variant", ["w8a8", "w4a16"])
def test_engine_config_is_native_vllm_without_mtp(variant: str) -> None:
    spec = runner.model_spec(variant)
    config = runner.engine_config(variant)

    assert runner.VLLM_VERSION == "0.21.0"
    assert config["model"] == spec["model_id"]
    assert config["revision"] == spec["revision"]
    assert config["generation_config"] == "vllm"
    assert "speculative_config" not in config


def test_complete_schema_guard_rejects_compact_schema() -> None:
    complete = {
        "full_name": "",
        "email": "",
        "location": "",
        "phone": "",
        "summary": "",
        "linkedin_url": None,
        "github_url": None,
        "skills": [],
        "work_experience": [],
        "education": [],
        "projects": [],
        "certifications": [],
    }
    assert "full_name" in complete
    assert "name" not in complete
    runner.require_complete_schema(complete)

    with pytest.raises(ValueError, match="complete reduced schema"):
        runner.require_complete_schema({"n": "", "e": ""})


def test_resume_config_records_model_and_rejects_changed_variant(tmp_path: Path) -> None:
    output = tmp_path / "responses.jsonl"
    config = runner.run_config(variant="w8a8", max_new_tokens=512, chunk_size=4)
    runner.ensure_resume_config(output, config)
    runner.ensure_resume_config(output, config)

    stored = json.loads(runner.metadata_path(output).read_text(encoding="utf-8"))
    assert stored["configuration"]["model_id"] == "numind/NuExtract3-W8A8"
    assert stored["configuration"]["model_revision"] == runner.MODELS["w8a8"]["revision"]
    with pytest.raises(RuntimeError, match="different configuration"):
        runner.ensure_resume_config(
            output,
            runner.run_config(variant="w4a16", max_new_tokens=512, chunk_size=4),
        )


def test_completed_ids_supports_resumable_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "responses.jsonl"
    output.write_text('{"cv_id":"cv_00001"}\n{"cv_id":"cv_00002"}\n', encoding="utf-8")

    assert runner.completed_ids(output) == {"cv_00001", "cv_00002"}


def test_json_roundtrip_serializes_runtime_values() -> None:
    class StringReason(str):
        pass

    response = runner.json_roundtrip(
        {"reason": StringReason("stop"), "path": Path("/tmp/result")}
    )

    assert response == {"reason": "stop", "path": "/tmp/result"}
    assert type(response["reason"]) is str


def test_sampler_fallback_is_explicit() -> None:
    assert runner.RUNTIME_ENV["VLLM_USE_FLASHINFER_SAMPLER"] == "0"

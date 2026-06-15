import importlib.util
import json
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).parents[1] / "modal_apps" / "nuextract3_vllm_trial.py"
SPEC = importlib.util.spec_from_file_location("nuextract3_vllm_trial", RUNNER_PATH)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_mtp_is_the_only_engine_difference() -> None:
    baseline = runner.engine_config(use_mtp=False)
    mtp = runner.engine_config(use_mtp=True)

    assert baseline | {"speculative_config": runner.MTP_SPECULATIVE_CONFIG} == mtp
    assert baseline["speculative_config"] is None
    assert mtp["speculative_config"] == {
        "method": "qwen3_next_mtp",
        "num_speculative_tokens": 2,
    }


def test_resume_config_rejects_changed_trial(tmp_path: Path) -> None:
    output = tmp_path / "responses.jsonl"
    baseline = runner.run_config(
        use_mtp=False,
        max_new_tokens=512,
        include_evidence_text=False,
        chunk_size=4,
    )
    runner.ensure_resume_config(output, baseline)
    runner.ensure_resume_config(output, baseline)

    stored = json.loads(runner.metadata_path(output).read_text(encoding="utf-8"))
    assert stored["configuration"] == baseline
    with pytest.raises(RuntimeError, match="different configuration"):
        runner.ensure_resume_config(
            output,
            runner.run_config(
                use_mtp=True,
                max_new_tokens=512,
                include_evidence_text=False,
                chunk_size=4,
            ),
        )


def test_completed_ids_supports_resumable_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "responses.jsonl"
    output.write_text(
        '{"cv_id":"cv_00001"}\n{"cv_id":"cv_00002"}\n',
        encoding="utf-8",
    )

    assert runner.completed_ids(output) == {"cv_00001", "cv_00002"}


def test_package_version_tolerates_runtime_provided_packages() -> None:
    assert runner.package_version("definitely-not-an-installed-distribution") == "runtime-provided"


def test_missing_request_latency_falls_back_to_chunk_share() -> None:
    assert runner.request_latency_or_chunk_share(None, 12.0, 4) == 3.0
    assert runner.request_latency_or_chunk_share(1.5, 12.0, 4) == 1.5


def test_nonprimitive_stop_reason_is_serialized() -> None:
    class StringReason(str):
        pass

    assert runner.serializable_reason({"token": 1}) == "{'token': 1}"
    serialized = runner.serializable_reason(StringReason("stop"))
    assert serialized == "stop"
    assert type(serialized) is str

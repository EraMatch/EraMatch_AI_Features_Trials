import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).parents[1] / "modal_apps" / "mapper_trial.py"


def load_mapper_module() -> Any:
    spec = importlib.util.spec_from_file_location("mapper_trial", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_configuration_requires_supported_model_and_pinned_revision() -> None:
    mapper = load_mapper_module()
    revision = "a" * 40
    mapper.validate_model_configuration("Qwen/Qwen3-4B-Instruct-2507", revision)

    with pytest.raises(ValueError, match="unsupported model_id"):
        mapper.validate_model_configuration("unknown/model", revision)
    with pytest.raises(ValueError, match="40-character"):
        mapper.validate_model_configuration("Qwen/Qwen3-0.6B", "")
    with pytest.raises(ValueError, match="40-character"):
        mapper.validate_model_configuration("microsoft/Phi-4-mini-instruct", "main")


def test_qwen_chat_template_disables_thinking_but_phi_uses_native_template() -> None:
    mapper = load_mapper_module()

    assert mapper.chat_template_kwargs("Qwen/Qwen3-0.6B")["enable_thinking"] is False
    assert "enable_thinking" not in mapper.chat_template_kwargs("microsoft/Phi-4-mini-instruct")


def test_model_dtype_keyword_matches_isolated_transformers_environment() -> None:
    mapper = load_mapper_module()

    assert mapper.model_dtype_keyword("Qwen/Qwen3-4B-Instruct-2507") == "dtype"
    assert mapper.model_dtype_keyword("microsoft/Phi-4-mini-instruct") == "torch_dtype"


def test_instruction_uses_parser_request_schema_and_default_template() -> None:
    mapper = load_mapper_module()
    instruction = mapper.build_instruction({"cv_id": "cv_1", "text": "Alice knows Python."})

    assert '"skills": []' in instruction
    assert '"work_experience": [' in instruction
    assert "Alice knows Python." in instruction


def test_pending_chunks_resume_only_matching_model_revision(tmp_path: Path) -> None:
    mapper = load_mapper_module()
    requests_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    requests_path.write_text(
        "\n".join(
            json.dumps({"cv_id": cv_id, "text": cv_id, "schema": {}})
            for cv_id in ("a", "b", "c")
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        "\n".join(
            [
                json.dumps({"cv_id": "a", "model_id": "Qwen/Qwen3-0.6B", "revision": "rev"}),
                json.dumps({"cv_id": "b", "model_id": "Qwen/Qwen3-0.6B", "revision": "old"}),
            ]
        ),
        encoding="utf-8",
    )

    revision = "b" * 40
    chunks = mapper.load_pending_chunks(
        requests_path,
        output_path,
        model_id="Qwen/Qwen3-0.6B",
        revision=revision,
        chunk_size=2,
        max_records=0,
    )

    assert [[row["cv_id"] for row in chunk] for chunk in chunks] == [["a", "b"], ["c"]]

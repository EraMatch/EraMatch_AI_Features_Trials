import json
from pathlib import Path

from eraparse.parsers import _write_value


def test_write_value_serializes_json_and_text(tmp_path: Path) -> None:
    json_path = tmp_path / "value.json"
    text_path = tmp_path / "value.md"
    _write_value(json_path, '{"a": 1}')
    _write_value(text_path, "markdown")
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": 1}
    assert text_path.read_text(encoding="utf-8") == "markdown"

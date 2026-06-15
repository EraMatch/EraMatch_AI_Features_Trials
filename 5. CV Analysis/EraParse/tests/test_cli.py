import json
from pathlib import Path

from typer.testing import CliRunner

from eraparse.cli import app

runner = CliRunner()


def test_eval_cli_returns_nonzero_for_invalid_prediction(
    tmp_path: Path, reduced_target: dict[str, object]
) -> None:
    truth = tmp_path / "truth.json"
    prediction = tmp_path / "prediction.json"
    truth.write_text(json.dumps(reduced_target), encoding="utf-8")
    prediction.write_text("{bad json", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "eval",
            "score",
            "--truth",
            str(truth),
            "--prediction",
            str(prediction),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert '"json_valid": false' in result.stdout

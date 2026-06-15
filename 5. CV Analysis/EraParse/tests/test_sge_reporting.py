import json
from pathlib import Path

from eraparse.sge_reporting import collect_trial_summaries, write_trial_report


def test_trial_report_collects_training_and_validation_metrics(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "query-only"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "mode": "sge",
                "steps": 100,
                "completed_steps": 200,
                "primary_decoder": "sequence",
                "training_evaluation": {"macro_score": 0.8, "schema_valid_rate": 1.0},
                "validation_evaluation": {"macro_score": 0.7, "schema_valid_rate": 1.0},
                "runtime_seconds": 12.5,
            }
        ),
        encoding="utf-8",
    )

    rows = collect_trial_summaries(tmp_path / "runs")
    assert rows[0]["validation_macro"] == 0.7
    assert rows[0]["completed_steps"] == 200

    output = tmp_path / "report.json"
    report = write_trial_report(tmp_path / "runs", output)
    assert report["trial_count"] == 1
    assert "query-only" in output.with_suffix(".md").read_text(encoding="utf-8")

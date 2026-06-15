import json
from pathlib import Path

from eraparse.trials import (
    chunk_pending_requests,
    ingest_nuextract_results,
    parse_generated_json,
    repair_prediction_work_records,
)


def test_parse_generated_json_handles_markdown_and_prefix() -> None:
    parsed, error = parse_generated_json('<|output|>```json\n{"full_name": "Jane"}\n```')
    assert error is None
    assert parsed == {"full_name": "Jane"}


def test_parse_generated_json_reports_invalid_output() -> None:
    parsed, error = parse_generated_json("not json")
    assert parsed is None
    assert error


def test_parse_generated_json_trims_invalid_trailing_extra_field() -> None:
    raw = (
        '{"full_name":"Jane","email":"jane@example.com","location":"Paris","phone":"123",'
        '"summary":"Hi","skills":[],"work_experience":[],"education":[],"projects":[],'
        '"certifications":[],"additional_info":{"publication":["Bad "quote" field"]}}'
    )
    parsed, error = parse_generated_json(raw)
    assert error is None
    assert parsed is not None
    assert "additional_info" not in parsed
    assert parsed["full_name"] == "Jane"


def test_parse_generated_json_trims_extra_field_with_extra_closing_brace() -> None:
    raw = (
        '{"full_name":"Jane","email":"jane@example.com","location":"Paris","phone":"123",'
        '"summary":"Hi","skills":[],"work_experience":[],"education":[],"projects":[],'
        '"certifications":[]}, "etcetera":["noise"]}'
    )
    parsed, error = parse_generated_json(raw)
    assert error is None
    assert parsed is not None
    assert "etcetera" not in parsed
    assert parsed["email"] == "jane@example.com"


def test_chunk_pending_requests_skips_completed_ids() -> None:
    requests = [{"cv_id": f"cv_{index:05d}"} for index in range(7)]
    chunks = chunk_pending_requests(requests, {"cv_00001", "cv_00005"}, chunk_size=2)
    assert [[row["cv_id"] for row in chunk] for chunk in chunks] == [
        ["cv_00000", "cv_00002"],
        ["cv_00003", "cv_00004"],
        ["cv_00006"],
    ]


def test_ingest_nuextract_results(tmp_path: Path, reduced_target: dict[str, object]) -> None:
    requests = [
        {
            "cv_id": "cv_00001",
            "split": "debug_50",
            "text": json.dumps(reduced_target),
            "truth": reduced_target,
        }
    ]
    responses = [
        {
            "cv_id": "cv_00001",
            "raw_output": json.dumps(reduced_target),
            "latency_seconds": 0.1,
            "input_tokens": 10,
            "output_tokens": 20,
        }
    ]
    summary = ingest_nuextract_results(
        requests,
        responses,
        representation="oracle_text",
        output_dir=tmp_path / "outputs",
        run_db=tmp_path / "runs.duckdb",
        manifest_hash="test",
    )
    assert summary["aggregate"]["document_count"] == 1
    assert summary["aggregate"]["schema_valid_rate"] == 1.0
    assert Path(summary["results_path"]).is_file()
    assert Path(summary["results_path"]).parent.name == summary["run_id"]


def test_ingest_supports_alternate_mapper_metadata(
    tmp_path: Path, reduced_target: dict[str, object]
) -> None:
    summary = ingest_nuextract_results(
        [
            {
                "cv_id": "cv_00001",
                "split": "debug_50",
                "text": json.dumps(reduced_target),
                "truth": reduced_target,
            }
        ],
        [
            {
                "cv_id": "cv_00001",
                "raw_output": json.dumps(reduced_target),
                "latency_seconds": 0.1,
            }
        ],
        representation="pymupdf_text",
        output_dir=tmp_path / "outputs",
        run_db=tmp_path / "runs.duckdb",
        model_id="Qwen/Qwen3-0.6B",
        revision="revision",
        run_kind="mapper_comparison",
    )
    assert summary["model_id"] == "Qwen/Qwen3-0.6B"
    assert summary["revision"] == "revision"
    assert summary["run_id"].startswith("qwen3-0.6b-")


def test_generic_mapper_ingest_cli_is_available() -> None:
    from typer.testing import CliRunner

    from eraparse.cli import app

    result = CliRunner().invoke(app, ["trials", "ingest-mapper", "--help"])
    assert result.exit_code == 0
    assert "--model-id" in result.stdout
    assert "--revision" in result.stdout


def test_nuextract_ingest_cli_supports_partial_smokes() -> None:
    from typer.testing import CliRunner

    from eraparse.cli import app

    result = CliRunner().invoke(app, ["trials", "ingest-nuextract3", "--help"])
    assert result.exit_code == 0
    assert "--allow-partial" in result.stdout


def test_route_fields_help_exposes_matched_result_inputs() -> None:
    from typer.testing import CliRunner

    from eraparse.cli import app

    result = CliRunner().invoke(app, ["trials", "route-fields", "--help"])
    assert result.exit_code == 0
    assert "--primary-results" in result.stdout
    assert "--specialist-results" in result.stdout
    assert "--output" in result.stdout


def test_focused_router_cli_commands_are_available() -> None:
    from typer.testing import CliRunner

    from eraparse.cli import app

    prepare = CliRunner().invoke(app, ["trials", "router-prepare-focused", "--help"])
    fuse = CliRunner().invoke(app, ["trials", "router-fuse-focused", "--help"])
    assert prepare.exit_code == 0
    assert "--mapper-requests" in prepare.stdout
    assert fuse.exit_code == 0
    assert "--specialist-responses" in fuse.stdout


def test_ingest_preserves_visual_resource_metrics(
    tmp_path: Path, reduced_target: dict[str, object]
) -> None:
    summary = ingest_nuextract_results(
        [
            {
                "cv_id": "cv_00001",
                "split": "validation",
                "evidence_text": json.dumps(reduced_target),
                "truth": reduced_target,
            }
        ],
        [
            {
                "cv_id": "cv_00001",
                "raw_output": json.dumps(reduced_target),
                "latency_seconds": 0.3,
                "encoder_latency_seconds": 0.1,
                "decoder_latency_seconds": 0.2,
                "visual_tokens": 4800,
            }
        ],
        representation="donut_direct_visual",
        output_dir=tmp_path / "outputs",
        run_db=tmp_path / "runs.duckdb",
        model_id="naver-clova-ix/donut-base",
        revision="revision",
        run_kind="direct_visual_baseline",
    )
    result = json.loads(Path(summary["results_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert result["visual_tokens"] == 4800
    assert result["encoder_latency_seconds"] == 0.1
    assert result["decoder_latency_seconds"] == 0.2


def test_repair_prediction_work_records_normalizes_required_strings() -> None:
    repaired, events = repair_prediction_work_records(
        {
            "work_experience": [
                {
                    "job_title": "Engineer",
                    "company": "Acme",
                    "start_date": "Jan 2020",
                    "end_date": None,
                    "duration": "Jan 2020 - Present",
                }
            ]
        }
    )
    assert repaired is not None
    assert repaired["work_experience"][0]["end_date"] == "Present"
    assert repaired["work_experience"][0]["duration"] == "Jan 2020 - Present"
    assert any(event.kind == "work_end_date_repaired" for event in events)


def test_ingest_nuextract_results_can_repair_work_records(
    tmp_path: Path, reduced_target: dict[str, object]
) -> None:
    prediction = dict(reduced_target)
    prediction["work_experience"] = [
        {
            "job_title": "Senior Engineer",
            "company": "Era Labs",
            "start_date": "Jan 2020",
            "end_date": None,
            "duration": "Jan 2020 - Present",
        }
    ]
    summary = ingest_nuextract_results(
        [
            {
                "cv_id": "cv_00001",
                "split": "validation",
                "evidence_text": json.dumps(reduced_target),
                "truth": reduced_target,
            }
        ],
        [
            {
                "cv_id": "cv_00001",
                "raw_output": json.dumps(prediction),
                "latency_seconds": 0.2,
            }
        ],
        representation="nuextract3_visual",
        output_dir=tmp_path / "outputs",
        run_db=tmp_path / "runs.duckdb",
        model_id="numind/NuExtract3",
        revision="revision",
        run_kind="structured_vlm_upper_bound",
        repair_work_records=True,
    )
    assert summary["aggregate"]["schema_valid_rate"] == 1.0
    assert summary["repair_work_records"] is True
    result = json.loads(Path(summary["results_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert result["prediction_raw"]["work_experience"][0]["end_date"] is None
    assert result["prediction"]["work_experience"][0]["end_date"] == "Present"
    assert any(event["kind"] == "work_end_date_repaired" for event in result["repair_events"])


def test_ingest_nuextract_results_expands_compact_schema(
    tmp_path: Path, reduced_target: dict[str, object]
) -> None:
    from eraparse.compact_schema import reduced_to_compact

    summary = ingest_nuextract_results(
        [
            {
                "cv_id": "cv_00001",
                "split": "debug_250",
                "evidence_text": json.dumps(reduced_target),
                "truth": reduced_target,
            }
        ],
        [
            {
                "cv_id": "cv_00001",
                "raw_output": json.dumps(reduced_to_compact(reduced_target)),
                "latency_seconds": 0.2,
                "output_tokens": 80,
            }
        ],
        representation="nuextract3_visual_compact",
        output_dir=tmp_path / "outputs",
        run_db=tmp_path / "runs.duckdb",
        model_id="numind/NuExtract3",
        revision="revision",
        run_kind="structured_vlm_speed_ablation",
        compact_schema=True,
    )
    assert summary["compact_schema"] is True
    assert summary["aggregate"]["macro_score"] == 1.0
    result = json.loads(Path(summary["results_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert result["prediction_raw"]["n"] == "Jane Doe"
    assert result["prediction"]["full_name"] == "Jane Doe"


def test_ingest_nuextract_results_can_ingest_partial_smoke(
    tmp_path: Path, reduced_target: dict[str, object]
) -> None:
    requests = [
        {
            "cv_id": cv_id,
            "split": "debug_50",
            "text": json.dumps(reduced_target),
            "truth": reduced_target,
        }
        for cv_id in ("cv_00001", "cv_00002")
    ]
    summary = ingest_nuextract_results(
        requests,
        [{"cv_id": "cv_00001", "raw_output": json.dumps(reduced_target)}],
        representation="smoke",
        output_dir=tmp_path / "outputs",
        run_db=tmp_path / "runs.duckdb",
        allow_partial=True,
    )
    assert summary["aggregate"]["document_count"] == 1
    assert summary["allow_partial"] is True

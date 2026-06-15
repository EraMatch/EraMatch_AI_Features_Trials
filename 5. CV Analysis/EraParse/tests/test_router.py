import json
from pathlib import Path

from eraparse.router import (
    build_focused_specialist_requests,
    fuse_focused_specialist_responses,
    observable_field_features,
    primary_only_field_features,
    route_primary_only_specialist,
    route_specialist,
    run_selective_field_router,
)


def _row(
    cv_id: str,
    *,
    work: list[dict[str, str]],
    certifications: list[dict[str, str]],
    latency: float,
) -> dict[str, object]:
    prediction = {
        "full_name": "",
        "email": "",
        "location": "",
        "phone": "",
        "summary": "",
        "linkedin_url": None,
        "github_url": None,
        "skills": [],
        "work_experience": work,
        "education": [],
        "projects": None,
        "certifications": certifications,
    }
    return {
        "cv_id": cv_id,
        "prediction": prediction,
        "latency_seconds": latency,
        "evaluation": {
            "field_results": [
                {"path": "work_experience", "supported": True},
                {"path": "certifications", "supported": True},
            ]
        },
    }


def test_observable_features_do_not_include_truth_or_scores() -> None:
    primary = _row("cv_1", work=[{"company": "A"}], certifications=[], latency=1.0)
    specialist = _row("cv_1", work=[{"company": "B"}], certifications=[], latency=2.0)

    features = observable_field_features(primary, specialist, "work_experience")

    assert features == {
        "disagrees": True,
        "primary_count": 1,
        "specialist_count": 1,
        "primary_supported": True,
    }


def test_router_uses_qwen_for_same_count_disagreement_in_work() -> None:
    features = {
        "disagrees": True,
        "primary_count": 2,
        "specialist_count": 2,
        "primary_supported": True,
    }
    assert route_specialist("work_experience", features)


def test_router_keeps_primary_when_work_record_counts_differ() -> None:
    features = {
        "disagrees": True,
        "primary_count": 2,
        "specialist_count": 1,
        "primary_supported": True,
    }
    assert not route_specialist("work_experience", features)


def test_router_uses_qwen_for_nonempty_certification_count_disagreement() -> None:
    features = {
        "disagrees": True,
        "primary_count": 1,
        "specialist_count": 2,
        "primary_supported": True,
    }
    assert route_specialist("certifications", features)


def test_router_does_not_escalate_empty_primary_certifications() -> None:
    features = {
        "disagrees": True,
        "primary_count": 0,
        "specialist_count": 1,
        "primary_supported": True,
    }
    assert not route_specialist("certifications", features)


def test_run_router_fuses_fields_and_accounts_for_selective_latency(tmp_path: Path) -> None:
    primary = _row(
        "cv_1",
        work=[{"company": "primary"}],
        certifications=[{"name": "primary"}],
        latency=1.0,
    )
    specialist = _row(
        "cv_1",
        work=[{"company": "specialist"}],
        certifications=[{"name": "specialist"}, {"name": "second"}],
        latency=4.0,
    )
    output = tmp_path / "responses.jsonl"

    summary = run_selective_field_router([primary], [specialist], output)
    result = json.loads(output.read_text().strip())
    prediction = json.loads(result["raw_output"])

    assert prediction["work_experience"] == [{"company": "specialist"}]
    assert prediction["certifications"] == [
        {"name": "specialist"},
        {"name": "second"},
    ]
    assert result["latency_seconds"] == 5.0
    assert result["routed_fields"] == ["work_experience", "certifications"]
    assert summary["escalated_documents"] == 1
    assert summary["escalation_rate"] == 1.0


def test_primary_only_router_uses_no_specialist_prediction_features() -> None:
    primary = _row(
        "cv_1",
        work=[{"company": "A"}],
        certifications=[{"name": "C"}],
        latency=1.0,
    )

    features = primary_only_field_features(primary, "work_experience")

    assert features == {
        "primary_count": 1,
        "primary_characters": len(json.dumps([{"company": "A"}], sort_keys=True)),
        "primary_supported": True,
    }


def test_primary_only_router_routes_supported_short_work_and_certifications() -> None:
    assert route_primary_only_specialist(
        "work_experience",
        {"primary_count": 1, "primary_characters": 200, "primary_supported": True},
    )
    assert route_primary_only_specialist(
        "certifications",
        {"primary_count": 1, "primary_characters": 40, "primary_supported": True},
    )
    assert not route_primary_only_specialist(
        "work_experience",
        {"primary_count": 2, "primary_characters": 500, "primary_supported": True},
    )


def test_build_focused_requests_includes_only_routed_fields(tmp_path: Path) -> None:
    primary = _row(
        "cv_1",
        work=[{"company": "A"}],
        certifications=[],
        latency=1.0,
    )
    output = tmp_path / "focused.jsonl"

    summary = build_focused_specialist_requests(
        [primary],
        [{"cv_id": "cv_1", "text": "CV text", "parser_id": "pymupdf4llm"}],
        output,
    )
    request = json.loads(output.read_text().strip())

    assert request["schema"] == {
        "work_experience": [
            {
                "job_title": "",
                "company": "",
                "start_date": "",
                "end_date": "",
                "duration": "",
            }
        ]
    }
    assert request["routed_fields"] == ["work_experience"]
    assert summary["request_count"] == 1


def test_fuse_focused_responses_preserves_complete_primary_prediction(tmp_path: Path) -> None:
    primary = _row(
        "cv_1",
        work=[{"company": "A"}],
        certifications=[],
        latency=1.0,
    )
    output = tmp_path / "fused.jsonl"

    summary = fuse_focused_specialist_responses(
        [primary],
        [
            {
                "cv_id": "cv_1",
                "raw_output": json.dumps({"work_experience": [{"company": "B"}]}),
                "latency_seconds": 2.0,
            }
        ],
        {"cv_1": ["work_experience"]},
        output,
    )
    result = json.loads(output.read_text().strip())
    prediction = json.loads(result["raw_output"])

    assert prediction["work_experience"] == [{"company": "B"}]
    assert prediction["education"] == []
    assert result["latency_seconds"] == 3.0
    assert summary["specialist_response_count"] == 1

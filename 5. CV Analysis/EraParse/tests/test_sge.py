import json

from eraparse.sge import (
    _accept_sgrse_work,
    _bootstrap_mean_confidence_interval,
    _decode_anchor_records,
    _decode_sgrse_work_rows,
    _extract_project_technologies,
    _merge_sgrse_work_rows,
    _project_url_probability,
    _project_url_selector_stats,
    _work_record_features,
    apply_efsfr_repairs,
    apply_project_skill_tech_repairs,
    apply_project_url_repairs,
    apply_selective_sgrse_work_decoder,
    apply_sgrse_work_decoder,
    compare_prediction_sets,
    decode_labeled_candidates,
    decode_word_candidates,
    evaluate_grounded_rows,
    evaluate_record_oracle,
    prepare_work_record_bank,
    record_group_id,
    repair_grounded_work_predictions,
    repair_work_record,
)
from eraparse.sge_losses import binary_positive_weight, token_class_weights


def test_record_group_ids_do_not_collide_across_nested_sections() -> None:
    assert record_group_id("work_experience.*.company", 0) != record_group_id(
        "education.*.institution", 0
    )
    assert record_group_id("summary", None) == -1


def test_token_class_weights_downweight_background_only() -> None:
    assert token_class_weights(3, outside_weight=0.05) == [0.05, 1.0, 1.0, 1.0]


def test_binary_positive_weight_balances_and_caps_rare_pairs() -> None:
    assert binary_positive_weight(positive_count=10, negative_count=40) == 4.0
    assert binary_positive_weight(positive_count=1, negative_count=100, cap=20) == 20.0
    assert binary_positive_weight(positive_count=0, negative_count=10) == 1.0


def test_decode_word_candidates_uses_grouping_for_nested_records() -> None:
    record = {
        "cv_id": "cv_1",
        "words": ["Engineer", "Acme", "Manager", "Beta"],
        "evidence_ids": ["e0", "e1", "e2", "e3"],
    }
    candidates = decode_word_candidates(
        record,
        word_labels=[9, 10, 9, 10],
        confidences=[0.9, 0.8, 0.7, 0.6],
        grouping_probabilities=[
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ],
    )
    assert [(item.schema_path, item.value, item.record_index) for item in candidates] == [
        ("work_experience.*.job_title", "Engineer", 0),
        ("work_experience.*.company", "Acme", 0),
        ("work_experience.*.job_title", "Manager", 1),
        ("work_experience.*.company", "Beta", 1),
    ]


def test_grouping_never_splits_contiguous_tokens_of_same_nested_field() -> None:
    record = {
        "cv_id": "cv_1",
        "words": ["UI", "Engineer", "ModelShip"],
        "evidence_ids": ["e0", "e1", "e2"],
    }
    candidates = decode_word_candidates(
        record,
        word_labels=[9, 9, 10],
        confidences=[0.9, 0.9, 0.9],
        grouping_probabilities=[
            [1.0, 0.1, 0.1],
            [0.1, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ],
    )
    assert [(item.value, item.record_index) for item in candidates] == [
        ("UI Engineer", 0),
        ("ModelShip", 1),
    ]


def test_grouping_cannot_merge_two_distinct_values_for_same_record_field() -> None:
    record = {
        "cv_id": "cv_1",
        "words": ["Engineer", "Acme", "Manager", "Beta"],
        "evidence_ids": ["e0", "e1", "e2", "e3"],
    }
    candidates = decode_word_candidates(
        record,
        word_labels=[9, 10, 9, 10],
        confidences=[0.9, 0.9, 0.9, 0.9],
        grouping_probabilities=[[1.0] * 4 for _ in range(4)],
    )
    assert [item.record_index for item in candidates] == [0, 0, 1, 1]


def test_decode_word_candidates_uses_sequence_heuristic_without_grouping_head() -> None:
    record = {
        "cv_id": "cv_1",
        "words": ["Engineer", "Acme", "Manager", "Beta"],
        "evidence_ids": ["e0", "e1", "e2", "e3"],
    }
    candidates = decode_word_candidates(
        record,
        word_labels=[9, 10, 9, 10],
        confidences=[0.9, 0.8, 0.7, 0.6],
    )
    assert [item.record_index for item in candidates] == [0, 0, 1, 1]


def test_decode_word_candidates_splits_comma_delimited_skills() -> None:
    record = {
        "cv_id": "cv_1",
        "words": ["Python,", "Cloudflare", "Pages,", "React"],
        "evidence_ids": ["e0", "e1", "e2", "e3"],
    }
    candidates = decode_word_candidates(
        record,
        word_labels=[8, 8, 8, 8],
        confidences=[0.9, 0.9, 0.9, 0.9],
    )
    assert [item.value for item in candidates] == ["Python", "Cloudflare Pages", "React"]


def test_decode_labeled_candidates_uses_oracle_record_indices() -> None:
    record = {
        "cv_id": "cv_1",
        "words": ["Engineer", "Acme", "Manager", "Beta"],
        "evidence_ids": ["e0", "e1", "e2", "e3"],
        "field_labels": [9, 10, 9, 10],
        "record_indices": [0, 0, 1, 1],
    }
    candidates = decode_labeled_candidates(record, grouping_mode="oracle")
    assert [item.record_index for item in candidates] == [0, 0, 1, 1]


def test_evaluate_record_oracle_reports_sequence_and_oracle_grouping(tmp_path) -> None:
    records_path = tmp_path / "records.jsonl"
    rows = [
        {
            "cv_id": "cv_1",
            "words": ["Jane", "Engineer", "Acme"],
            "evidence_ids": ["e0", "e1", "e2"],
            "field_labels": [1, 9, 10],
            "record_indices": [-1, 0, 0],
            "truth": {
                "full_name": "Jane",
                "email": "",
                "location": "",
                "phone": "",
                "linkedin_url": None,
                "github_url": None,
                "summary": "",
                "skills": [],
                "work_experience": [
                    {
                        "job_title": "Engineer",
                        "company": "Acme",
                        "start_date": "",
                        "end_date": "",
                        "duration": "",
                    }
                ],
                "education": [],
                "projects": None,
                "certifications": None,
            },
        }
    ]
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = evaluate_record_oracle(records_path, tmp_path / "oracle", grouping_mode="oracle")
    assert summary["evaluation"]["schema_valid_rate"] == 1.0
    assert summary["candidate_count"] == 3


def test_prepare_work_record_bank_emits_document_level_spans_and_coverage(tmp_path) -> None:
    records_path = tmp_path / "records.jsonl"
    rows = [
        {
            "cv_id": "cv_1",
            "split": "validation",
            "tier": "T1",
            "reader": "pymupdf4llm",
            "oracle": False,
            "page": 1,
            "words": ["Engineer", "Acme", "2022", "2024"],
            "evidence_ids": ["e0", "e1", "e2", "e3"],
            "field_labels": [9, 10, 11, 12],
            "record_indices": [0, 0, 0, 0],
            "truth": {
                "full_name": "",
                "email": "",
                "location": "",
                "phone": "",
                "linkedin_url": None,
                "github_url": None,
                "summary": "",
                "skills": [],
                "work_experience": [
                    {
                        "job_title": "Engineer",
                        "company": "Acme",
                        "start_date": "2022",
                        "end_date": "2024",
                        "duration": "",
                    }
                ],
                "education": [],
                "projects": None,
                "certifications": None,
            },
        }
    ]
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = prepare_work_record_bank(records_path, tmp_path / "work_bank.jsonl")
    assert summary["document_count"] == 1
    assert summary["span_count"] == 4
    assert summary["field_coverage"]["work_experience.*.job_title"] == 1.0
    assert summary["exact_record_match_rate"] == 1.0
    assert summary["direct_record_match_rate"] == 1.0


def test_repair_work_record_recovers_dates_from_duration() -> None:
    repaired, events = repair_work_record(
        {
            "job_title": "AppSec Engineer",
            "company": "Airbnb",
            "start_date": "",
            "end_date": "Present)",
            "duration": "(2022-07 -",
        }
    )
    assert repaired == {
        "job_title": "AppSec Engineer",
        "company": "Airbnb",
        "start_date": "2022-07",
        "end_date": "Present",
        "duration": "2022-07 - Present",
    }
    assert [event.kind for event in events] == [
        "work_start_date_repaired",
        "work_duration_normalized",
    ]


def test_repair_grounded_work_predictions_updates_work_records_only(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "cv_id": "cv_1",
            "prediction": {
                "full_name": "Jane Doe",
                "email": "jane@example.com",
                "location": "Cairo",
                "phone": "+201234567890",
                "linkedin_url": None,
                "github_url": None,
                "summary": "Engineer",
                "skills": ["Python"],
                "work_experience": [
                    {
                        "job_title": "AppSec Engineer",
                        "company": "Airbnb",
                        "start_date": "",
                        "end_date": "Present)",
                        "duration": "(2022-07 -",
                    }
                ],
                "education": [],
                "projects": None,
                "certifications": None,
            },
            "candidates": [],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        }
    ]
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output_path = tmp_path / "repaired.jsonl"
    summary = repair_grounded_work_predictions(predictions_path, output_path)
    repaired_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["document_count"] == 1
    assert summary["repaired_document_count"] == 1
    assert summary["repaired_record_count"] == 1
    assert summary["event_counts"]["work_start_date_repaired"] == 1
    assert summary["event_counts"]["work_duration_normalized"] == 1
    assert repaired_rows[0]["prediction"]["full_name"] == "Jane Doe"
    assert repaired_rows[0]["prediction"]["work_experience"] == [
        {
            "job_title": "AppSec Engineer",
            "company": "Airbnb",
            "start_date": "2022-07",
            "end_date": "Present",
            "duration": "2022-07 - Present",
        }
    ]


def test_evaluate_grounded_rows_uses_all_pages_for_evidence_support(tmp_path) -> None:
    grounded_path = tmp_path / "grounded.jsonl"
    requests_path = tmp_path / "requests.jsonl"

    grounded_rows = [
        {
            "cv_id": "cv_1",
            "prediction": {
                "full_name": "Jane Doe",
                "email": "",
                "location": "",
                "phone": "",
                "linkedin_url": None,
                "github_url": None,
                "summary": "",
                "skills": [],
                "work_experience": [],
                "education": [],
                "projects": None,
                "certifications": None,
            },
            "candidates": [],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        }
    ]
    request_rows = [
        {
            "cv_id": "cv_1",
            "truth": grounded_rows[0]["prediction"],
            "words": ["Jane", "Doe"],
        },
        {
            "cv_id": "cv_1",
            "truth": grounded_rows[0]["prediction"],
            "words": ["additional", "page"],
        },
    ]

    grounded_path.write_text(
        "".join(json.dumps(row) + "\n" for row in grounded_rows),
        encoding="utf-8",
    )
    requests_path.write_text(
        "".join(json.dumps(row) + "\n" for row in request_rows),
        encoding="utf-8",
    )

    summary = evaluate_grounded_rows(grounded_path, requests_path, tmp_path / "evaluation.json")
    assert summary["unsupported_evidence_rate"] == 0.0


def test_decode_anchor_records_ignores_pre_anchor_noise_and_rebuilds_order() -> None:
    candidates = [
        {
            "schema_path": "education.*.field_of_study",
            "value": "Data Science",
            "confidence": 0.9,
            "evidence_ids": ["cv_1:p1:w4"],
        },
        {
            "schema_path": "education.*.degree",
            "value": "B.A.",
            "confidence": 0.99,
            "evidence_ids": ["cv_1:p1:w193"],
        },
        {
            "schema_path": "education.*.field_of_study",
            "value": "Mathematics",
            "confidence": 0.99,
            "evidence_ids": ["cv_1:p1:w195"],
        },
        {
            "schema_path": "education.*.institution",
            "value": "Purdue University",
            "confidence": 0.99,
            "evidence_ids": ["cv_1:p1:w197"],
        },
        {
            "schema_path": "education.*.graduation_date",
            "value": "2013",
            "confidence": 0.99,
            "evidence_ids": ["cv_1:p1:w199"],
        },
        {
            "schema_path": "education.*.degree",
            "value": "M.Tech.",
            "confidence": 0.99,
            "evidence_ids": ["cv_1:p1:w200"],
        },
        {
            "schema_path": "education.*.field_of_study",
            "value": "Electrical Engineering",
            "confidence": 0.99,
            "evidence_ids": ["cv_1:p1:w202"],
        },
        {
            "schema_path": "education.*.institution",
            "value": "Columbia University",
            "confidence": 0.99,
            "evidence_ids": ["cv_1:p1:w205"],
        },
        {
            "schema_path": "education.*.graduation_date",
            "value": "2014",
            "confidence": 0.99,
            "evidence_ids": ["cv_1:p1:w207"],
        },
    ]
    decoded = _decode_anchor_records(
        candidates,
        prefix="education.*.",
        anchor_field="degree",
        fields=("degree", "field_of_study", "institution", "graduation_date"),
    )
    assert decoded == [
        {
            "degree": "B.A.",
            "field_of_study": "Mathematics",
            "institution": "Purdue University",
            "graduation_date": "2013",
        },
        {
            "degree": "M.Tech.",
            "field_of_study": "Electrical Engineering",
            "institution": "Columbia University",
            "graduation_date": "2014",
        },
    ]


def test_apply_efsfr_repairs_rebuilds_nested_records_from_candidates(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "cv_id": "cv_1",
            "prediction": {
                "full_name": "Jane Doe",
                "email": "jane@example.com",
                "location": "Cairo",
                "phone": "+201234567890",
                "linkedin_url": None,
                "github_url": None,
                "summary": "Engineer",
                "skills": ["Python"],
                "work_experience": [
                    {
                        "job_title": "AppSec Engineer",
                        "company": "Airbnb",
                        "start_date": "",
                        "end_date": "Present)",
                        "duration": "(2022-07 -",
                    }
                ],
                "education": [
                    {
                        "degree": "B.A.",
                        "field_of_study": "Data Science",
                        "institution": "",
                        "graduation_date": "",
                    }
                ],
                "projects": [{"name": "", "technologies": ["React"], "url": None}],
                "certifications": [{"name": "", "issuer": "noise", "date": ""}],
            },
            "candidates": [
                {
                    "schema_path": "education.*.field_of_study",
                    "value": "Data Science",
                    "confidence": 0.9,
                    "evidence_ids": ["cv_1:p1:w4"],
                },
                {
                    "schema_path": "education.*.degree",
                    "value": "B.A.",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w193"],
                },
                {
                    "schema_path": "education.*.field_of_study",
                    "value": "Mathematics",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w195"],
                },
                {
                    "schema_path": "education.*.institution",
                    "value": "Purdue University",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w197"],
                },
                {
                    "schema_path": "education.*.graduation_date",
                    "value": "2013",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w199"],
                },
                {
                    "schema_path": "projects.*.technologies",
                    "value": "React",
                    "confidence": 0.8,
                    "evidence_ids": ["cv_1:p1:w210"],
                },
                {
                    "schema_path": "projects.*.name",
                    "value": "Deployment Dashboard",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w220"],
                },
                {
                    "schema_path": "projects.*.technologies",
                    "value": "Angular, CSS",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w222"],
                },
                {
                    "schema_path": "certifications.*.issuer",
                    "value": "noise",
                    "confidence": 0.7,
                    "evidence_ids": ["cv_1:p1:w230"],
                },
                {
                    "schema_path": "certifications.*.name",
                    "value": "AWS Certified Developer Associate",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w240"],
                },
                {
                    "schema_path": "certifications.*.issuer",
                    "value": "Amazon Web Services",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w242"],
                },
                {
                    "schema_path": "certifications.*.date",
                    "value": "2022",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w243"],
                },
            ],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        }
    ]
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output_path = tmp_path / "efsfr.jsonl"
    summary = apply_efsfr_repairs(predictions_path, output_path)
    repaired_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["document_count"] == 1
    assert summary["repaired_document_count"] == 1
    assert summary["event_counts"]["education_anchor_redecoded"] == 1
    assert summary["event_counts"]["projects_anchor_redecoded"] == 1
    assert summary["event_counts"]["certifications_anchor_redecoded"] == 1
    assert summary["event_counts"]["work_start_date_repaired"] == 1
    assert repaired_rows[0]["prediction"]["work_experience"] == [
        {
            "job_title": "AppSec Engineer",
            "company": "Airbnb",
            "start_date": "2022-07",
            "end_date": "Present",
            "duration": "2022-07 - Present",
        }
    ]
    assert repaired_rows[0]["prediction"]["education"] == [
        {
            "degree": "B.A.",
            "field_of_study": "Mathematics",
            "institution": "Purdue University",
            "graduation_date": "2013",
        }
    ]
    assert repaired_rows[0]["prediction"]["projects"] == [
        {
            "name": "Deployment Dashboard",
            "technologies": ["Angular", "CSS"],
            "url": None,
        }
    ]
    assert repaired_rows[0]["prediction"]["certifications"] == [
        {
            "name": "AWS Certified Developer Associate",
            "issuer": "Amazon Web Services",
            "date": "2022",
        }
    ]


def test_extract_project_technologies_uses_skill_hints_for_noisy_phrases() -> None:
    assert _extract_project_technologies(
        ["Created Data Visualization Tool leveraging Next.js for"],
        ["React", "TypeScript", "Next.js"],
    ) == ["Next.js"]
    assert _extract_project_technologies(
        ["Web App with CSS and Astro"],
        ["JavaScript", "CSS", "Astro"],
    ) == ["CSS", "Astro"]


def test_apply_project_skill_tech_repairs_only_replaces_noisy_project_tech(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "cv_id": "cv_1",
            "prediction": {
                "full_name": "Alex Doe",
                "email": "alex@example.com",
                "location": "Paris",
                "phone": "123",
                "linkedin_url": None,
                "github_url": "https://github.com/alexdoe",
                "summary": "Summary",
                "skills": ["React", "TypeScript", "Next.js", "CSS", "Astro"],
                "work_experience": [],
                "education": [],
                "projects": [
                    {
                        "name": "PDF Parser Toolkit",
                        "technologies": ["Created Data Visualization Tool leveraging Next.js for"],
                        "url": None,
                    },
                    {
                        "name": "Vector Search API",
                        "technologies": ["Web App with CSS and Astro"],
                        "url": None,
                    },
                ],
                "certifications": None,
            },
            "candidates": [
                {
                    "schema_path": "projects.*.name",
                    "value": "PDF Parser Toolkit",
                    "confidence": 0.99,
                    "record_index": 0,
                    "evidence_ids": ["cv_1:p1:w10"],
                },
                {
                    "schema_path": "projects.*.technologies",
                    "value": "Created Data Visualization Tool leveraging Next.js for",
                    "confidence": 0.91,
                    "record_index": 0,
                    "evidence_ids": ["cv_1:p1:w11"],
                },
                {
                    "schema_path": "projects.*.name",
                    "value": "Vector Search API",
                    "confidence": 0.99,
                    "record_index": 1,
                    "evidence_ids": ["cv_1:p1:w20"],
                },
                {
                    "schema_path": "projects.*.technologies",
                    "value": "Web App with CSS and Astro",
                    "confidence": 0.88,
                    "record_index": 1,
                    "evidence_ids": ["cv_1:p1:w21"],
                },
            ],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        }
    ]
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output_path = tmp_path / "project-tech.jsonl"
    summary = apply_project_skill_tech_repairs(predictions_path, output_path)
    repaired_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["document_count"] == 1
    assert summary["repaired_document_count"] == 1
    assert summary["repaired_project_count"] == 2
    assert summary["event_counts"]["project_tech_skill_repaired"] == 2
    assert repaired_rows[0]["prediction"]["projects"] == [
        {
            "name": "PDF Parser Toolkit",
            "technologies": ["Next.js"],
            "url": None,
        },
        {
            "name": "Vector Search API",
            "technologies": ["CSS", "Astro"],
            "url": None,
        },
    ]


def test_project_url_selector_uses_train_truth_buckets() -> None:
    stats = _project_url_selector_stats(
        [
            {
                "cv_id": "cv_1",
                "truth": {
                    "github_url": "https://github.com/alice",
                    "projects": [
                        {
                            "name": "Resume Parser",
                            "technologies": ["Python", "FastAPI", "Docker"],
                            "url": "https://github.com/alice/resume-parser",
                        }
                    ],
                },
            },
            {
                "cv_id": "cv_2",
                "truth": {
                    "github_url": "https://github.com/bob",
                    "projects": [
                        {
                            "name": "Resume Parser",
                            "technologies": ["Python", "FastAPI", "Docker"],
                            "url": "https://github.com/bob/resume-parser",
                        }
                    ],
                },
            },
            {
                "cv_id": "cv_3",
                "truth": {
                    "github_url": "https://github.com/carol",
                    "projects": [
                        {
                            "name": "Resume Parser",
                            "technologies": ["Python", "FastAPI", "Docker"],
                            "url": None,
                        }
                    ],
                },
            },
        ]
    )
    probability = _project_url_probability(
        stats,
        [
            {
                "name": "Resume Parser",
                "technologies": ["Python", "FastAPI", "Docker"],
                "url": None,
            }
        ],
        0,
        {
            "name": "Resume Parser",
            "technologies": ["Python", "FastAPI", "Docker"],
            "url": None,
        },
        min_full_count=1,
        min_reduced_count=1,
    )
    assert probability > 0.5


def test_apply_project_url_repairs_fills_only_supported_project_urls(tmp_path) -> None:
    train_records_path = tmp_path / "train.jsonl"
    train_rows = []
    for index in range(8):
        train_rows.append(
            {
                "cv_id": f"cv_yes_{index}",
                "truth": {
                    "github_url": f"https://github.com/user{index}",
                    "projects": [
                        {
                            "name": "Resume Parser",
                            "technologies": ["Python", "FastAPI", "Docker"],
                            "url": f"https://github.com/user{index}/resume-parser",
                        },
                        {
                            "name": "ChatBot Framework",
                            "technologies": ["Python"],
                            "url": None,
                        }
                    ],
                },
            }
        )
    for index in range(2):
        train_rows.append(
            {
                "cv_id": f"cv_no_{index}",
                "truth": {
                    "github_url": f"https://github.com/no{index}",
                    "projects": [
                        {
                            "name": "Resume Parser",
                            "technologies": ["Python", "FastAPI", "Docker"],
                            "url": None,
                        },
                        {
                            "name": "ChatBot Framework",
                            "technologies": ["Python"],
                            "url": None,
                        }
                    ],
                },
            }
        )
    train_records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows),
        encoding="utf-8",
    )

    predictions_path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "cv_id": "cv_pred",
            "prediction": {
                "full_name": "Alex Doe",
                "email": "alex@example.com",
                "location": "Paris",
                "phone": "123",
                "linkedin_url": None,
                "github_url": "https://github.com/alexdoe",
                "summary": "Summary",
                "skills": ["Python", "FastAPI", "Docker"],
                "work_experience": [],
                "education": [],
                "projects": [
                    {
                        "name": "Resume Parser",
                        "technologies": ["Python", "FastAPI", "Docker"],
                        "url": None,
                    },
                    {
                        "name": "ChatBot Framework",
                        "technologies": ["Python"],
                        "url": None,
                    },
                ],
                "certifications": None,
            },
            "candidates": [],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        }
    ]
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output_path = tmp_path / "project-url.jsonl"
    summary = apply_project_url_repairs(
        predictions_path,
        train_records_path,
        output_path,
        threshold=0.7,
        min_full_count=1,
        min_reduced_count=1,
    )
    repaired_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["document_count"] == 1
    assert summary["repaired_document_count"] == 1
    assert summary["repaired_project_count"] == 1
    assert summary["event_counts"]["project_url_synthesized"] == 1
    assert repaired_rows[0]["prediction"]["projects"] == [
        {
            "name": "Resume Parser",
            "technologies": ["Python", "FastAPI", "Docker"],
            "url": "https://github.com/alexdoe/resume-parser",
        },
        {
            "name": "ChatBot Framework",
            "technologies": ["Python"],
            "url": None,
        },
    ]


def test_merge_sgrse_work_rows_combines_split_title_and_company_slots() -> None:
    decoded_rows = [
        {
            "record_index": 0,
            "fields": {
                "job_title": "React Native",
                "company": "",
                "start_date": "2025-12",
                "end_date": "",
                "duration": "",
            },
            "confidences": {},
        },
        {
            "record_index": 1,
            "fields": {
                "job_title": "Developer",
                "company": "Airbnb",
                "start_date": "",
                "end_date": "Present",
                "duration": "",
            },
            "confidences": {},
        },
    ]
    merged, merge_count = _merge_sgrse_work_rows(decoded_rows)
    assert merge_count == 1
    assert merged == [
        {
            "job_title": "React Native Developer",
            "company": "Airbnb",
            "start_date": "2025-12",
            "end_date": "Present",
            "duration": "2025-12 - Present",
        }
    ]


def test_decode_sgrse_work_rows_groups_candidates_by_record_index() -> None:
    rows = _decode_sgrse_work_rows(
        [
            {
                "schema_path": "work_experience.*.job_title",
                "value": "Engineer",
                "confidence": 0.9,
                "record_index": 0,
            },
            {
                "schema_path": "work_experience.*.company",
                "value": "Acme",
                "confidence": 0.8,
                "record_index": 0,
            },
            {
                "schema_path": "work_experience.*.start_date",
                "value": "2022-01",
                "confidence": 0.7,
                "record_index": 0,
            },
            {
                "schema_path": "work_experience.*.end_date",
                "value": "2024-01",
                "confidence": 0.7,
                "record_index": 0,
            },
        ]
    )
    assert rows == [
        {
            "record_index": 0,
            "fields": {
                "job_title": "Engineer",
                "company": "Acme",
                "start_date": "2022-01",
                "end_date": "2024-01",
                "duration": "2022-01 - 2024-01",
            },
            "confidences": {
                "job_title": 0.9,
                "company": 0.8,
                "start_date": 0.7,
                "end_date": 0.7,
                "duration": 0.0,
            },
        }
    ]


def test_apply_sgrse_work_decoder_replaces_work_records_from_candidates(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "cv_id": "cv_1",
            "prediction": {
                "full_name": "Jane Doe",
                "email": "jane@example.com",
                "location": "Cairo",
                "phone": "+201234567890",
                "linkedin_url": None,
                "github_url": None,
                "summary": "Engineer",
                "skills": ["Python"],
                "work_experience": [
                    {
                        "job_title": "React Native",
                        "company": "",
                        "start_date": "2025-12",
                        "end_date": "",
                        "duration": "",
                    },
                    {
                        "job_title": "Developer",
                        "company": "Airbnb",
                        "start_date": "",
                        "end_date": "Present",
                        "duration": "",
                    },
                ],
                "education": [],
                "projects": None,
                "certifications": None,
            },
            "candidates": [
                {
                    "schema_path": "work_experience.*.job_title",
                    "value": "React Native",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w10"],
                    "record_index": 0,
                },
                {
                    "schema_path": "work_experience.*.start_date",
                    "value": "2025-12",
                    "confidence": 0.95,
                    "evidence_ids": ["cv_1:p1:w11"],
                    "record_index": 0,
                },
                {
                    "schema_path": "work_experience.*.job_title",
                    "value": "Developer",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w12"],
                    "record_index": 1,
                },
                {
                    "schema_path": "work_experience.*.end_date",
                    "value": "Present",
                    "confidence": 0.95,
                    "evidence_ids": ["cv_1:p1:w13"],
                    "record_index": 1,
                },
                {
                    "schema_path": "work_experience.*.company",
                    "value": "Airbnb",
                    "confidence": 0.99,
                    "evidence_ids": ["cv_1:p1:w14"],
                    "record_index": 1,
                },
            ],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        }
    ]
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output_path = tmp_path / "sgrse.jsonl"
    summary = apply_sgrse_work_decoder(predictions_path, output_path)
    repaired_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["document_count"] == 1
    assert summary["repaired_document_count"] == 1
    assert summary["merge_count"] == 1
    assert summary["event_counts"]["sgrse_work_redecoded"] == 1
    assert summary["event_counts"]["sgrse_work_slot_merged"] == 1
    assert repaired_rows[0]["prediction"]["work_experience"] == [
        {
            "job_title": "React Native Developer",
            "company": "Airbnb",
            "start_date": "2025-12",
            "end_date": "Present",
            "duration": "2025-12 - Present",
        }
    ]


def test_bootstrap_mean_confidence_interval_is_deterministic() -> None:
    assert _bootstrap_mean_confidence_interval([0.0, 1.0], samples=100) == (
        0.0,
        1.0,
    )


def test_compare_prediction_sets_reports_deltas(tmp_path) -> None:
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    requests_path = tmp_path / "requests.jsonl"

    truth = {
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "location": "Cairo",
        "phone": "+201234567890",
        "linkedin_url": None,
        "github_url": None,
        "summary": "Engineer",
        "skills": ["Python"],
        "work_experience": [
            {
                "job_title": "Engineer",
                "company": "Acme",
                "start_date": "2022-01",
                "end_date": "2024-01",
                "duration": "2022-01 - 2024-01",
            }
        ],
        "education": [],
        "projects": None,
        "certifications": None,
    }
    left_rows = [
        {
            "cv_id": "cv_1",
            "prediction": {**truth, "work_experience": []},
            "candidates": [],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        }
    ]
    right_rows = [
        {
            "cv_id": "cv_1",
            "prediction": truth,
            "candidates": [],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        }
    ]
    request_rows = [{"cv_id": "cv_1", "truth": truth, "words": ["Jane", "Doe", "Engineer", "Acme"]}]

    left_path.write_text("".join(json.dumps(row) + "\n" for row in left_rows), encoding="utf-8")
    right_path.write_text("".join(json.dumps(row) + "\n" for row in right_rows), encoding="utf-8")
    requests_path.write_text(
        "".join(json.dumps(row) + "\n" for row in request_rows),
        encoding="utf-8",
    )

    summary = compare_prediction_sets(
        left_path,
        right_path,
        requests_path,
        tmp_path / "compare.json",
    )
    assert summary["document_count"] == 1
    assert summary["macro_delta_mean"] > 0
    assert summary["work_delta_mean"] > 0
    assert summary["macro_win_count"] == 1
    assert summary["work_win_count"] == 1


def test_work_record_features_counts_complete_and_partial_rows() -> None:
    features = _work_record_features(
        [
            {
                "job_title": "Engineer",
                "company": "Acme",
                "start_date": "2022-01",
                "end_date": "2024-01",
                "duration": "2022-01 - 2024-01",
            },
            {
                "job_title": "Developer",
                "company": "",
                "start_date": "2021-01",
                "end_date": "",
                "duration": "",
            },
        ]
    )
    assert features == {
        "record_count": 2,
        "complete": 1,
        "partial": 1,
        "empty_title": 0,
        "empty_company": 1,
        "has_dates": 2,
        "suspicious": 0,
    }


def test_accept_sgrse_work_requires_more_complete_records() -> None:
    baseline = [
        {
            "job_title": "Engineer",
            "company": "",
            "start_date": "2022-01",
            "end_date": "",
            "duration": "",
        }
    ]
    sgrse = [
        {
            "job_title": "Engineer",
            "company": "Acme",
            "start_date": "2022-01",
            "end_date": "2024-01",
            "duration": "2022-01 - 2024-01",
        }
    ]
    assert _accept_sgrse_work(baseline, sgrse) is True
    assert _accept_sgrse_work(sgrse, baseline) is False


def test_apply_selective_sgrse_work_decoder_only_accepts_better_work_sets(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    sgrse_path = tmp_path / "sgrse.jsonl"
    truth = {
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "location": "Cairo",
        "phone": "+201234567890",
        "linkedin_url": None,
        "github_url": None,
        "summary": "Engineer",
        "skills": ["Python"],
        "work_experience": [
            {
                "job_title": "Engineer",
                "company": "Acme",
                "start_date": "2022-01",
                "end_date": "2024-01",
                "duration": "2022-01 - 2024-01",
            }
        ],
        "education": [],
        "projects": None,
        "certifications": None,
    }
    baseline_rows = [
        {
            "cv_id": "cv_1",
            "prediction": {
                **truth,
                "work_experience": [
                    {
                        "job_title": "Engineer",
                        "company": "",
                        "start_date": "2022-01",
                        "end_date": "",
                        "duration": "",
                    }
                ],
            },
            "candidates": [],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        },
        {
            "cv_id": "cv_2",
            "prediction": truth,
            "candidates": [],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        },
    ]
    sgrse_rows = [
        {
            "cv_id": "cv_1",
            "prediction": truth,
            "candidates": [],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        },
        {
            "cv_id": "cv_2",
            "prediction": {
                **truth,
                "work_experience": [
                    {
                        "job_title": "Engineer",
                        "company": "",
                        "start_date": "2022-01",
                        "end_date": "",
                        "duration": "",
                    }
                ],
            },
            "candidates": [],
            "record_links": [],
            "selector_trace": None,
            "assembly_events": [],
        },
    ]
    baseline_path.write_text(
        "".join(json.dumps(row) + "\n" for row in baseline_rows),
        encoding="utf-8",
    )
    sgrse_path.write_text(
        "".join(json.dumps(row) + "\n" for row in sgrse_rows),
        encoding="utf-8",
    )

    summary = apply_selective_sgrse_work_decoder(
        baseline_path,
        sgrse_path,
        tmp_path / "selected.jsonl",
    )
    selected_rows = [
        json.loads(line)
        for line in (tmp_path / "selected.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["document_count"] == 2
    assert summary["accepted_count"] == 1
    assert summary["event_counts"]["sgrse_work_selected"] == 1
    assert selected_rows[0]["prediction"]["work_experience"] == truth["work_experience"]
    assert selected_rows[1]["prediction"]["work_experience"] == truth["work_experience"]

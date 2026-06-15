from pathlib import Path

from eraparse.evidence import (
    align_graph_to_fields,
    canonical_field_path,
    graph_from_pymupdf4llm_json,
    truth_field_annotations,
    validate_evidence_graphs,
)


def test_canonical_field_path_normalizes_nested_and_list_indices() -> None:
    assert canonical_field_path("contact_info.email") == ("email", None)
    assert canonical_field_path("skills.3") == ("skills", None)
    assert canonical_field_path("skills.3.skill_name") == ("skills", None)
    assert canonical_field_path("work_experience.2.company") == (
        "work_experience.*.company",
        2,
    )
    assert canonical_field_path("projects.1.technologies.2") == (
        "projects.*.technologies",
        1,
    )


def test_truth_field_annotations_cover_skills_and_nested_reduced_fields() -> None:
    annotations = truth_field_annotations(
        {
            "full_name": "Jane Doe",
            "skills": ["Python"],
            "work_experience": [{"company": "Acme", "job_title": "Engineer"}],
        }
    )
    assert {(item["field_path"], item["text"]) for item in annotations} >= {
        ("full_name", "Jane Doe"),
        ("skills.0.skill_name", "Python"),
        ("work_experience.0.company", "Acme"),
        ("work_experience.0.job_title", "Engineer"),
    }


def test_pymupdf4llm_json_becomes_ordered_word_evidence() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 200,
                    "boxes": [
                        {"textlines": [{"spans": [{"text": "Jane Doe", "bbox": [10, 20, 50, 40]}]}]}
                    ],
                }
            ],
        },
    )
    assert [unit.text for unit in graph.units] == ["Jane", "Doe"]
    assert graph.units[0].bbox_norm == (100, 100, 300, 200)
    assert graph.units[1].reading_order == 1


def test_pymupdf4llm_json_preserves_table_cells_with_null_textlines() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {
                            "textlines": None,
                            "table": {
                                "cells": [
                                    [[0, 0, 50, 20], [50, 0, 100, 20]],
                                    [[0, 20, 50, 40], [50, 20, 100, 40]],
                                ],
                                "extract": [
                                    ["Acme Corp", "2024"],
                                    ["Engineer", ""],
                                ],
                            },
                        }
                    ],
                }
            ],
        },
    )
    assert [unit.text for unit in graph.units] == ["Acme", "Corp", "2024", "Engineer"]
    assert graph.units[0].bbox_norm == (0, 0, 222, 200)
    assert graph.units[-1].bbox_norm == (0, 200, 500, 400)


def test_pymupdf4llm_json_reconstructs_words_split_across_styled_spans() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {
                            "textlines": [
                                {
                                    "bbox": [0, 0, 80, 10],
                                    "spans": [
                                        {"text": "J", "bbox": [0, 0, 5, 10]},
                                        {"text": "essica R", "bbox": [6, 0, 45, 10]},
                                        {"text": "obinson", "bbox": [46, 0, 80, 10]},
                                    ],
                                }
                            ]
                        }
                    ],
                }
            ],
        },
    )
    assert [unit.text for unit in graph.units] == ["Jessica", "Robinson"]


def test_alignment_uses_complete_ordered_sequences_for_repeated_values() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {
                            "textlines": [
                                {
                                    "spans": [
                                        {
                                            "text": "Engineer Acme Engineer Beta",
                                            "bbox": [0, 0, 100, 10],
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ],
        },
    )
    aligned = align_graph_to_fields(
        graph,
        [
            {
                "field_path": "work_experience.0.job_title",
                "text": "Engineer",
                "page": 1,
                "order_index": 0,
            },
            {
                "field_path": "work_experience.1.job_title",
                "text": "Engineer",
                "page": 1,
                "order_index": 1,
            },
        ],
    )
    labeled = [unit for unit in aligned.units if unit.field_path]
    assert [unit.record_index for unit in labeled] == [0, 1]


def test_alignment_matches_punctuation_rich_value_stored_as_one_unit() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {
                            "textlines": [
                                {
                                    "spans": [
                                        {
                                            "text": "jane.doe@example.com",
                                            "bbox": [0, 0, 50, 10],
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ],
        },
    )
    aligned = align_graph_to_fields(
        graph,
        [
            {
                "field_path": "contact_info.email",
                "text": "jane.doe@example.com",
                "page": 1,
                "order_index": 0,
            }
        ],
    )
    assert aligned.units[0].field_path == "email"


def test_alignment_does_not_absorb_leading_separator() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {
                            "textlines": [
                                {
                                    "spans": [
                                        {
                                            "text": "· jane@example.com",
                                            "bbox": [0, 0, 50, 10],
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ],
        },
    )
    aligned = align_graph_to_fields(
        graph,
        [
            {
                "field_path": "contact_info.email",
                "text": "jane@example.com",
                "page": 1,
                "order_index": 0,
            }
        ],
    )
    assert [unit.text for unit in aligned.units if unit.field_path == "email"] == [
        "jane@example.com"
    ]


def test_alignment_never_overwrites_existing_word_labels() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {
                            "textlines": [
                                {
                                    "spans": [
                                        {"text": "Python", "bbox": [0, 0, 20, 10]},
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ],
        },
    )
    graph = graph.model_copy(
        update={"units": [graph.units[0].model_copy(update={"field_path": "skills"})]}
    )
    aligned = align_graph_to_fields(
        graph,
        [
            {
                "field_path": "summary",
                "text": "Python",
                "page": 1,
                "order_index": 0,
            }
        ],
    )
    assert aligned.units[0].field_path == "skills"


def test_alignment_does_not_duplicate_an_already_satisfied_annotation() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {
                            "textlines": [
                                {
                                    "spans": [
                                        {
                                            "text": "Acme Engineer Acme",
                                            "bbox": [0, 0, 100, 10],
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ],
        },
    )
    graph = graph.model_copy(
        update={
            "units": [
                graph.units[0].model_copy(
                    update={"field_path": "work_experience.*.company", "record_index": 0}
                ),
                *graph.units[1:],
            ]
        }
    )
    aligned = align_graph_to_fields(
        graph,
        [{"field_path": "work_experience.0.company", "text": "Acme"}],
    )
    assert [unit.text for unit in aligned.units if unit.field_path] == ["Acme"]


def test_alignment_prefers_compact_ordered_truth_groups() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {
                            "textlines": [
                                {
                                    "spans": [
                                        {
                                            "text": (
                                                "Python summary filler React "
                                                "Skills Python React SQL"
                                            ),
                                            "bbox": [0, 0, 100, 10],
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ],
        },
    )
    annotations = truth_field_annotations({"skills": ["Python", "React", "SQL"]})
    aligned = align_graph_to_fields(graph, annotations)
    assert [
        index for index, unit in enumerate(aligned.units) if unit.field_path == "skills"
    ] == [5, 6, 7]


def test_evidence_validation_rejects_oracle_without_permission() -> None:
    graph = graph_from_pymupdf4llm_json(
        "cv_1",
        {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {"textlines": [{"spans": [{"text": "Jane", "bbox": [0, 0, 10, 10]}]}]}
                    ],
                }
            ],
        },
    ).model_copy(update={"oracle": True})
    assert not validate_evidence_graphs([graph])["passed"]
    assert validate_evidence_graphs([graph], allow_oracle=True)["passed"]


def test_dataset_backed_pymupdf4llm_graph_fixture_exists() -> None:
    path = Path("artifacts/representations/pymupdf4llm_json")
    assert path.is_dir()

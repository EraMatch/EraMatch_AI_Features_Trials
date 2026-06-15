from eraparse.compact_schema import (
    COMPACT_SCHEMA_TEMPLATE,
    compact_to_reduced,
    reduced_to_compact,
)


def test_compact_schema_round_trip_preserves_reduced_target(
    reduced_target: dict[str, object],
) -> None:
    compact = reduced_to_compact(reduced_target)
    assert compact["n"] == reduced_target["full_name"]
    assert compact["w"][0] == [
        "ML Engineer",
        "Example AI",
        "2022-01",
        "Present",
        "2022-01 - Present",
    ]
    assert compact_to_reduced(compact) == {**reduced_target, "github_url": None}


def test_compact_schema_expands_missing_and_malformed_values() -> None:
    expanded = compact_to_reduced(
        {
            "n": "Jane Doe",
            "e": None,
            "s": "Python",
            "w": [["Engineer", "Acme"]],
            "d": None,
            "p": [["Parser", ["Python"], None]],
            "c": [["AWS"]],
        }
    )
    assert expanded["full_name"] == "Jane Doe"
    assert expanded["email"] == ""
    assert expanded["skills"] == ["Python"]
    assert expanded["work_experience"] == [
        {
            "job_title": "Engineer",
            "company": "Acme",
            "start_date": "",
            "end_date": "",
            "duration": "",
        }
    ]
    assert expanded["education"] == []
    assert expanded["projects"] == [
        {"name": "Parser", "technologies": ["Python"], "url": None}
    ]
    assert expanded["certifications"] == [{"name": "AWS", "issuer": "", "date": ""}]


def test_compact_schema_template_has_only_canonical_aliases() -> None:
    assert set(COMPACT_SCHEMA_TEMPLATE) == {
        "n",
        "e",
        "l",
        "ph",
        "li",
        "gh",
        "su",
        "s",
        "w",
        "d",
        "p",
        "c",
    }

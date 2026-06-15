from eraparse.assembly import assemble_prediction
from eraparse.models import FieldCandidate


def candidate(
    path: str,
    value: object,
    confidence: float = 0.9,
    record_index: int | None = None,
) -> FieldCandidate:
    return FieldCandidate(
        schema_path=path,
        value=value,
        confidence=confidence,
        record_index=record_index,
    )


def test_assembler_guarantees_reduced_schema_and_groups_nested_records() -> None:
    grounded = assemble_prediction(
        "cv_1",
        [
            candidate("full_name", "Jane Doe"),
            candidate("email", "jane@example.com"),
            candidate("location", "Cairo"),
            candidate("phone", "123"),
            candidate("summary", "Engineer"),
            candidate("skills", "Python"),
            candidate("skills", "Python", 0.8),
            candidate("work_experience.*.job_title", "Engineer", record_index=0),
            candidate("work_experience.*.company", "Acme", record_index=0),
            candidate("education.*.degree", "BSc", record_index=0),
            candidate("education.*.institution", "University", record_index=0),
        ],
    )
    assert grounded.prediction.skills == ["Python"]
    assert grounded.prediction.work_experience[0].company == "Acme"
    assert grounded.prediction.education[0].degree == "BSc"


def test_assembler_excludes_nested_candidate_without_record_index() -> None:
    grounded = assemble_prediction(
        "cv_1",
        [
            candidate("full_name", ""),
            candidate("email", ""),
            candidate("location", ""),
            candidate("phone", ""),
            candidate("summary", ""),
            candidate("work_experience.*.company", "Acme"),
        ],
    )
    assert grounded.prediction.work_experience == []
    assert grounded.assembly_events[0].kind == "missing_record_index"


def test_assembler_splits_set_values_and_trims_layout_punctuation() -> None:
    grounded = assemble_prediction(
        "cv_1",
        [
            candidate("full_name", "Jane Doe"),
            candidate("email", "jane@example.com"),
            candidate("location", "Cairo"),
            candidate("phone", "123"),
            candidate("summary", "Engineer"),
            candidate("skills", "[Python, Cloudflare Pages]"),
            candidate("projects.*.name", "Parser", record_index=0),
            candidate(
                "projects.*.technologies",
                "[Python, TypeScript, React]",
                record_index=0,
            ),
            candidate("certifications.*.issuer", "Meta,", record_index=0),
        ],
    )
    assert grounded.prediction.skills == ["Cloudflare Pages", "Python"]
    assert grounded.prediction.projects is not None
    assert grounded.prediction.projects[0].technologies == ["Python", "React", "TypeScript"]
    assert grounded.prediction.certifications is not None
    assert grounded.prediction.certifications[0].issuer == "Meta"

from collections.abc import Mapping, Sequence
from typing import Any

from eraparse.constants import REDUCED_SCHEMA_TEMPLATE

COMPACT_SCHEMA_TEMPLATE: dict[str, Any] = {
    "n": "",
    "e": "",
    "l": "",
    "ph": "",
    "li": "",
    "gh": "",
    "su": "",
    "s": [],
    "w": [["", "", "", "", ""]],
    "d": [["", "", "", ""]],
    "p": [["", [], ""]],
    "c": [["", "", ""]],
}


def _text(value: Any) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


def _optional_text(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Sequence):
        return []
    return [text for item in value if (text := _text(item))]


def _row(value: Any, width: int) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return [None] * width
    return [*value[:width], *([None] * max(0, width - len(value)))]


def reduced_to_compact(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n": _text(target.get("full_name")),
        "e": _text(target.get("email")),
        "l": _text(target.get("location")),
        "ph": _text(target.get("phone")),
        "li": _optional_text(target.get("linkedin_url")),
        "gh": _optional_text(target.get("github_url")),
        "su": _text(target.get("summary")),
        "s": _strings(target.get("skills")),
        "w": [
            [
                _text(record.get("job_title")),
                _text(record.get("company")),
                _text(record.get("start_date")),
                _text(record.get("end_date")),
                _text(record.get("duration")),
            ]
            for record in target.get("work_experience", [])
            if isinstance(record, Mapping)
        ],
        "d": [
            [
                _text(record.get("degree")),
                _text(record.get("field_of_study")),
                _text(record.get("institution")),
                _text(record.get("graduation_date")),
            ]
            for record in target.get("education", [])
            if isinstance(record, Mapping)
        ],
        "p": None
        if target.get("projects") is None
        else [
            [
                _text(record.get("name")),
                _strings(record.get("technologies")),
                _optional_text(record.get("url")),
            ]
            for record in target.get("projects", [])
            if isinstance(record, Mapping)
        ],
        "c": None
        if target.get("certifications") is None
        else [
            [
                _text(record.get("name")),
                _text(record.get("issuer")),
                _text(record.get("date")),
            ]
            for record in target.get("certifications", [])
            if isinstance(record, Mapping)
        ],
    }


def compact_to_reduced(compact: Mapping[str, Any]) -> dict[str, Any]:
    work = [_row(record, 5) for record in compact.get("w", []) or []]
    education = [_row(record, 4) for record in compact.get("d", []) or []]
    projects = None
    if compact.get("p") is not None:
        projects = [_row(record, 3) for record in compact.get("p", []) or []]
    certifications = None
    if compact.get("c") is not None:
        certifications = [_row(record, 3) for record in compact.get("c", []) or []]

    expanded = {
        "full_name": _text(compact.get("n")),
        "email": _text(compact.get("e")),
        "location": _text(compact.get("l")),
        "phone": _text(compact.get("ph")),
        "linkedin_url": _optional_text(compact.get("li")),
        "github_url": _optional_text(compact.get("gh")),
        "summary": _text(compact.get("su")),
        "skills": _strings(compact.get("s")),
        "work_experience": [
            {
                "job_title": _text(row[0]),
                "company": _text(row[1]),
                "start_date": _text(row[2]),
                "end_date": _text(row[3]),
                "duration": _text(row[4]),
            }
            for row in work
        ],
        "education": [
            {
                "degree": _text(row[0]),
                "field_of_study": _text(row[1]),
                "institution": _text(row[2]),
                "graduation_date": _text(row[3]),
            }
            for row in education
        ],
        "projects": None
        if projects is None
        else [
            {
                "name": _text(row[0]),
                "technologies": _strings(row[1]),
                "url": _optional_text(row[2]),
            }
            for row in projects
        ],
        "certifications": None
        if certifications is None
        else [
            {"name": _text(row[0]), "issuer": _text(row[1]), "date": _text(row[2])}
            for row in certifications
        ],
    }
    return {key: expanded[key] for key in REDUCED_SCHEMA_TEMPLATE}

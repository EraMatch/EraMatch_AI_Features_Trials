import re
from collections import defaultdict
from collections.abc import Iterable

from eraparse.models import FieldCandidate, GroundedPrediction, ReducedCVTarget, ValidationEvent


def _best(candidates: Iterable[FieldCandidate]) -> FieldCandidate | None:
    return max(candidates, key=lambda item: item.confidence, default=None)


def _clean(value: object) -> str:
    return str(value).strip().strip("[]{}|·,; ")


def _value(candidates: Iterable[FieldCandidate], default: str = "") -> str:
    candidate = _best(candidates)
    return _clean(candidate.value) if candidate is not None else default


def _set_values(candidates: Iterable[FieldCandidate]) -> list[str]:
    values = set()
    for candidate in candidates:
        for item in re.split(r"[,;|]", str(candidate.value).strip().strip("[]{}")):
            cleaned = _clean(item)
            if cleaned:
                values.add(cleaned)
    return sorted(values)


def _record_defaults(kind: str) -> dict[str, object]:
    if kind == "work_experience":
        return {"job_title": "", "company": "", "start_date": "", "end_date": "", "duration": ""}
    if kind == "education":
        return {"degree": "", "field_of_study": "", "institution": "", "graduation_date": ""}
    if kind == "projects":
        return {"name": "", "technologies": [], "url": None}
    return {"name": "", "issuer": "", "date": ""}


def assemble_prediction(cv_id: str, candidates: list[FieldCandidate]) -> GroundedPrediction:
    by_path: dict[str, list[FieldCandidate]] = defaultdict(list)
    nested: dict[str, dict[int, dict[str, list[FieldCandidate]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    events: list[ValidationEvent] = []
    for candidate in candidates:
        match = re.match(
            r"^(work_experience|education|projects|certifications)\.\*\.(.+)$",
            candidate.schema_path,
        )
        if match:
            if candidate.record_index is None:
                events.append(
                    ValidationEvent(
                        kind="missing_record_index",
                        path=candidate.schema_path,
                        message="nested candidate was excluded because it has no record index",
                    )
                )
                continue
            nested[match.group(1)][candidate.record_index][match.group(2)].append(candidate)
        else:
            by_path[candidate.schema_path].append(candidate)

    skills = _set_values(by_path["skills"])
    values: dict[str, object] = {
        "full_name": _value(by_path["full_name"]),
        "email": _value(by_path["email"]),
        "location": _value(by_path["location"]),
        "phone": _value(by_path["phone"]),
        "linkedin_url": _value(by_path["linkedin_url"]) or None,
        "github_url": _value(by_path["github_url"]) or None,
        "summary": _value(by_path["summary"]),
        "skills": skills,
    }
    for kind in ("work_experience", "education", "projects", "certifications"):
        records = []
        for record_index in sorted(nested[kind]):
            record = _record_defaults(kind)
            for field, field_candidates in nested[kind][record_index].items():
                if field == "technologies":
                    record[field] = _set_values(field_candidates)
                else:
                    record[field] = _value(field_candidates)
            records.append(record)
        if kind in {"work_experience", "education"}:
            values[kind] = records
        else:
            values[kind] = records or None
    prediction = ReducedCVTarget.model_validate(values)
    return GroundedPrediction(
        cv_id=cv_id,
        prediction=prediction,
        candidates=candidates,
        assembly_events=events,
    )

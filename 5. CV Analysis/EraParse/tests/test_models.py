import pytest
from pydantic import ValidationError

from eraparse.models import ReducedCVTarget


def test_reduced_schema_accepts_missing_optional_keys(reduced_target: dict[str, object]) -> None:
    reduced_target.pop("projects")
    reduced_target.pop("certifications")
    reduced_target.pop("linkedin_url")
    model = ReducedCVTarget.model_validate(reduced_target)
    assert model.projects is None
    assert model.linkedin_url is None


def test_reduced_schema_accepts_nullable_arrays(reduced_target: dict[str, object]) -> None:
    model = ReducedCVTarget.model_validate(reduced_target)
    assert model.projects is None
    assert model.certifications is None


def test_reduced_schema_rejects_missing_required_field(reduced_target: dict[str, object]) -> None:
    reduced_target.pop("email")
    with pytest.raises(ValidationError):
        ReducedCVTarget.model_validate(reduced_target)


def test_reduced_schema_rejects_extra_keys(reduced_target: dict[str, object]) -> None:
    reduced_target["unknown"] = "value"
    with pytest.raises(ValidationError):
        ReducedCVTarget.model_validate(reduced_target)

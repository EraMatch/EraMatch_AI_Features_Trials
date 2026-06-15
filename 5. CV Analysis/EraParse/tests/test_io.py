from pathlib import Path

from eraparse.io import atomic_write_json, read_json, stable_hash


def test_stable_hash_is_repeatable_and_seeded() -> None:
    assert stable_hash("cv_1", seed=1) == stable_hash("cv_1", seed=1)
    assert stable_hash("cv_1", seed=1) != stable_hash("cv_1", seed=2)


def test_atomic_json_write(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "value.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert read_json(path) == {"a": 1, "b": 2}

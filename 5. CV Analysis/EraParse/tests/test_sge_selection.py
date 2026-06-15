from eraparse.sge_selection import evidence_support_fraction, select_overfit_records


def record(cv_id: str, words: list[str], truth: dict[str, object]) -> dict[str, object]:
    return {
        "cv_id": cv_id,
        "page": 1,
        "words": words,
        "truth": truth,
    }


def test_evidence_support_fraction_measures_truth_values_present_in_words() -> None:
    rows = [
        record(
            "cv_1",
            ["Jane", "Doe", "Python"],
            {"full_name": "Jane Doe", "skills": ["Python"]},
        )
    ]
    assert evidence_support_fraction(rows) == 1.0


def test_select_overfit_records_keeps_all_pages_for_best_supported_documents() -> None:
    rows = [
        record("low", ["Jane"], {"full_name": "Jane", "skills": ["Missing"]}),
        record("high", ["John"], {"full_name": "John"}),
        {**record("high", ["Python"], {"full_name": "John"}), "page": 2},
    ]
    selected, summary = select_overfit_records(rows, document_count=1, min_coverage=0.9)
    assert [row["page"] for row in selected] == [1, 2]
    assert summary["selected_ids"] == ["high"]

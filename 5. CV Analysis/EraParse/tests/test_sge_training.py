import pytest

from eraparse.sge_training import (
    _load_model,
    first_subword_positions,
    resolve_primary_decoder,
    training_schedule,
    unfreeze_final_layoutlmv3_layers,
    validate_resume_checkpoint,
)


def test_training_schedule_cycles_tiny_subset_for_overfit() -> None:
    records = [{"cv_id": "a"}, {"cv_id": "b"}]
    assert [row["cv_id"] for row in training_schedule(records, max_steps=5)] == [
        "a",
        "b",
        "a",
        "b",
        "a",
    ]


def test_training_schedule_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="at least one"):
        training_schedule([], max_steps=1)


def test_training_schedule_allows_evaluation_only_zero_steps() -> None:
    assert training_schedule([{"cv_id": "a"}], max_steps=0) == []


def test_model_loader_rejects_negative_unfreeze_count_before_loading() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        _load_model("sge", unfreeze_final_layers=-1)


def test_model_loader_rejects_negative_loss_weight_before_loading() -> None:
    with pytest.raises(ValueError, match="grouping_weight cannot be negative"):
        _load_model("sge", grouping_weight=-1)


def test_model_loader_rejects_zero_query_layers_before_loading() -> None:
    with pytest.raises(ValueError, match="query_layers must be at least one"):
        _load_model("sge", query_layers=0)


def test_unfreeze_final_layoutlmv3_layers_changes_only_requested_layers() -> None:
    class Parameter:
        requires_grad = False

    class Layer:
        def __init__(self) -> None:
            self.parameter = Parameter()

        def parameters(self) -> list[Parameter]:
            return [self.parameter]

    layers = [Layer() for _ in range(5)]
    model = type(
        "Model",
        (),
        {
            "layoutlmv3": type(
                "LayoutLMv3",
                (),
                {"encoder": type("Encoder", (), {"layer": layers})()},
            )()
        },
    )()
    unfreeze_final_layoutlmv3_layers(model, 2)
    assert [layer.parameter.requires_grad for layer in layers] == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_first_subword_positions_excludes_special_and_repeated_tokens() -> None:
    assert first_subword_positions([None, 0, 0, 1, None, 2]) == [1, 3, 5]


def test_primary_decoder_defaults_to_deterministic_sequence() -> None:
    assert resolve_primary_decoder("sge", "auto") == "sequence"
    assert resolve_primary_decoder("sge", "learned") == "learned"
    assert resolve_primary_decoder("baseline", "auto") == "sequence"
    with pytest.raises(ValueError, match="does not support"):
        resolve_primary_decoder("baseline", "learned")


def test_resume_checkpoint_rejects_changed_loss_weights() -> None:
    checkpoint = {
        "mode": "sge",
        "seed": 7,
        "unfreeze_final_layers": 4,
        "query_layers": 2,
        "loss_weights": {
            "field_token": 1.0,
            "presence": 0.0,
            "grouping": 0.0,
            "evidence": 0.0,
        },
    }
    with pytest.raises(ValueError, match="loss_weights"):
        validate_resume_checkpoint(
            checkpoint,
            mode="sge",
            seed=7,
            unfreeze_final_layers=4,
            query_layers=2,
            presence_weight=0.25,
            grouping_weight=0.0,
            evidence_weight=0.0,
        )

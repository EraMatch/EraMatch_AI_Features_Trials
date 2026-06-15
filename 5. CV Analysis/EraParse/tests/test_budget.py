import pytest

from eraparse.budget import projected_t4_cost, require_budget


def test_t4_projection_uses_safety_factor() -> None:
    assert projected_t4_cost(3600) == pytest.approx(0.5904 * 1.3)


def test_budget_gate_rejects_excess_stage_spend() -> None:
    with pytest.raises(RuntimeError):
        require_budget(0.4, spent=2.7, budget=3.0)


def test_budget_gate_rejects_projection_beyond_available_credit() -> None:
    with pytest.raises(RuntimeError, match="available Modal credit"):
        require_budget(0.4, spent=0.0, budget=3.0, remaining_credit=0.3)

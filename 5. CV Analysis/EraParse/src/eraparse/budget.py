from eraparse.constants import CURRENT_STAGE_BUDGET_USD, T4_GPU_RATE_PER_SECOND


def projected_t4_cost(
    seconds: float,
    *,
    safety_factor: float = 1.3,
    rate_per_second: float = T4_GPU_RATE_PER_SECOND,
) -> float:
    if seconds < 0:
        raise ValueError("seconds must not be negative")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least 1")
    return seconds * rate_per_second * safety_factor


def require_budget(
    projected_cost: float,
    *,
    spent: float,
    budget: float = CURRENT_STAGE_BUDGET_USD,
    remaining_credit: float | None = None,
) -> None:
    values = [projected_cost, spent, budget]
    if remaining_credit is not None:
        values.append(remaining_credit)
    if min(values) < 0:
        raise ValueError("budget values must not be negative")
    if spent + projected_cost > budget:
        raise RuntimeError(
            f"projected stage spend ${spent + projected_cost:.2f} exceeds ${budget:.2f} budget"
        )
    if remaining_credit is not None and projected_cost > remaining_credit:
        raise RuntimeError(
            f"projected run cost ${projected_cost:.2f} exceeds "
            f"${remaining_credit:.2f} available Modal credit"
        )

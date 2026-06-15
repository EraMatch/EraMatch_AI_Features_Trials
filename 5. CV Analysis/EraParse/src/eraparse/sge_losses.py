def token_class_weights(num_fields: int, *, outside_weight: float = 0.05) -> list[float]:
    if num_fields < 1:
        raise ValueError("num_fields must be at least one")
    if not 0 < outside_weight <= 1:
        raise ValueError("outside_weight must be within (0, 1]")
    return [outside_weight, *([1.0] * num_fields)]


def binary_positive_weight(
    *,
    positive_count: int,
    negative_count: int,
    cap: float = 20.0,
) -> float:
    if min(positive_count, negative_count) < 0 or cap < 1:
        raise ValueError("counts must be non-negative and cap must be at least one")
    if positive_count == 0:
        return 1.0
    return min(negative_count / positive_count, cap)

def gradient_accumulation_step(
    micro_step: int,
    *,
    total_micro_steps: int,
    accumulation_steps: int,
) -> tuple[bool, float]:
    if micro_step < 1 or micro_step > total_micro_steps:
        raise ValueError("micro_step must be within the training schedule")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    if micro_step % accumulation_steps == 0:
        return True, 1.0
    if micro_step != total_micro_steps:
        return False, 1.0
    remainder = total_micro_steps % accumulation_steps
    return True, accumulation_steps / remainder

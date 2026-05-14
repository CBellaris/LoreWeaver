"""Cost estimation helpers for model services."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_yuan_per_1k: float
    output_yuan_per_1k: float
    estimated_yuan: float


def estimate_cost(
    usage: dict[str, int],
    *,
    input_yuan_per_1k: float = 0.0,
    output_yuan_per_1k: float = 0.0,
) -> CostEstimate:
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    estimated_yuan = (
        (input_tokens / 1000.0) * input_yuan_per_1k
        + (output_tokens / 1000.0) * output_yuan_per_1k
    )
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_yuan_per_1k=input_yuan_per_1k,
        output_yuan_per_1k=output_yuan_per_1k,
        estimated_yuan=round(estimated_yuan, 8),
    )

"""Token and cost accounting helpers for extraction."""

from __future__ import annotations

import re

from loreweaver.extraction.types import CostEstimate, TokenPrice


def estimate_tokens(text: str) -> int:
    """Cheap provider-agnostic token estimate used when API usage is unavailable."""
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    non_cjk_text = "".join(" " if "\u4e00" <= char <= "\u9fff" else char for char in text)
    word_like = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", non_cjk_text)
    return max(1, cjk_chars + len(word_like))


def estimate_cost(usage: dict[str, int], price: TokenPrice) -> CostEstimate:
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    estimated_yuan = (
        input_tokens / 1000 * price.input_yuan_per_1k
        + output_tokens / 1000 * price.output_yuan_per_1k
    )
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_yuan_per_1k=price.input_yuan_per_1k,
        output_yuan_per_1k=price.output_yuan_per_1k,
        estimated_yuan=round(estimated_yuan, 6),
    )

def _usage_or_estimate(
    usage: dict[str, int],
    messages: list[dict[str, str]],
    raw_output: str,
) -> dict[str, int]:
    if usage.get("input_tokens") or usage.get("output_tokens"):
        return usage
    input_tokens = estimate_tokens("\n".join(message["content"] for message in messages))
    output_tokens = estimate_tokens(raw_output)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": int(left.get("input_tokens", 0)) + int(right.get("input_tokens", 0)),
        "output_tokens": int(left.get("output_tokens", 0)) + int(right.get("output_tokens", 0)),
        "total_tokens": int(left.get("total_tokens", 0)) + int(right.get("total_tokens", 0)),
    }


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

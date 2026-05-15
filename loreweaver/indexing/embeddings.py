"""Embedding input and cache-key helpers for M1.4."""

from __future__ import annotations

import hashlib
import math

from loreweaver.model_services.config import ModelServiceConfig
from loreweaver.models.span import Span


def build_embedding_input(
    span: Span,
    *,
    include_key_quote: bool = False,
    include_located_text: bool = False,
) -> str:
    parts = [
        f"summary: {span.summary}",
        f"entities: {', '.join(span.entities)}",
    ]
    if include_key_quote and span.key_quote:
        parts.append(f"key_quote: {span.key_quote}")
    if include_located_text and span.located_text:
        parts.append(f"located_text: {span.located_text}")
    return "\n".join(part for part in parts if part.strip())


def embedding_cache_key(settings: ModelServiceConfig, input_text: str) -> tuple[str, str]:
    input_sha256 = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    dimension_key = settings.expected_dimensions if settings.expected_dimensions else "auto"
    raw_key = f"{settings.provider.name}:{settings.model}:{dimension_key}:{input_sha256}"
    cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return cache_key, input_sha256


def estimate_embedding_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))

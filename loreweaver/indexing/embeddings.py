"""Embedding clients and input builders for M1.4."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol

from loreweaver.config import AppConfig
from loreweaver.models.span import Span


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    api_key_env: str | None
    base_url: str | None
    expected_dimensions: int | None
    batch_size: int
    input_yuan_per_1k: float
    use_dimensions_param: bool

    @property
    def cache_model_key(self) -> str:
        dimension_key = self.expected_dimensions if self.expected_dimensions else "auto"
        return f"{self.provider}:{self.model}:{dimension_key}"


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    usage: dict[str, int]


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        """Embed texts and return vectors in the same order."""


class OpenAICompatibleEmbeddingClient:
    """Embedding client for OpenAI-compatible providers such as SiliconFlow."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        if not settings.api_key_env:
            raise ValueError(f"Provider {settings.provider} does not define api_key_env")
        api_key = os.environ.get(settings.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {settings.api_key_env}")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "The openai package is required for live embedding calls. "
                "Install optional M1 dependencies first."
            ) from error

        self._client = OpenAI(api_key=api_key, base_url=settings.base_url)
        self._settings = settings

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(vectors=[], usage={})

        request: dict[str, Any] = {
            "model": self._settings.model,
            "input": texts,
        }
        if self._settings.use_dimensions_param and self._settings.expected_dimensions:
            request["dimensions"] = self._settings.expected_dimensions

        response = self._client.embeddings.create(**request)
        data = sorted(response.data, key=lambda item: int(getattr(item, "index", 0) or 0))
        vectors = [[float(value) for value in item.embedding] for item in data]
        usage = {}
        if response.usage is not None:
            usage = {
                "input_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
            }
        return EmbeddingBatch(vectors=vectors, usage=usage)


class MockEmbeddingClient:
    """Deterministic embedding client for tests and no-API plumbing checks."""

    def __init__(self, dimensions: int = 8) -> None:
        if dimensions <= 0:
            raise ValueError("Mock embedding dimensions must be positive.")
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        vectors = [_hash_embedding(text, self.dimensions) for text in texts]
        usage = {
            "input_tokens": sum(estimate_embedding_tokens(text) for text in texts),
            "total_tokens": sum(estimate_embedding_tokens(text) for text in texts),
        }
        return EmbeddingBatch(vectors=vectors, usage=usage)


def embedding_settings_from_configs(
    *,
    config: AppConfig,
    models_config: AppConfig,
) -> EmbeddingSettings:
    indexing_config = config.values.get("indexing", {})
    model_settings = models_config.values.get("models", {}).get("embedding", {})
    provider = str(model_settings.get("provider", "siliconflow"))
    provider_settings = models_config.values.get("providers", {}).get(provider, {})

    expected_dimensions = model_settings.get(
        "expected_dimensions",
        indexing_config.get("embedding_dimensions"),
    )
    if expected_dimensions is not None:
        expected_dimensions = int(expected_dimensions)

    batch_size = int(
        indexing_config.get(
            "embedding_batch_size",
            model_settings.get("batch_size", 32),
        )
    )

    return EmbeddingSettings(
        provider=provider,
        model=str(model_settings.get("name", "")),
        api_key_env=provider_settings.get("api_key_env"),
        base_url=provider_settings.get("base_url"),
        expected_dimensions=expected_dimensions,
        batch_size=batch_size,
        input_yuan_per_1k=float(model_settings.get("input_yuan_per_1k", 0.0)),
        use_dimensions_param=bool(model_settings.get("use_dimensions_param", False)),
    )


def build_embedding_input(
    span: Span,
    *,
    include_key_quote: bool = False,
    include_located_text: bool = False,
) -> str:
    parts = [
        f"micro_topic: {span.micro_topic}",
        f"micro_summary: {span.micro_summary}",
        f"entities: {', '.join(span.entities)}",
        f"topics: {', '.join(span.topics)}",
    ]
    if include_key_quote and span.key_quote:
        parts.append(f"key_quote: {span.key_quote}")
    if include_located_text and span.located_text:
        parts.append(f"located_text: {span.located_text}")
    return "\n".join(part for part in parts if part.strip())


def embedding_cache_key(settings: EmbeddingSettings, input_text: str) -> tuple[str, str]:
    input_sha256 = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    raw_key = f"{settings.cache_model_key}:{input_sha256}"
    cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return cache_key, input_sha256


def estimate_embedding_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    raw = bytearray()
    seed = text.encode("utf-8")
    counter = 0
    while len(raw) < dimensions:
        raw.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    vector = [(byte / 255.0) - 0.5 for byte in raw[:dimensions]]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


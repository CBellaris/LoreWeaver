"""Embedding clients and input builders for M1.4."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Protocol

from loreweaver.config import AppConfig
from loreweaver.model_services import ModelServiceFactory
from loreweaver.model_services.clients.openai_compatible import OpenAICompatibleClient
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
        self._settings = settings
        self._client = OpenAICompatibleClient(
            _service_config_from_embedding_settings(settings)
        )

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(vectors=[], usage={})

        result = self._client.embed(texts)
        return EmbeddingBatch(vectors=result.vectors, usage=result.usage)


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
    service_config = ModelServiceFactory.from_configs(
        config=config,
        models_config=models_config,
    ).resolve("embedding")
    return EmbeddingSettings(
        provider=service_config.provider.name,
        model=service_config.model,
        api_key_env=service_config.api_key_env,
        base_url=service_config.base_url,
        expected_dimensions=service_config.expected_dimensions,
        batch_size=service_config.batch_size,
        input_yuan_per_1k=service_config.pricing.input_yuan_per_1k,
        use_dimensions_param=service_config.use_dimensions_param,
    )


def build_embedding_input(
    span: Span,
    *,
    include_key_quote: bool = False,
    include_located_text: bool = False,
) -> str:
    parts = [
        f"summary: {span.summary}",
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


def _service_config_from_embedding_settings(settings: EmbeddingSettings):
    from loreweaver.model_services.config import (
        ModelServiceConfig,
        PricingConfig,
        ProviderConfig,
    )

    return ModelServiceConfig(
        service="embedding",
        capability="embedding",
        provider=ProviderConfig(
            name=settings.provider,
            adapter="openai_compatible",
            api_key_env=settings.api_key_env,
            base_url=settings.base_url,
        ),
        model=settings.model,
        expected_dimensions=settings.expected_dimensions,
        batch_size=settings.batch_size,
        use_dimensions_param=settings.use_dimensions_param,
        pricing=PricingConfig(input_yuan_per_1k=settings.input_yuan_per_1k),
    )

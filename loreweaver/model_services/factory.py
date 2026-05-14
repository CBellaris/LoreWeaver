"""Factory for model service clients."""

from __future__ import annotations

from dataclasses import replace

from loreweaver.config import AppConfig
from loreweaver.model_services.clients.mock import (
    MockChatModel,
    MockEmbeddingModel,
    MockRerankModel,
    NoopRerankModel,
)
from loreweaver.model_services.clients.openai_compatible import OpenAICompatibleClient
from loreweaver.model_services.clients.rerank_http import HttpRerankClient
from loreweaver.model_services.config import ModelServiceConfig, resolve_model_service


class ModelServiceFactory:
    def __init__(self, *, config: AppConfig | None, models_config: AppConfig) -> None:
        self.config = config
        self.models_config = models_config

    @classmethod
    def from_configs(
        cls,
        *,
        models_config: AppConfig,
        config: AppConfig | None = None,
    ) -> "ModelServiceFactory":
        return cls(config=config, models_config=models_config)

    def resolve(self, service: str) -> ModelServiceConfig:
        return resolve_model_service(
            models_config=self.models_config,
            app_config=self.config,
            service=service,
        )

    def chat(self, service: str, *, mock: bool = False):
        service_config = self.resolve(service)
        if mock or service_config.provider.name == "mock":
            return MockChatModel(model=f"mock::{service_config.model or service}")
        return OpenAICompatibleClient(service_config)

    def embedding(self, service: str = "embedding", *, mock: bool = False):
        service_config = self.resolve(service)
        if mock or service_config.provider.name == "mock":
            return MockEmbeddingModel(
                model=f"mock::{service_config.model or service}",
                dimensions=service_config.expected_dimensions or 8,
            )
        return OpenAICompatibleClient(service_config)

    def reranker(self, service: str = "reranker", *, mock: bool = False, disabled: bool = False):
        service_config = self.resolve(service)
        if disabled or not service_config.enabled or service_config.provider.name == "noop":
            return NoopRerankModel()
        if mock or service_config.provider.name == "mock":
            return MockRerankModel(model=f"mock::{service_config.model or service}")
        adapter = service_config.provider.adapter
        if adapter == "openai_compatible" and service_config.capability == "rerank":
            service_config = replace(
                service_config,
                provider=replace(service_config.provider, adapter="http_rerank"),
            )
            adapter = "http_rerank"
        if adapter == "http_rerank":
            return HttpRerankClient(service_config)
        return HttpRerankClient(service_config)

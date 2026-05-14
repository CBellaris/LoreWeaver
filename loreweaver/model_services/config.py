"""Resolve model service configuration from current and future config shapes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from loreweaver.config import AppConfig


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    adapter: str
    api_key_env: str | None
    base_url: str | None


@dataclass(frozen=True)
class PricingConfig:
    input_yuan_per_1k: float = 0.0
    output_yuan_per_1k: float = 0.0


@dataclass(frozen=True)
class ModelServiceConfig:
    service: str
    capability: str
    provider: ProviderConfig
    model: str
    temperature: float | None = None
    max_output_tokens: int | None = None
    json_response_format: bool = False
    enabled: bool = True
    timeout_seconds: float = 30.0
    expected_dimensions: int | None = None
    batch_size: int = 32
    use_dimensions_param: bool = False
    pricing: PricingConfig = PricingConfig()
    batch_model: str | None = None
    batch_pricing: PricingConfig = PricingConfig()
    extra: dict[str, Any] | None = None

    @property
    def api_key_env(self) -> str | None:
        return self.provider.api_key_env

    @property
    def base_url(self) -> str | None:
        return self.provider.base_url


def resolve_model_service(
    *,
    models_config: AppConfig,
    service: str,
    app_config: AppConfig | None = None,
) -> ModelServiceConfig:
    values = models_config.values
    if values.get("services") or values.get("model_profiles"):
        resolved = _resolve_new_style(values, service)
    else:
        resolved = _resolve_legacy(values, service)
    if app_config is not None:
        resolved = _apply_app_overrides(resolved, app_config)
    return resolved


def _resolve_new_style(values: dict[str, Any], service: str) -> ModelServiceConfig:
    service_values = dict(values.get("services", {}).get(service, {}))
    profile_name = service_values.pop("profile", None)
    profile_values: dict[str, Any] = {}
    if profile_name:
        profile_values = dict(values.get("model_profiles", {}).get(str(profile_name), {}))
        if not profile_values:
            raise ValueError(f"Unknown model profile for service {service}: {profile_name}")
    merged = {**profile_values, **service_values}
    return _config_from_values(values=values, service=service, service_values=merged)


def _resolve_legacy(values: dict[str, Any], service: str) -> ModelServiceConfig:
    service_values = dict(values.get("models", {}).get(service, {}))
    return _config_from_values(values=values, service=service, service_values=service_values)


def _config_from_values(
    *,
    values: dict[str, Any],
    service: str,
    service_values: dict[str, Any],
) -> ModelServiceConfig:
    provider_name = str(service_values.get("provider", _default_provider(service)))
    provider = _provider_from_values(values, provider_name)
    capability = str(service_values.get("capability", _default_capability(service)))
    pricing_values = dict(service_values.get("pricing", {}))
    batch_pricing_values = dict(service_values.get("batch_pricing", {}))
    if not pricing_values:
        pricing_values = {
            "input_yuan_per_1k": service_values.get("input_yuan_per_1k", 0.0),
            "output_yuan_per_1k": service_values.get("output_yuan_per_1k", 0.0),
        }
    if not batch_pricing_values:
        batch_pricing_values = {
            "input_yuan_per_1k": service_values.get("batch_input_yuan_per_1k", 0.0),
            "output_yuan_per_1k": service_values.get("batch_output_yuan_per_1k", 0.0),
        }

    expected_dimensions = service_values.get("expected_dimensions")
    if expected_dimensions is not None:
        expected_dimensions = int(expected_dimensions)

    return ModelServiceConfig(
        service=service,
        capability=capability,
        provider=provider,
        model=str(service_values.get("model", service_values.get("name", ""))),
        temperature=_optional_float(service_values.get("temperature")),
        max_output_tokens=_optional_int(service_values.get("max_output_tokens")),
        json_response_format=bool(service_values.get("json_response_format", False)),
        enabled=bool(service_values.get("enabled", True)),
        timeout_seconds=float(service_values.get("timeout_seconds", 30)),
        expected_dimensions=expected_dimensions,
        batch_size=int(service_values.get("batch_size", 32)),
        use_dimensions_param=bool(service_values.get("use_dimensions_param", False)),
        pricing=PricingConfig(
            input_yuan_per_1k=float(pricing_values.get("input_yuan_per_1k", 0.0)),
            output_yuan_per_1k=float(pricing_values.get("output_yuan_per_1k", 0.0)),
        ),
        batch_model=_optional_str(service_values.get("batch_model", service_values.get("batch_name"))),
        batch_pricing=PricingConfig(
            input_yuan_per_1k=float(batch_pricing_values.get("input_yuan_per_1k", 0.0)),
            output_yuan_per_1k=float(batch_pricing_values.get("output_yuan_per_1k", 0.0)),
        ),
        extra=dict(service_values),
    )


def _provider_from_values(values: dict[str, Any], provider_name: str) -> ProviderConfig:
    provider_values = values.get("providers", {}).get(provider_name, {})
    return ProviderConfig(
        name=provider_name,
        adapter=str(provider_values.get("adapter", _default_adapter(provider_name))),
        api_key_env=provider_values.get("api_key_env", _default_api_key_env(provider_name)),
        base_url=provider_values.get("base_url"),
    )


def _apply_app_overrides(
    service_config: ModelServiceConfig,
    app_config: AppConfig,
) -> ModelServiceConfig:
    if service_config.service == "qa":
        qa_config = app_config.values.get("qa", {})
        model = str(qa_config.get("model") or service_config.model)
        temperature = (
            float(qa_config["temperature"])
            if "temperature" in qa_config
            else service_config.temperature
        )
        extra = dict(service_config.extra or {})
        if "require_citations" in qa_config:
            extra["require_citations"] = bool(qa_config["require_citations"])
        return replace(service_config, model=model, temperature=temperature, extra=extra)

    if service_config.service == "embedding":
        indexing_config = app_config.values.get("indexing", {})
        expected_dimensions = service_config.expected_dimensions
        if expected_dimensions is None and indexing_config.get("embedding_dimensions") is not None:
            expected_dimensions = int(indexing_config["embedding_dimensions"])
        batch_size = int(indexing_config.get("embedding_batch_size", service_config.batch_size))
        return replace(
            service_config,
            expected_dimensions=expected_dimensions,
            batch_size=batch_size,
        )

    return service_config


def _default_provider(service: str) -> str:
    if service == "eval_question_generator":
        return "deepseek"
    if service == "reranker":
        return "noop"
    if service == "embedding":
        return "siliconflow"
    return "openai"


def _default_capability(service: str) -> str:
    if service == "embedding":
        return "embedding"
    if service == "reranker":
        return "rerank"
    return "chat"


def _default_adapter(provider: str) -> str:
    if provider in {"mock", "noop"}:
        return provider
    if provider == "siliconflow_rerank":
        return "http_rerank"
    return "openai_compatible"


def _default_api_key_env(provider: str) -> str | None:
    defaults = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "siliconflow": "SILICONFLOW_API_KEY",
    }
    return defaults.get(provider)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)

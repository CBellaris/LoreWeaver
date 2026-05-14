"""Extraction model settings helpers."""

from __future__ import annotations

from typing import Any

from loreweaver.config import AppConfig
from loreweaver.extraction.types import TokenPrice
from loreweaver.model_services import ModelServiceFactory


def _model_settings(models_config: AppConfig) -> dict[str, Any]:
    service_config = ModelServiceFactory.from_configs(models_config=models_config).resolve(
        "extraction"
    )
    return {
        "provider": service_config.provider.name,
        "model": service_config.model or "gpt-4o-mini",
        "temperature": 0 if service_config.temperature is None else service_config.temperature,
        "input_yuan_per_1k": service_config.pricing.input_yuan_per_1k,
        "output_yuan_per_1k": service_config.pricing.output_yuan_per_1k,
        "json_response_format": service_config.json_response_format,
        "batch_model": service_config.batch_model,
        "batch_input_yuan_per_1k": service_config.batch_pricing.input_yuan_per_1k,
        "batch_output_yuan_per_1k": service_config.batch_pricing.output_yuan_per_1k,
    }


def _token_price(model_settings: dict[str, Any], *, batch_mode: bool) -> TokenPrice:
    input_key = "batch_input_yuan_per_1k" if batch_mode else "input_yuan_per_1k"
    output_key = "batch_output_yuan_per_1k" if batch_mode else "output_yuan_per_1k"
    return TokenPrice(
        input_yuan_per_1k=float(model_settings.get(input_key, 0.0)),
        output_yuan_per_1k=float(model_settings.get(output_key, 0.0)),
    )

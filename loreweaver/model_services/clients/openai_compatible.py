"""OpenAI-compatible chat, embedding, and batch clients."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

from loreweaver.model_services.config import ModelServiceConfig
from loreweaver.model_services.errors import MissingApiKeyError, MissingDependencyError
from loreweaver.model_services.json_utils import (
    chat_content_from_response,
    parse_json_object,
    usage_from_response,
)
from loreweaver.model_services.types import (
    BatchStatus,
    BatchSubmission,
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    JsonChatResult,
)


class OpenAICompatibleClient:
    """Client for providers that expose OpenAI-compatible endpoints."""

    def __init__(self, service_config: ModelServiceConfig) -> None:
        if not service_config.api_key_env:
            raise MissingApiKeyError(
                f"Provider {service_config.provider.name} does not define api_key_env"
            )
        api_key = os.environ.get(service_config.api_key_env)
        if not api_key:
            raise MissingApiKeyError(
                f"Missing API key environment variable: {service_config.api_key_env}"
            )
        try:
            from openai import OpenAI
        except ImportError as error:
            raise MissingDependencyError(
                "The openai package is required for live model service calls."
            ) from error

        self.service_config = service_config
        self.provider = service_config.provider.name
        self.model = service_config.model
        self._client = OpenAI(api_key=api_key, base_url=service_config.base_url)

    def complete(self, request: ChatRequest) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages,
        }
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.service_config.temperature
        )
        if temperature is not None:
            payload["temperature"] = temperature
        max_output_tokens = request.max_output_tokens or self.service_config.max_output_tokens
        if max_output_tokens:
            payload["max_tokens"] = max_output_tokens
        if request.response_format == "json_object" or (
            request.response_format == "none" and self.service_config.json_response_format
        ):
            payload["response_format"] = {"type": "json_object"}
        payload.update(request.extra)

        response = self._client.chat.completions.create(**payload)
        return ChatResult(
            content=chat_content_from_response(response),
            usage=usage_from_response(response),
            provider=self.provider,
            model=self.model,
            raw_response=response,
        )

    def complete_json(self, request: ChatRequest) -> JsonChatResult:
        result = self.complete(request)
        return JsonChatResult(
            payload=parse_json_object(result.content),
            content=result.content,
            usage=result.usage,
            provider=result.provider,
            model=result.model,
            raw_response=result.raw_response,
        )

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], usage={}, provider=self.provider, model=self.model)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if (
            self.service_config.use_dimensions_param
            and self.service_config.expected_dimensions
        ):
            payload["dimensions"] = self.service_config.expected_dimensions

        response = self._client.embeddings.create(**payload)
        data = sorted(response.data, key=lambda item: int(getattr(item, "index", 0) or 0))
        return EmbeddingResult(
            vectors=[[float(value) for value in item.embedding] for item in data],
            usage=usage_from_response(response),
            provider=self.provider,
            model=self.model,
        )

    def submit_chat_batch(
        self,
        *,
        input_path: Path,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> BatchSubmission:
        with input_path.open("rb") as file_obj:
            uploaded = self._client.files.create(file=file_obj, purpose="batch")
        input_file_id = _uploaded_file_id(uploaded)
        batch = self._client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata or {},
        )
        return BatchSubmission(
            batch_id=str(batch.id),
            input_file_id=input_file_id,
            status=str(getattr(batch, "status", "")),
            output_file_id=_optional_str(getattr(batch, "output_file_id", None)),
            error_file_id=_optional_str(getattr(batch, "error_file_id", None)),
            request_counts=_request_counts_dict(getattr(batch, "request_counts", None)),
        )

    def retrieve_chat_batch(self, batch_id: str) -> BatchStatus:
        batch = self._client.batches.retrieve(batch_id)
        return BatchStatus(
            batch_id=str(batch.id),
            status=str(getattr(batch, "status", "")),
            input_file_id=_optional_str(getattr(batch, "input_file_id", None)),
            output_file_id=_optional_str(getattr(batch, "output_file_id", None)),
            error_file_id=_optional_str(getattr(batch, "error_file_id", None)),
            request_counts=_request_counts_dict(getattr(batch, "request_counts", None)),
        )

    def download_file_text(self, file_id: str) -> str:
        if file_id.startswith(("http://", "https://")):
            with urllib.request.urlopen(file_id, timeout=60) as response:
                return response.read().decode("utf-8")
        content = self._client.files.content(file_id)
        if hasattr(content, "text"):
            return str(content.text)
        if hasattr(content, "content"):
            raw_content = content.content
            if isinstance(raw_content, bytes):
                return raw_content.decode("utf-8")
            return str(raw_content)
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return str(content)


def _uploaded_file_id(uploaded: Any) -> str:
    if isinstance(uploaded, dict):
        return str(uploaded.get("id", ""))
    return str(getattr(uploaded, "id", ""))


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _request_counts_dict(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): int(count or 0) for key, count in value.items()}
    result: dict[str, int] = {}
    for key in ("total", "completed", "failed"):
        count = getattr(value, key, None)
        if count is not None:
            result[key] = int(count or 0)
    return result

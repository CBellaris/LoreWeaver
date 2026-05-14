"""Shared request and response types for model services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

JsonResponseFormat = Literal["none", "json_object"]


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, str]]
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_format: JsonResponseFormat = "none"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResult:
    content: str
    usage: dict[str, int]
    provider: str
    model: str
    raw_response: Any | None = None


@dataclass(frozen=True)
class JsonChatResult:
    payload: dict[str, Any]
    content: str
    usage: dict[str, int]
    provider: str
    model: str
    raw_response: Any | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    usage: dict[str, int]
    provider: str
    model: str


@dataclass(frozen=True)
class RerankScore:
    index: int
    score: float


@dataclass(frozen=True)
class RerankServiceResult:
    scores: list[RerankScore]
    usage: dict[str, int]
    provider: str
    model: str


@dataclass(frozen=True)
class BatchSubmission:
    batch_id: str
    input_file_id: str
    status: str
    output_file_id: str | None
    error_file_id: str | None
    request_counts: dict[str, int]


@dataclass(frozen=True)
class BatchStatus:
    batch_id: str
    status: str
    input_file_id: str | None
    output_file_id: str | None
    error_file_id: str | None
    request_counts: dict[str, int]

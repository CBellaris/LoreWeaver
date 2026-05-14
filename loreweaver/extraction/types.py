"""Shared extraction runtime types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from loreweaver.extraction.locator import LocatorResult
from loreweaver.extraction.schemas import SpanCandidatePayload
from loreweaver.model_services import ChatRequest, ChatResult
from loreweaver.models.span import Span


class ChatClient(Protocol):
    provider: str
    model: str

    def complete(self, request: ChatRequest) -> ChatResult:
        """Return raw JSON text and token usage."""


@dataclass(frozen=True)
class BatchOutput:
    custom_id: str
    raw_output: str | None
    usage: dict[str, int]
    error: str | None


@dataclass(frozen=True)
class BatchApplyOutcome:
    results: list["ExtractionResult"]
    retry_windows: list[CandidateWindow]
    retry_usage_by_window: dict[str, dict[str, int]]
    retry_reasons: dict[str, str]


@dataclass(frozen=True)
class TokenPrice:
    input_yuan_per_1k: float = 0.0
    output_yuan_per_1k: float = 0.0


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_yuan_per_1k: float
    output_yuan_per_1k: float
    estimated_yuan: float


@dataclass(frozen=True)
class ExtractionResult:
    span: Span
    payload: SpanCandidatePayload | None
    locator_result: LocatorResult | None
    status: str
    failure_reason: str | None
    raw_output: str | None
    attempts: int
    usage: dict[str, int]
    cost: CostEstimate


class WindowPayloadParseError(ValueError):
    """Raised when a full window payload cannot be parsed or schema-validated."""

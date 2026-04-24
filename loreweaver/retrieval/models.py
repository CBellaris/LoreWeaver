"""Shared retrieval data structures for M1.6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loreweaver.models.span import Span


@dataclass(frozen=True)
class RetrievalHit:
    span_id: str
    source: str
    score: float
    span: Span | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnionCandidate:
    span_id: str
    span: Span
    sources: list[str]
    source_scores: dict[str, float]
    normalized_scores: dict[str, float]
    fused_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankCandidate:
    candidate: UnionCandidate
    text: str


@dataclass(frozen=True)
class RerankResult:
    span_id: str
    score: float
    rank: int
    provider: str
    model: str
    text_sha256: str

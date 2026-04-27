"""Evidence pack models for M1.7."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class MergedEvidenceInterval:
    document_id: str
    chapter_id: str
    chapter_title: str
    start_idx: int
    end_idx: int
    source_span_ids: list[str]
    retrieval_sources: list[str]
    rerank_score: float
    priority_score: float


@dataclass(frozen=True)
class EvidenceBlock:
    citation_id: str
    document_id: str
    chapter_id: str
    chapter_title: str
    start_idx: int
    end_idx: int
    text: str
    source_span_ids: list[str]
    retrieval_sources: list[str]
    rerank_score: float


@dataclass(frozen=True)
class QueryEvidencePack:
    query_id: str
    document_id: str
    user_question: str
    query_type: str
    retrieved_span_ids: list[str]
    cluster_ids: list[str]
    merged_intervals: list[dict[str, Any]]
    evidence_blocks: list[dict[str, Any]]
    retrieval_sources: dict[str, Any]
    rerank_scores: dict[str, float]
    token_estimate: int
    answer: str | None
    created_at: datetime

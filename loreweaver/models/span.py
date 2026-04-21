"""Span data model placeholders for M1.3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Span:
    span_id: str
    document_id: str
    chapter_id: str
    window_id: str
    window_start: int
    window_end: int
    summary: str
    entities: list[str]
    topics: list[str]
    salience_score: float
    exact_text_quote: str
    quote_start_idx: int | None
    quote_end_idx: int | None
    locator_confidence: float
    locator_status: str
    created_at: datetime


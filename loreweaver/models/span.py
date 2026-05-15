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
    span_index_in_window: int
    window_start: int
    window_end: int
    span_type: str
    summary: str
    entities: list[str]
    salience_score: float
    start_anchor_quote: str
    end_anchor_quote: str
    key_quote: str
    span_start_idx: int | None
    span_end_idx: int | None
    located_text: str
    locator_confidence: float
    locator_status: str
    created_at: datetime

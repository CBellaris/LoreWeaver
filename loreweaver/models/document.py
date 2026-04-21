"""Document data model placeholders for M1.1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    author: str | None
    source_path: str
    normalized_path: str
    total_chars: int
    total_chapters: int
    content_hash: str
    created_at: datetime


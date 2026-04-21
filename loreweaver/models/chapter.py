"""Chapter data model placeholders for M1.1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    chapter_id: str
    document_id: str
    chapter_index: int
    chapter_title: str
    start_idx: int
    end_idx: int
    char_count: int


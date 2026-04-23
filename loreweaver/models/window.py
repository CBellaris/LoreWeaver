"""Candidate window data model for M1.2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateWindow:
    window_id: str
    document_id: str
    chapter_id: str
    window_index: int
    window_start: int
    window_end: int
    text: str
    uncovered_text: str = ""

    @property
    def char_count(self) -> int:
        return self.window_end - self.window_start

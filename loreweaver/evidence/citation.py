"""Citation id helpers for M1.7 and M1.8."""

from __future__ import annotations

import re

_CITATION_PATTERN = re.compile(r"^\[E[0-9]{3}\]$")


def make_citation_id(index: int) -> str:
    if index < 1:
        raise ValueError("Citation index must be 1-based.")
    return f"[E{index:03d}]"


def validate_citation_ids(citation_ids: list[str]) -> None:
    if len(citation_ids) != len(set(citation_ids)):
        raise ValueError("Citation ids must be unique.")
    invalid = [citation_id for citation_id in citation_ids if not _CITATION_PATTERN.match(citation_id)]
    if invalid:
        raise ValueError(f"Invalid citation ids: {invalid}")

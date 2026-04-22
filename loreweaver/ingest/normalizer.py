"""Text normalization utilities for M1.1."""

from __future__ import annotations

import re
from dataclasses import dataclass


SEPARATOR_LINE_RE = re.compile(r"^[ \t]*[-=*_]{6,}[ \t]*$", re.MULTILINE)
TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)
BLANK_LINES_RE = re.compile(r"\n{3,}")
AD_LINE_RE = re.compile(
    r"^[ \t]*(?:本书来自|更多精彩|请收藏|最新网址|手机用户请浏览|"
    r"www\.|http://|https://).*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class NormalizationReport:
    raw_chars: int
    normalized_chars: int
    chars_removed: int


def normalize_text(
    text: str,
    *,
    normalize_newlines: bool = True,
    remove_extra_blank_lines: bool = True,
) -> tuple[str, NormalizationReport]:
    """Create the canonical Layer 0 text used by all downstream coordinates."""
    raw_chars = len(text)
    normalized = text.replace("\ufeff", "")

    if normalize_newlines:
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    normalized = TRAILING_SPACE_RE.sub("", normalized)
    normalized = SEPARATOR_LINE_RE.sub("", normalized)
    normalized = AD_LINE_RE.sub("", normalized)

    if remove_extra_blank_lines:
        normalized = BLANK_LINES_RE.sub("\n\n", normalized)

    normalized = normalized.strip() + "\n" if normalized.strip() else ""
    report = NormalizationReport(
        raw_chars=raw_chars,
        normalized_chars=len(normalized),
        chars_removed=raw_chars - len(normalized),
    )
    return normalized, report

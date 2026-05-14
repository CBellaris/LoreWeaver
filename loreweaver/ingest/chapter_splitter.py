"""Chapter boundary detection for M1.1."""

from __future__ import annotations

import re
from dataclasses import dataclass

from loreweaver.models.chapter import Chapter


CN_NUMERAL = "一二三四五六七八九十百千万零〇两0-9"
REAL_CHAPTER_RE = re.compile(
    rf"^(?:第[{CN_NUMERAL}]+章|[{CN_NUMERAL}]+章|Chapter\s+[0-9]+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChapterSplitReport:
    total_chapters: int
    shortest_chapter_chars: int
    longest_chapter_chars: int
    boundary_warnings: list[str]
    strategy: str


def split_chapters(
    text: str,
    *,
    document_id: str,
    chapter_patterns: list[str],
    max_chapters: int | None = None,
) -> tuple[list[Chapter], ChapterSplitReport]:
    """Split normalized text into ordered, non-overlapping chapter intervals."""
    headings = _find_headings(text, chapter_patterns)
    strategy = "configured_patterns"

    real_headings = [heading for heading in headings if REAL_CHAPTER_RE.match(heading.title)]
    if real_headings:
        headings = real_headings
        strategy = "real_chapter_patterns"

    if not headings:
        chapters = [_whole_document_chapter(text, document_id=document_id)]
        strategy = "whole_document_fallback"
    else:
        chapters = _chapters_from_headings(text, document_id=document_id, headings=headings)

    if max_chapters is not None:
        if max_chapters < 1:
            raise ValueError("--max-chapters must be greater than zero")
        chapters = chapters[:max_chapters]

    warnings = _validate_chapters(text, chapters)
    lengths = [chapter.char_count for chapter in chapters]
    report = ChapterSplitReport(
        total_chapters=len(chapters),
        shortest_chapter_chars=min(lengths) if lengths else 0,
        longest_chapter_chars=max(lengths) if lengths else 0,
        boundary_warnings=warnings,
        strategy=strategy,
    )
    return chapters, report


@dataclass(frozen=True)
class _Heading:
    title: str
    start_idx: int


def _find_headings(text: str, chapter_patterns: list[str]) -> list[_Heading]:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in chapter_patterns]
    headings: list[_Heading] = []
    offset = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start = offset
        offset += len(line)

        if not stripped or len(stripped) > 80:
            continue

        if any(pattern.match(stripped) for pattern in compiled):
            title_start = line_start + line.index(line.lstrip())
            headings.append(_Heading(title=stripped, start_idx=title_start))

    return _dedupe_headings(headings)


def _dedupe_headings(headings: list[_Heading]) -> list[_Heading]:
    seen: set[int] = set()
    deduped: list[_Heading] = []
    for heading in headings:
        if heading.start_idx in seen:
            continue
        seen.add(heading.start_idx)
        deduped.append(heading)
    return sorted(deduped, key=lambda heading: heading.start_idx)


def _chapters_from_headings(
    text: str,
    *,
    document_id: str,
    headings: list[_Heading],
) -> list[Chapter]:
    chapters: list[Chapter] = []
    for index, heading in enumerate(headings):
        end_idx = headings[index + 1].start_idx if index + 1 < len(headings) else len(text)
        chapters.append(
            Chapter(
                chapter_id=f"{document_id}_ch{index + 1:04d}",
                document_id=document_id,
                chapter_index=index + 1,
                chapter_title=heading.title,
                start_idx=heading.start_idx,
                end_idx=end_idx,
                char_count=end_idx - heading.start_idx,
            )
        )
    return chapters


def _whole_document_chapter(
    text: str,
    *,
    document_id: str,
) -> Chapter:
    return Chapter(
        chapter_id=f"{document_id}_ch0000",
        document_id=document_id,
        chapter_index=0,
        chapter_title="Whole Document",
        start_idx=0,
        end_idx=len(text),
        char_count=len(text),
    )


def _validate_chapters(text: str, chapters: list[Chapter]) -> list[str]:
    warnings: list[str] = []
    previous_end = None
    for chapter in chapters:
        if (
            chapter.start_idx < 0
            or chapter.end_idx > len(text)
            or chapter.start_idx >= chapter.end_idx
        ):
            warnings.append(f"{chapter.chapter_id}: invalid interval")
        if previous_end is not None and chapter.start_idx < previous_end:
            warnings.append(f"{chapter.chapter_id}: overlaps previous chapter")
        previous_end = chapter.end_idx
        if chapter.char_count != chapter.end_idx - chapter.start_idx:
            warnings.append(f"{chapter.chapter_id}: char_count mismatch")
        if chapter.char_count < 200:
            warnings.append(
                f"{chapter.chapter_id}: very short chapter ({chapter.char_count} chars)"
            )

    if chapters and chapters[0].start_idx > 2000:
        warnings.append(f"uncovered prefix is large ({chapters[0].start_idx} chars)")
    return warnings

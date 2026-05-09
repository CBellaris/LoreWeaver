"""Candidate window splitting for M1.2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.models.chapter import Chapter
from loreweaver.models.window import CandidateWindow
from loreweaver.progress import ProgressReporter
from loreweaver.storage.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class WindowSplitReport:
    document_id: str
    split_mode: str
    total_windows: int
    total_chapters: int
    average_window_chars: float
    shortest_window_chars: int
    longest_window_chars: int
    short_window_count: int
    configured_window_size_chars: int
    configured_overlap_ratio: float
    effective_stride_chars: int
    effective_overlap_chars: int
    min_window_chars: int
    max_window_chars: int
    per_chapter_window_counts: dict[str, int]
    boundary_warnings: list[str]


def build_candidate_windows(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    run_id: str,
    document_id: str | None = None,
    window_size_chars: int | None = None,
    overlap_ratio: float | None = None,
    min_window_chars: int | None = None,
    max_window_chars: int | None = None,
    split_by_chapter: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Load chapters from SQLite, split normalized text, and persist windows."""
    if progress is not None:
        progress.emit("stage_start", stage="windows.load", label="Load document and chapters", current=0, total=3, unit="steps")
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()

    document = store.get_document(document_id)
    chapters = store.list_chapters(document.document_id)
    if not chapters:
        raise ValueError(f"No chapters found for document_id={document.document_id}")

    normalized_path = Path(document.normalized_path)
    normalized_text = normalized_path.read_text(encoding="utf-8")
    if len(normalized_text) != document.total_chars:
        raise ValueError(
            "Normalized text length does not match Document.total_chars: "
            f"{len(normalized_text)} != {document.total_chars}"
        )

    if progress is not None:
        progress.emit(
            "stage_start",
            stage="windows.split",
            label="Build candidate windows",
            current=1,
            total=3,
            unit="steps",
            detail={"chapter_count": len(chapters)},
        )
    window_config = config.values.get("window", {})
    size = (
        window_size_chars
        if window_size_chars is not None
        else int(window_config.get("size_chars", 1200))
    )
    overlap = (
        overlap_ratio
        if overlap_ratio is not None
        else float(window_config.get("overlap_ratio", 0.2))
    )
    min_chars = (
        min_window_chars
        if min_window_chars is not None
        else int(window_config.get("min_chars", 300))
    )
    max_chars = (
        max_window_chars
        if max_window_chars is not None
        else int(window_config.get("max_chars", 1600))
    )

    windows, split_report = split_candidate_windows(
        normalized_text,
        document_id=document.document_id,
        chapters=chapters,
        window_size_chars=size,
        overlap_ratio=overlap,
        min_window_chars=min_chars,
        max_window_chars=max_chars,
        split_by_chapter=split_by_chapter,
    )
    if progress is not None:
        progress.emit(
            "stage_start",
            stage="windows.sqlite",
            label="Persist candidate windows",
            current=2,
            total=3,
            unit="steps",
            detail={"window_count": len(windows)},
        )
    store.upsert_candidate_windows(document.document_id, windows)

    report = {
        "run_id": run_id,
        "document": {
            "document_id": document.document_id,
            "title": document.title,
            "normalized_path": document.normalized_path,
            "total_chars": document.total_chars,
            "total_chapters": document.total_chapters,
        },
        "window_split": asdict(split_report),
        "windows_preview": [
            {
                "window_id": window.window_id,
                "chapter_id": window.chapter_id,
                "window_index": window.window_index,
                "window_start": window.window_start,
                "window_end": window.window_end,
                "char_count": window.char_count,
            }
            for window in windows[:5]
        ],
        "sqlite_path": str(storage_config.sqlite_path),
    }

    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{run_id}_windows_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.insert_window_report(run_id, document.document_id, report)
    if progress is not None:
        progress.emit(
            "completed",
            stage="windows.completed",
            label="Window split completed",
            current=3,
            total=3,
            unit="steps",
            status="completed",
            detail={
                "document_id": document.document_id,
                "window_count": len(windows),
                "report_path": str(report_path),
            },
        )
    return report


def split_candidate_windows(
    text: str,
    *,
    document_id: str,
    chapters: list[Chapter],
    window_size_chars: int = 1200,
    overlap_ratio: float = 0.2,
    min_window_chars: int = 300,
    max_window_chars: int = 1600,
    split_by_chapter: bool = False,
) -> tuple[list[CandidateWindow], WindowSplitReport]:
    """Split each chapter into deterministic overlapping windows."""
    if not split_by_chapter:
        _validate_window_config(
            window_size_chars=window_size_chars,
            overlap_ratio=overlap_ratio,
            min_window_chars=min_window_chars,
            max_window_chars=max_window_chars,
        )

    stride = max(1, int(round(window_size_chars * (1 - overlap_ratio))))
    windows: list[CandidateWindow] = []
    per_chapter_counts: dict[str, int] = {}

    for chapter in chapters:
        if split_by_chapter:
            chapter_windows = [_chapter_as_window(text, document_id=document_id, chapter=chapter)]
        else:
            chapter_windows = _split_chapter_windows(
                text,
                document_id=document_id,
                chapter=chapter,
                window_size_chars=window_size_chars,
                stride_chars=stride,
                min_window_chars=min_window_chars,
            )
        per_chapter_counts[chapter.chapter_id] = len(chapter_windows)
        windows.extend(chapter_windows)

    warnings = _validate_windows(
        text,
        chapters,
        windows,
        max_window_chars=max_window_chars,
        enforce_max_window_chars=not split_by_chapter,
    )
    lengths = [window.char_count for window in windows]
    report = WindowSplitReport(
        document_id=document_id,
        split_mode="chapter" if split_by_chapter else "sliding_window",
        total_windows=len(windows),
        total_chapters=len(chapters),
        average_window_chars=round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        shortest_window_chars=min(lengths) if lengths else 0,
        longest_window_chars=max(lengths) if lengths else 0,
        short_window_count=sum(1 for length in lengths if length < min_window_chars),
        configured_window_size_chars=window_size_chars,
        configured_overlap_ratio=overlap_ratio,
        effective_stride_chars=stride,
        effective_overlap_chars=max(0, window_size_chars - stride),
        min_window_chars=min_window_chars,
        max_window_chars=max_window_chars,
        per_chapter_window_counts=per_chapter_counts,
        boundary_warnings=warnings,
    )
    return windows, report


def _chapter_as_window(
    text: str,
    *,
    document_id: str,
    chapter: Chapter,
) -> CandidateWindow:
    return CandidateWindow(
        window_id=f"{chapter.chapter_id}_win0001",
        document_id=document_id,
        chapter_id=chapter.chapter_id,
        window_index=1,
        window_start=chapter.start_idx,
        window_end=chapter.end_idx,
        text=text[chapter.start_idx : chapter.end_idx],
    )


def _split_chapter_windows(
    text: str,
    *,
    document_id: str,
    chapter: Chapter,
    window_size_chars: int,
    stride_chars: int,
    min_window_chars: int,
) -> list[CandidateWindow]:
    chapter_length = chapter.end_idx - chapter.start_idx
    if chapter_length <= 0:
        return []

    intervals: list[tuple[int, int]] = []
    start = chapter.start_idx
    while start < chapter.end_idx:
        end = min(start + window_size_chars, chapter.end_idx)
        intervals.append((start, end))
        if end == chapter.end_idx:
            break
        start += stride_chars

    if len(intervals) > 1:
        last_start, last_end = intervals[-1]
        if last_end - last_start < min_window_chars:
            previous_start, _ = intervals[-2]
            intervals[-2] = (previous_start, chapter.end_idx)
            intervals.pop()

    return [
        CandidateWindow(
            window_id=f"{chapter.chapter_id}_win{index:04d}",
            document_id=document_id,
            chapter_id=chapter.chapter_id,
            window_index=index,
            window_start=start_idx,
            window_end=end_idx,
            text=text[start_idx:end_idx],
        )
        for index, (start_idx, end_idx) in enumerate(intervals, start=1)
    ]


def _validate_window_config(
    *,
    window_size_chars: int,
    overlap_ratio: float,
    min_window_chars: int,
    max_window_chars: int,
) -> None:
    if window_size_chars < 1:
        raise ValueError("window_size_chars must be greater than zero")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in the range [0, 1)")
    if min_window_chars < 1:
        raise ValueError("min_window_chars must be greater than zero")
    if max_window_chars < min_window_chars:
        raise ValueError("max_window_chars must be greater than or equal to min_window_chars")
    if window_size_chars > max_window_chars:
        raise ValueError("window_size_chars must be less than or equal to max_window_chars")


def _validate_windows(
    text: str,
    chapters: list[Chapter],
    windows: list[CandidateWindow],
    *,
    max_window_chars: int,
    enforce_max_window_chars: bool = True,
) -> list[str]:
    warnings: list[str] = []
    chapters_by_id = {chapter.chapter_id: chapter for chapter in chapters}

    for window in windows:
        chapter = chapters_by_id.get(window.chapter_id)
        if chapter is None:
            warnings.append(f"{window.window_id}: missing parent chapter")
            continue
        if window.window_start < chapter.start_idx or window.window_end > chapter.end_idx:
            warnings.append(f"{window.window_id}: crosses chapter boundary")
        if window.window_start >= window.window_end:
            warnings.append(f"{window.window_id}: invalid interval")
        if window.window_end > len(text):
            warnings.append(f"{window.window_id}: exceeds normalized text length")
        if window.text != text[window.window_start : window.window_end]:
            warnings.append(f"{window.window_id}: text does not match normalized slice")
        if enforce_max_window_chars and window.char_count > max_window_chars:
            warnings.append(
                f"{window.window_id}: exceeds max_window_chars ({window.char_count})"
            )

    return warnings

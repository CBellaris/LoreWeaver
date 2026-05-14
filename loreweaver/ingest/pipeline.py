"""M1.1 ingestion pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.ingest.chapter_splitter import split_chapters
from loreweaver.ingest.normalizer import normalize_text
from loreweaver.ingest.reader import read_text_file
from loreweaver.models.document import Document
from loreweaver.progress import ProgressReporter
from loreweaver.storage.sqlite_store import SQLiteStore


def ingest_text(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    run_id: str,
    source_path: Path,
    title: str | None = None,
    author: str | None = None,
    max_chapters: int | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    if progress is not None:
        progress.emit("stage_start", stage="ingest.read", label="Read source text", current=0, total=4, unit="steps")
    raw_text, encoding = read_text_file(source_path)
    ingest_config = config.values.get("ingest", {})
    if progress is not None:
        progress.emit("stage_start", stage="ingest.normalize", label="Normalize source text", current=1, total=4, unit="steps")
    normalized_text, normalization_report = normalize_text(
        raw_text,
        normalize_newlines=bool(ingest_config.get("normalize_newlines", True)),
        remove_extra_blank_lines=bool(ingest_config.get("remove_extra_blank_lines", True)),
    )
    content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    document_id = f"doc_{content_hash[:12]}"

    normalized_dir = config.data_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = normalized_dir / f"{document_id}.txt"
    with normalized_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized_text)

    if progress is not None:
        progress.emit("stage_start", stage="ingest.chapters", label="Split chapters", current=2, total=4, unit="steps")
    chapters, split_report = split_chapters(
        normalized_text,
        document_id=document_id,
        chapter_patterns=list(ingest_config.get("chapter_patterns", [])),
        max_chapters=max_chapters,
    )

    sample_config = config.values.get("sample", {})
    document = Document(
        document_id=document_id,
        title=title or sample_config.get("title") or source_path.stem,
        author=author if author is not None else sample_config.get("author"),
        source_path=str(source_path),
        normalized_path=str(normalized_path),
        total_chars=len(normalized_text),
        total_chapters=len(chapters),
        content_hash=content_hash,
        created_at=datetime.now(timezone.utc),
    )

    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    if progress is not None:
        progress.emit("stage_start", stage="ingest.sqlite", label="Persist document metadata", current=3, total=4, unit="steps")
    store.upsert_document_with_chapters(document, chapters)

    report = {
        "run_id": run_id,
        "document": {
            "document_id": document.document_id,
            "title": document.title,
            "author": document.author,
            "source_path": document.source_path,
            "normalized_path": document.normalized_path,
            "total_chars": document.total_chars,
            "total_chapters": document.total_chapters,
            "content_hash": document.content_hash,
        },
        "encoding": encoding,
        "normalization": asdict(normalization_report),
        "chapter_split": asdict(split_report),
        "chapters_preview": [
            {
                "chapter_index": chapter.chapter_index,
                "chapter_title": chapter.chapter_title,
                "start_idx": chapter.start_idx,
                "end_idx": chapter.end_idx,
                "char_count": chapter.char_count,
            }
            for chapter in chapters[:5]
        ],
        "sqlite_path": str(storage_config.sqlite_path),
    }

    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{run_id}_ingest_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.insert_ingest_report(run_id, document.document_id, report)
    if progress is not None:
        progress.emit(
            "completed",
            stage="ingest.completed",
            label="Ingest completed",
            current=4,
            total=4,
            unit="steps",
            status="completed",
            detail={"document_id": document.document_id, "report_path": str(report_path)},
        )
    return report

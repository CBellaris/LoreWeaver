"""Chapter-level evaluation corpus builder for M1.9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.logging import new_run_id
from loreweaver.storage.sqlite_store import SQLiteStore


def build_chapter_corpus(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    document_id: str | None = None,
    chapter_start: int = 1,
    chapter_end: int = 100,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export persisted normalized chapters as the long-context eval input."""
    if chapter_start <= 0 or chapter_end < chapter_start:
        raise ValueError("chapter_start/chapter_end must define a positive inclusive range.")

    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    document = store.get_document(document_id)
    chapters = [
        chapter
        for chapter in store.list_chapters(document.document_id)
        if chapter_start <= chapter.chapter_index <= chapter_end
    ]
    if not chapters:
        raise ValueError(
            f"No chapters found for document {document.document_id} "
            f"in range {chapter_start}-{chapter_end}."
        )

    normalized_text = Path(document.normalized_path).read_text(encoding="utf-8")
    run_id = new_run_id("eval_corpus")
    payload = {
        "run_id": run_id,
        "document": {
            "document_id": document.document_id,
            "title": document.title,
            "author": document.author,
            "content_hash": document.content_hash,
            "normalized_path": document.normalized_path,
        },
        "chapter_start": chapter_start,
        "chapter_end": chapter_end,
        "chapter_count": len(chapters),
        "char_count": sum(chapter.char_count for chapter in chapters),
        "chapters": [
            {
                "chapter_id": chapter.chapter_id,
                "chapter_index": chapter.chapter_index,
                "chapter_title": chapter.chapter_title,
                "start_idx": chapter.start_idx,
                "end_idx": chapter.end_idx,
                "char_count": chapter.char_count,
                "text": normalized_text[chapter.start_idx : chapter.end_idx],
            }
            for chapter in chapters
        ],
    }

    if output_path is None:
        output_path = (
            config.data_dir
            / "eval"
            / "corpora"
            / f"{document.document_id}_ch{chapter_start:03d}_{chapter_end:03d}.json"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload["corpus_path"] = str(destination)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_corpus(path: str | Path) -> dict[str, Any]:
    corpus_path = Path(path)
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if "chapters" not in payload or "document" not in payload:
        raise ValueError(f"Invalid eval corpus: {corpus_path}")
    return payload

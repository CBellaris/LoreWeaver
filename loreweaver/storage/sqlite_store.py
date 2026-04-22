"""SQLite metadata store for M1.1 onward."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from loreweaver.models.chapter import Chapter
from loreweaver.models.document import Document
from loreweaver.models.window import CandidateWindow


class SQLiteStore:
    """Small SQLite metadata store for deterministic M1 ingestion outputs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    author TEXT,
                    source_path TEXT NOT NULL,
                    normalized_path TEXT NOT NULL,
                    total_chars INTEGER NOT NULL,
                    total_chapters INTEGER NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chapter_index INTEGER NOT NULL,
                    chapter_title TEXT NOT NULL,
                    start_idx INTEGER NOT NULL,
                    end_idx INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_chapters_document_index
                ON chapters(document_id, chapter_index);

                CREATE TABLE IF NOT EXISTS ingest_reports (
                    run_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE TABLE IF NOT EXISTS candidate_windows (
                    window_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    window_index INTEGER NOT NULL,
                    window_start INTEGER NOT NULL,
                    window_end INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id),
                    FOREIGN KEY(chapter_id) REFERENCES chapters(chapter_id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_windows_chapter_index
                ON candidate_windows(chapter_id, window_index);

                CREATE INDEX IF NOT EXISTS idx_candidate_windows_document
                ON candidate_windows(document_id);

                CREATE TABLE IF NOT EXISTS window_reports (
                    run_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                """
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def upsert_document_with_chapters(
        self,
        document: Document,
        chapters: Iterable[Chapter],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, title, author, source_path, normalized_path,
                    total_chars, total_chapters, content_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title=excluded.title,
                    author=excluded.author,
                    source_path=excluded.source_path,
                    normalized_path=excluded.normalized_path,
                    total_chars=excluded.total_chars,
                    total_chapters=excluded.total_chapters,
                    content_hash=excluded.content_hash
                """,
                (
                    document.document_id,
                    document.title,
                    document.author,
                    document.source_path,
                    document.normalized_path,
                    document.total_chars,
                    document.total_chapters,
                    document.content_hash,
                    document.created_at.isoformat(),
                ),
            )
            connection.execute(
                "DELETE FROM candidate_windows WHERE document_id = ?",
                (document.document_id,),
            )
            connection.execute(
                "DELETE FROM chapters WHERE document_id = ?",
                (document.document_id,),
            )
            connection.executemany(
                """
                INSERT INTO chapters (
                    chapter_id, document_id, chapter_index, chapter_title,
                    start_idx, end_idx, char_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chapter.chapter_id,
                        chapter.document_id,
                        chapter.chapter_index,
                        chapter.chapter_title,
                        chapter.start_idx,
                        chapter.end_idx,
                        chapter.char_count,
                    )
                    for chapter in chapters
                ],
            )

    def insert_ingest_report(self, run_id: str, document_id: str, report: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO ingest_reports (
                    run_id, document_id, report_json, created_at
                )
                VALUES (?, ?, ?, datetime('now'))
                """,
                (run_id, document_id, json.dumps(report, ensure_ascii=False, indent=2)),
            )

    def get_document(self, document_id: str | None = None) -> Document:
        with self.connect() as connection:
            if document_id is None:
                row = connection.execute(
                    """
                    SELECT * FROM documents
                    ORDER BY created_at DESC, document_id DESC
                    LIMIT 1
                    """
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM documents WHERE document_id = ?",
                    (document_id,),
                ).fetchone()

        if row is None:
            requested = document_id or "<latest>"
            raise ValueError(f"Document not found: {requested}")
        return _document_from_row(row)

    def list_chapters(self, document_id: str) -> list[Chapter]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chapters
                WHERE document_id = ?
                ORDER BY chapter_index
                """,
                (document_id,),
            ).fetchall()
        return [_chapter_from_row(row) for row in rows]

    def upsert_candidate_windows(
        self,
        document_id: str,
        windows: Iterable[CandidateWindow],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM candidate_windows WHERE document_id = ?",
                (document_id,),
            )
            connection.executemany(
                """
                INSERT INTO candidate_windows (
                    window_id, document_id, chapter_id, window_index,
                    window_start, window_end, text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        window.window_id,
                        window.document_id,
                        window.chapter_id,
                        window.window_index,
                        window.window_start,
                        window.window_end,
                        window.text,
                    )
                    for window in windows
                ],
            )

    def list_candidate_windows(self, document_id: str) -> list[CandidateWindow]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT w.*
                FROM candidate_windows w
                JOIN chapters c ON w.chapter_id = c.chapter_id
                WHERE w.document_id = ?
                ORDER BY c.chapter_index, w.window_index
                """,
                (document_id,),
            ).fetchall()
        return [_candidate_window_from_row(row) for row in rows]

    def insert_window_report(self, run_id: str, document_id: str, report: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO window_reports (
                    run_id, document_id, report_json, created_at
                )
                VALUES (?, ?, ?, datetime('now'))
                """,
                (run_id, document_id, json.dumps(report, ensure_ascii=False, indent=2)),
            )


def _document_from_row(row: sqlite3.Row) -> Document:
    return Document(
        document_id=row["document_id"],
        title=row["title"],
        author=row["author"],
        source_path=row["source_path"],
        normalized_path=row["normalized_path"],
        total_chars=row["total_chars"],
        total_chapters=row["total_chapters"],
        content_hash=row["content_hash"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _chapter_from_row(row: sqlite3.Row) -> Chapter:
    return Chapter(
        chapter_id=row["chapter_id"],
        document_id=row["document_id"],
        chapter_index=row["chapter_index"],
        chapter_title=row["chapter_title"],
        start_idx=row["start_idx"],
        end_idx=row["end_idx"],
        char_count=row["char_count"],
    )


def _candidate_window_from_row(row: sqlite3.Row) -> CandidateWindow:
    return CandidateWindow(
        window_id=row["window_id"],
        document_id=row["document_id"],
        chapter_id=row["chapter_id"],
        window_index=row["window_index"],
        window_start=row["window_start"],
        window_end=row["window_end"],
        text=row["text"],
    )

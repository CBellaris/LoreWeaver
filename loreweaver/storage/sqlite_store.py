"""SQLite metadata store for M1.1 onward."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from loreweaver.models.chapter import Chapter
from loreweaver.models.document import Document
from loreweaver.models.span import Span
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
                    uncovered_text TEXT NOT NULL DEFAULT '',
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
            _ensure_column(
                connection,
                table_name="candidate_windows",
                column_name="uncovered_text",
                definition="TEXT NOT NULL DEFAULT ''",
            )

    def initialize_extraction_tables(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS locator_candidates;
                DROP TABLE IF EXISTS extraction_failures;
                DROP TABLE IF EXISTS extraction_reports;
                DROP TABLE IF EXISTS spans;

                CREATE TABLE spans (
                    span_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    window_id TEXT NOT NULL,
                    span_index_in_window INTEGER NOT NULL,
                    window_start INTEGER NOT NULL,
                    window_end INTEGER NOT NULL,
                    micro_topic TEXT NOT NULL,
                    span_type TEXT NOT NULL,
                    micro_summary TEXT NOT NULL,
                    entities_json TEXT NOT NULL,
                    topics_json TEXT NOT NULL,
                    salience_score REAL NOT NULL,
                    start_anchor_quote TEXT NOT NULL,
                    end_anchor_quote TEXT NOT NULL,
                    key_quote TEXT NOT NULL,
                    overlap_reason TEXT NOT NULL,
                    span_start_idx INTEGER,
                    span_end_idx INTEGER,
                    located_text TEXT NOT NULL,
                    locator_confidence REAL NOT NULL,
                    locator_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id),
                    FOREIGN KEY(chapter_id) REFERENCES chapters(chapter_id),
                    FOREIGN KEY(window_id) REFERENCES candidate_windows(window_id)
                );

                CREATE INDEX idx_spans_document_status
                ON spans(document_id, locator_status);

                CREATE INDEX idx_spans_window
                ON spans(window_id);

                CREATE TABLE locator_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    span_id TEXT NOT NULL,
                    start_idx INTEGER NOT NULL,
                    end_idx INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    matched_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(span_id) REFERENCES spans(span_id)
                );

                CREATE INDEX idx_locator_candidates_span
                ON locator_candidates(span_id);

                CREATE TABLE extraction_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    window_id TEXT NOT NULL,
                    span_id TEXT,
                    stage TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    raw_output TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(window_id) REFERENCES candidate_windows(window_id),
                    FOREIGN KEY(span_id) REFERENCES spans(span_id)
                );

                CREATE INDEX idx_extraction_failures_window
                ON extraction_failures(window_id);

                CREATE TABLE extraction_reports (
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
            self._delete_extraction_outputs_for_document(connection, document.document_id)
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
            self._delete_extraction_outputs_for_document(connection, document_id)
            connection.execute(
                "DELETE FROM candidate_windows WHERE document_id = ?",
                (document_id,),
            )
            connection.executemany(
                """
                INSERT INTO candidate_windows (
                    window_id, document_id, chapter_id, window_index,
                    window_start, window_end, text, uncovered_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                        window.uncovered_text,
                    )
                    for window in windows
                ],
            )

    def _delete_extraction_outputs_for_document(
        self,
        connection: sqlite3.Connection,
        document_id: str,
    ) -> None:
        if not _table_exists(connection, "spans"):
            return
        span_ids = [
            row["span_id"]
            for row in connection.execute(
                "SELECT span_id FROM spans WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        ]
        if not span_ids:
            return
        placeholders = ",".join("?" for _ in span_ids)
        if _table_exists(connection, "locator_candidates"):
            connection.execute(
                f"DELETE FROM locator_candidates WHERE span_id IN ({placeholders})",
                span_ids,
            )
        if _table_exists(connection, "extraction_failures"):
            connection.execute(
                f"DELETE FROM extraction_failures WHERE span_id IN ({placeholders})",
                span_ids,
            )
        connection.execute(
            f"DELETE FROM spans WHERE span_id IN ({placeholders})",
            span_ids,
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

    def delete_spans_for_windows(self, window_ids: Iterable[str]) -> None:
        ids = list(window_ids)
        if not ids:
            return
        with self.connect() as connection:
            if not _table_exists(connection, "spans"):
                return
            placeholders = ",".join("?" for _ in ids)
            span_ids = [
                row["span_id"]
                for row in connection.execute(
                    f"SELECT span_id FROM spans WHERE window_id IN ({placeholders})",
                    ids,
                ).fetchall()
            ]
            if not span_ids:
                connection.execute(
                    f"UPDATE candidate_windows SET uncovered_text = '' "
                    f"WHERE window_id IN ({placeholders})",
                    ids,
                )
                return
            span_placeholders = ",".join("?" for _ in span_ids)
            if _table_exists(connection, "locator_candidates"):
                connection.execute(
                    f"DELETE FROM locator_candidates WHERE span_id IN ({span_placeholders})",
                    span_ids,
                )
            if _table_exists(connection, "extraction_failures"):
                connection.execute(
                    f"DELETE FROM extraction_failures WHERE span_id IN ({span_placeholders})",
                    span_ids,
                )
            connection.execute(
                f"DELETE FROM spans WHERE span_id IN ({span_placeholders})",
                span_ids,
            )
            connection.execute(
                f"UPDATE candidate_windows SET uncovered_text = '' "
                f"WHERE window_id IN ({placeholders})",
                ids,
            )

    def update_window_uncovered_text(self, window_id: str, uncovered_text: str) -> None:
        with self.connect() as connection:
            _ensure_column(
                connection,
                table_name="candidate_windows",
                column_name="uncovered_text",
                definition="TEXT NOT NULL DEFAULT ''",
            )
            connection.execute(
                """
                UPDATE candidate_windows
                SET uncovered_text = ?
                WHERE window_id = ?
                """,
                (uncovered_text, window_id),
            )

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

    def upsert_span(self, span: Span) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO spans (
                    span_id, document_id, chapter_id, window_id, span_index_in_window,
                    window_start, window_end, micro_topic, span_type, micro_summary,
                    entities_json, topics_json, salience_score, start_anchor_quote,
                    end_anchor_quote, key_quote, overlap_reason, span_start_idx,
                    span_end_idx, located_text, locator_confidence, locator_status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(span_id) DO UPDATE SET
                    span_index_in_window=excluded.span_index_in_window,
                    micro_topic=excluded.micro_topic,
                    span_type=excluded.span_type,
                    micro_summary=excluded.micro_summary,
                    entities_json=excluded.entities_json,
                    topics_json=excluded.topics_json,
                    salience_score=excluded.salience_score,
                    start_anchor_quote=excluded.start_anchor_quote,
                    end_anchor_quote=excluded.end_anchor_quote,
                    key_quote=excluded.key_quote,
                    overlap_reason=excluded.overlap_reason,
                    span_start_idx=excluded.span_start_idx,
                    span_end_idx=excluded.span_end_idx,
                    located_text=excluded.located_text,
                    locator_confidence=excluded.locator_confidence,
                    locator_status=excluded.locator_status,
                    created_at=excluded.created_at
                """,
                (
                    span.span_id,
                    span.document_id,
                    span.chapter_id,
                    span.window_id,
                    span.span_index_in_window,
                    span.window_start,
                    span.window_end,
                    span.micro_topic,
                    span.span_type,
                    span.micro_summary,
                    json.dumps(span.entities, ensure_ascii=False),
                    json.dumps(span.topics, ensure_ascii=False),
                    span.salience_score,
                    span.start_anchor_quote,
                    span.end_anchor_quote,
                    span.key_quote,
                    span.overlap_reason,
                    span.span_start_idx,
                    span.span_end_idx,
                    span.located_text,
                    span.locator_confidence,
                    span.locator_status,
                    span.created_at.isoformat(),
                ),
            )

    def list_spans(self, document_id: str, *, located_only: bool = False) -> list[Span]:
        query = "SELECT * FROM spans WHERE document_id = ?"
        params: tuple[object, ...] = (document_id,)
        if located_only:
            query += " AND locator_status = ?"
            params = (document_id, "located")
        query += " ORDER BY window_start, span_id"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_span_from_row(row) for row in rows]

    def insert_locator_candidates(self, span_id: str, candidates: Iterable[object]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM locator_candidates WHERE span_id = ?", (span_id,))
            connection.executemany(
                """
                INSERT INTO locator_candidates (
                    span_id, start_idx, end_idx, confidence, strategy, matched_text, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                [
                    (
                        span_id,
                        candidate.start_idx,
                        candidate.end_idx,
                        candidate.confidence,
                        candidate.strategy,
                        candidate.matched_text,
                    )
                    for candidate in candidates
                ],
            )

    def insert_extraction_failure(
        self,
        *,
        window_id: str,
        span_id: str | None,
        stage: str,
        reason: str,
        attempts: int,
        raw_output: str | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO extraction_failures (
                    window_id, span_id, stage, reason, attempts, raw_output, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (window_id, span_id, stage, reason, attempts, raw_output),
            )

    def insert_extraction_report(self, run_id: str, document_id: str, report: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO extraction_reports (
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
        uncovered_text=row["uncovered_text"] if "uncovered_text" in row.keys() else "",
    )


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_column(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    if any(row["name"] == column_name for row in rows):
        return
    connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def _span_from_row(row: sqlite3.Row) -> Span:
    return Span(
        span_id=row["span_id"],
        document_id=row["document_id"],
        chapter_id=row["chapter_id"],
        window_id=row["window_id"],
        span_index_in_window=row["span_index_in_window"],
        window_start=row["window_start"],
        window_end=row["window_end"],
        micro_topic=row["micro_topic"],
        span_type=row["span_type"],
        micro_summary=row["micro_summary"],
        entities=json.loads(row["entities_json"]),
        topics=json.loads(row["topics_json"]),
        salience_score=row["salience_score"],
        start_anchor_quote=row["start_anchor_quote"],
        end_anchor_quote=row["end_anchor_quote"],
        key_quote=row["key_quote"],
        overlap_reason=row["overlap_reason"],
        span_start_idx=row["span_start_idx"],
        span_end_idx=row["span_end_idx"],
        located_text=row["located_text"],
        locator_confidence=row["locator_confidence"],
        locator_status=row["locator_status"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )

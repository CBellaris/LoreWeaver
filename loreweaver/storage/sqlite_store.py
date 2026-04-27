"""SQLite metadata store for M1.1 onward."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from loreweaver.models.chapter import Chapter
from loreweaver.models.cluster import CenterSpanCluster, SpanEdge
from loreweaver.models.document import Document
from loreweaver.models.evidence import QueryEvidencePack
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
                CREATE TABLE IF NOT EXISTS spans (
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

                CREATE INDEX IF NOT EXISTS idx_spans_document_status
                ON spans(document_id, locator_status);

                CREATE INDEX IF NOT EXISTS idx_spans_window
                ON spans(window_id);

                CREATE TABLE IF NOT EXISTS locator_candidates (
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

                CREATE INDEX IF NOT EXISTS idx_locator_candidates_span
                ON locator_candidates(span_id);

                CREATE TABLE IF NOT EXISTS extraction_failures (
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

                CREATE INDEX IF NOT EXISTS idx_extraction_failures_window
                ON extraction_failures(window_id);

                CREATE TABLE IF NOT EXISTS extraction_reports (
                    run_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                """
            )

    def initialize_index_tables(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_input
                ON embedding_cache(provider, model, input_sha256);

                CREATE TABLE IF NOT EXISTS index_reports (
                    run_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE TABLE IF NOT EXISTS query_runs (
                    query_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    user_question TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                """
            )

    def initialize_graph_tables(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS center_span_clusters (
                    cluster_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    center_span_id TEXT NOT NULL,
                    cluster_name TEXT NOT NULL,
                    cluster_type TEXT NOT NULL,
                    micro_summary TEXT NOT NULL,
                    member_span_ids_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id),
                    FOREIGN KEY(center_span_id) REFERENCES spans(span_id)
                );

                CREATE INDEX IF NOT EXISTS idx_center_span_clusters_document
                ON center_span_clusters(document_id, cluster_type);

                CREATE TABLE IF NOT EXISTS span_edges (
                    edge_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    from_type TEXT NOT NULL,
                    to_type TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_span_edges_document_type
                ON span_edges(document_id, edge_type);

                CREATE INDEX IF NOT EXISTS idx_span_edges_from
                ON span_edges(from_id);

                CREATE INDEX IF NOT EXISTS idx_span_edges_to
                ON span_edges(to_id);

                CREATE TABLE IF NOT EXISTS graph_reports (
                    run_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                """
            )

    def initialize_evidence_tables(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_packs (
                    query_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    user_question TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    pack_json TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    answer TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_packs_document
                ON evidence_packs(document_id, created_at);
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
            self._delete_graph_outputs_for_document(connection, document.document_id)
            self._delete_evidence_outputs_for_document(connection, document.document_id)
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
            self._delete_graph_outputs_for_document(connection, document_id)
            self._delete_evidence_outputs_for_document(connection, document_id)
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

    def _delete_graph_outputs_for_document(
        self,
        connection: sqlite3.Connection,
        document_id: str,
    ) -> None:
        if _table_exists(connection, "span_edges"):
            connection.execute(
                "DELETE FROM span_edges WHERE document_id = ?",
                (document_id,),
            )
        if _table_exists(connection, "center_span_clusters"):
            connection.execute(
                "DELETE FROM center_span_clusters WHERE document_id = ?",
                (document_id,),
            )

    def _delete_evidence_outputs_for_document(
        self,
        connection: sqlite3.Connection,
        document_id: str,
    ) -> None:
        if _table_exists(connection, "evidence_packs"):
            connection.execute(
                "DELETE FROM evidence_packs WHERE document_id = ?",
                (document_id,),
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
            placeholders = ",".join("?" for _ in ids)
            document_ids = [
                row["document_id"]
                for row in connection.execute(
                    f"SELECT DISTINCT document_id FROM candidate_windows "
                    f"WHERE window_id IN ({placeholders})",
                    ids,
                ).fetchall()
            ]
            for document_id in document_ids:
                self._delete_graph_outputs_for_document(connection, document_id)
                self._delete_evidence_outputs_for_document(connection, document_id)
            if _table_exists(connection, "extraction_failures"):
                connection.execute(
                    f"DELETE FROM extraction_failures WHERE window_id IN ({placeholders})",
                    ids,
                )
            if not _table_exists(connection, "spans"):
                return
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

    def list_window_extraction_status(self, document_id: str) -> list[dict]:
        with self.connect() as connection:
            has_spans = _table_exists(connection, "spans")
            has_failures = _table_exists(connection, "extraction_failures")
            query = """
                SELECT
                    w.window_id,
                    w.document_id,
                    w.chapter_id,
                    c.chapter_index,
                    w.window_index,
                    w.window_start,
                    w.window_end,
                    length(w.text) AS char_count,
                    COALESCE(s.span_count, 0) AS span_count,
                    COALESCE(s.located_count, 0) AS located_count,
                    COALESCE(s.failed_span_count, 0) AS failed_span_count
            """
            if has_failures:
                query += ", COALESCE(f.failure_count, 0) AS failure_count"
            else:
                query += ", 0 AS failure_count"
            query += """
                FROM candidate_windows w
                JOIN chapters c ON w.chapter_id = c.chapter_id
            """
            if has_spans:
                query += """
                    LEFT JOIN (
                        SELECT
                            window_id,
                            COUNT(*) AS span_count,
                            SUM(CASE WHEN locator_status = 'located' THEN 1 ELSE 0 END)
                                AS located_count,
                            SUM(CASE WHEN locator_status != 'located' THEN 1 ELSE 0 END)
                                AS failed_span_count
                        FROM spans
                        GROUP BY window_id
                    ) s ON s.window_id = w.window_id
                """
            else:
                query += """
                    LEFT JOIN (
                        SELECT NULL AS window_id, 0 AS span_count, 0 AS located_count,
                            0 AS failed_span_count
                    ) s ON 0
                """
            if has_failures:
                query += """
                    LEFT JOIN (
                        SELECT window_id, COUNT(*) AS failure_count
                        FROM extraction_failures
                        GROUP BY window_id
                    ) f ON f.window_id = w.window_id
                """
            query += """
                WHERE w.document_id = ?
                ORDER BY c.chapter_index, w.window_index
            """
            rows = connection.execute(query, (document_id,)).fetchall()
        statuses = []
        for index, row in enumerate(rows, start=1):
            span_count = int(row["span_count"] or 0)
            located_count = int(row["located_count"] or 0)
            failed_span_count = int(row["failed_span_count"] or 0)
            failure_count = int(row["failure_count"] or 0)
            extracted = span_count > 0 or failure_count > 0
            statuses.append(
                {
                    "global_index": index,
                    "window_id": row["window_id"],
                    "document_id": row["document_id"],
                    "chapter_id": row["chapter_id"],
                    "chapter_index": row["chapter_index"],
                    "window_index": row["window_index"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "char_count": row["char_count"],
                    "span_count": span_count,
                    "located_count": located_count,
                    "failed_count": failed_span_count + failure_count,
                    "status": "extracted" if extracted else "pending",
                }
            )
        return statuses

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

    def list_top_salience_spans(self, document_id: str, *, limit: int = 30) -> list[Span]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM spans
                WHERE document_id = ? AND locator_status = ?
                ORDER BY salience_score DESC, span_start_idx ASC, span_id ASC
                LIMIT ?
                """,
                (document_id, "located", limit),
            ).fetchall()
        return [_span_from_row(row) for row in rows]

    def get_span(self, span_id: str) -> Span:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM spans WHERE span_id = ?",
                (span_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Span not found: {span_id}")
        return _span_from_row(row)

    def list_spans_by_ids(self, span_ids: Iterable[str]) -> list[Span]:
        ids = list(dict.fromkeys(span_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM spans WHERE span_id IN ({placeholders})",
                ids,
            ).fetchall()
        parsed_spans = [_span_from_row(row) for row in rows]
        spans_by_id = {span.span_id: span for span in parsed_spans}
        return [spans_by_id[span_id] for span_id in ids if span_id in spans_by_id]

    def get_embedding_cache(self, cache_key: str) -> list[float] | None:
        with self.connect() as connection:
            if not _table_exists(connection, "embedding_cache"):
                return None
            row = connection.execute(
                "SELECT vector_json FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return [float(value) for value in json.loads(row["vector_json"])]

    def upsert_embedding_cache(
        self,
        *,
        cache_key: str,
        provider: str,
        model: str,
        input_sha256: str,
        input_text: str,
        vector: list[float],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO embedding_cache (
                    cache_key, provider, model, dimensions, input_sha256,
                    input_text, vector_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(cache_key) DO UPDATE SET
                    provider=excluded.provider,
                    model=excluded.model,
                    dimensions=excluded.dimensions,
                    input_sha256=excluded.input_sha256,
                    input_text=excluded.input_text,
                    vector_json=excluded.vector_json,
                    created_at=excluded.created_at
                """,
                (
                    cache_key,
                    provider,
                    model,
                    len(vector),
                    input_sha256,
                    input_text,
                    json.dumps(vector),
                ),
            )

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

    def insert_index_report(self, run_id: str, document_id: str, report: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO index_reports (
                    run_id, document_id, report_json, created_at
                )
                VALUES (?, ?, ?, datetime('now'))
                """,
                (run_id, document_id, json.dumps(report, ensure_ascii=False, indent=2)),
            )

    def insert_query_run(
        self,
        query_id: str,
        document_id: str,
        user_question: str,
        report: dict,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO query_runs (
                    query_id, document_id, user_question, report_json, created_at
                )
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (
                    query_id,
                    document_id,
                    user_question,
                    json.dumps(report, ensure_ascii=False, indent=2),
                ),
            )

    def replace_graph(
        self,
        *,
        document_id: str,
        clusters: Iterable[CenterSpanCluster],
        edges: Iterable[SpanEdge],
    ) -> None:
        cluster_list = list(clusters)
        edge_list = list(edges)
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM span_edges WHERE document_id = ?",
                (document_id,),
            )
            connection.execute(
                "DELETE FROM center_span_clusters WHERE document_id = ?",
                (document_id,),
            )
            connection.executemany(
                """
                INSERT INTO center_span_clusters (
                    cluster_id, document_id, center_span_id, cluster_name, cluster_type,
                    micro_summary, member_span_ids_json, confidence, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        cluster.cluster_id,
                        cluster.document_id,
                        cluster.center_span_id,
                        cluster.cluster_name,
                        cluster.cluster_type,
                        cluster.micro_summary,
                        json.dumps(cluster.member_span_ids, ensure_ascii=False),
                        cluster.confidence,
                        cluster.status,
                        cluster.created_at.isoformat(),
                    )
                    for cluster in cluster_list
                ],
            )
            connection.executemany(
                """
                INSERT INTO span_edges (
                    edge_id, document_id, from_id, to_id, from_type, to_type,
                    edge_type, weight, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        edge.edge_id,
                        edge.document_id,
                        edge.from_id,
                        edge.to_id,
                        edge.from_type,
                        edge.to_type,
                        edge.edge_type,
                        edge.weight,
                        edge.source,
                        edge.created_at.isoformat(),
                    )
                    for edge in edge_list
                ],
            )

    def list_center_span_clusters(
        self,
        document_id: str,
        *,
        cluster_id: str | None = None,
    ) -> list[CenterSpanCluster]:
        query = "SELECT * FROM center_span_clusters WHERE document_id = ?"
        params: list[object] = [document_id]
        if cluster_id is not None:
            query += " AND cluster_id = ?"
            params.append(cluster_id)
        query += " ORDER BY cluster_type, cluster_name, cluster_id"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_cluster_from_row(row) for row in rows]

    def list_span_edges(
        self,
        document_id: str,
        *,
        edge_type: str | None = None,
        from_id: str | None = None,
    ) -> list[SpanEdge]:
        query = "SELECT * FROM span_edges WHERE document_id = ?"
        params: list[object] = [document_id]
        if edge_type is not None:
            query += " AND edge_type = ?"
            params.append(edge_type)
        if from_id is not None:
            query += " AND from_id = ?"
            params.append(from_id)
        query += " ORDER BY edge_type, from_id, to_id"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_edge_from_row(row) for row in rows]

    def insert_graph_report(self, run_id: str, document_id: str, report: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO graph_reports (
                    run_id, document_id, report_json, created_at
                )
                VALUES (?, ?, ?, datetime('now'))
                """,
                (run_id, document_id, json.dumps(report, ensure_ascii=False, indent=2)),
            )

    def insert_evidence_pack(self, pack: QueryEvidencePack, *, report: dict) -> None:
        pack_payload = {
            "query_id": pack.query_id,
            "document_id": pack.document_id,
            "user_question": pack.user_question,
            "query_type": pack.query_type,
            "retrieved_span_ids": pack.retrieved_span_ids,
            "cluster_ids": pack.cluster_ids,
            "merged_intervals": pack.merged_intervals,
            "evidence_blocks": pack.evidence_blocks,
            "retrieval_sources": pack.retrieval_sources,
            "rerank_scores": pack.rerank_scores,
            "token_estimate": pack.token_estimate,
            "answer": pack.answer,
            "created_at": pack.created_at.isoformat(),
        }
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO evidence_packs (
                    query_id, document_id, user_question, query_type,
                    pack_json, report_json, answer, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack.query_id,
                    pack.document_id,
                    pack.user_question,
                    pack.query_type,
                    json.dumps(pack_payload, ensure_ascii=False, indent=2),
                    json.dumps(report, ensure_ascii=False, indent=2),
                    pack.answer,
                    pack.created_at.isoformat(),
                ),
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


def _cluster_from_row(row: sqlite3.Row) -> CenterSpanCluster:
    return CenterSpanCluster(
        cluster_id=row["cluster_id"],
        document_id=row["document_id"],
        center_span_id=row["center_span_id"],
        cluster_name=row["cluster_name"],
        cluster_type=row["cluster_type"],
        micro_summary=row["micro_summary"],
        member_span_ids=json.loads(row["member_span_ids_json"]),
        confidence=row["confidence"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _edge_from_row(row: sqlite3.Row) -> SpanEdge:
    return SpanEdge(
        edge_id=row["edge_id"],
        document_id=row["document_id"],
        from_id=row["from_id"],
        to_id=row["to_id"],
        from_type=row["from_type"],
        to_type=row["to_type"],
        edge_type=row["edge_type"],
        weight=row["weight"],
        source=row["source"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )

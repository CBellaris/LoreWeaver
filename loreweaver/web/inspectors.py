"""Read-only inspectors for the local LoreWeaver debugging UI."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig, load_config
from loreweaver.storage.sqlite_store import SQLiteStore


def load_web_configs(
    *,
    config_path: str = "configs/default.yaml",
    storage_config_path: str = "configs/storage.yaml",
    models_config_path: str = "configs/models.yaml",
) -> tuple[AppConfig, AppConfig, AppConfig]:
    config = load_config(config_path)
    storage_config = load_config(storage_config_path)
    models_config = load_config(models_config_path)
    return config, storage_config, models_config


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class DebugInspector:
    def __init__(
        self,
        *,
        config_path: str = "configs/default.yaml",
        storage_config_path: str = "configs/storage.yaml",
        models_config_path: str = "configs/models.yaml",
    ) -> None:
        self.config_path = config_path
        self.storage_config_path = storage_config_path
        self.models_config_path = models_config_path

    @property
    def configs(self) -> tuple[AppConfig, AppConfig, AppConfig]:
        return load_web_configs(
            config_path=self.config_path,
            storage_config_path=self.storage_config_path,
            models_config_path=self.models_config_path,
        )

    def overview(self) -> dict[str, Any]:
        config, storage_config, models_config = self.configs
        sqlite_path = storage_config.sqlite_path
        sample_path = config.sample_source_path
        counts = _table_counts(sqlite_path)
        latest_document = None
        documents = self.documents()
        if documents:
            latest_document = documents[0]
        reports = self.reports(limit=8)
        providers = models_config.values.get("providers", {})
        env = {
            "SILICONFLOW_API_KEY": bool(os.environ.get("SILICONFLOW_API_KEY")),
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "QDRANT_URL": bool(os.environ.get("QDRANT_URL")),
            "NEO4J_URI": bool(os.environ.get("NEO4J_URI")),
        }
        return {
            "project": config.values.get("project", {}),
            "paths": {
                "config": str(config.path),
                "storage_config": str(storage_config.path),
                "models_config": str(models_config.path),
                "data_dir": str(config.data_dir),
                "sqlite": str(sqlite_path),
                "sample": str(sample_path) if sample_path else None,
            },
            "sample": {
                "exists": bool(sample_path and sample_path.exists()),
                "bytes": sample_path.stat().st_size if sample_path and sample_path.exists() else 0,
            },
            "counts": counts,
            "latest_document": latest_document,
            "recent_reports": reports,
            "env": env,
            "providers": sorted(providers.keys()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def documents(self) -> list[dict[str, Any]]:
        _, storage_config, _ = self.configs
        if not storage_config.sqlite_path.exists():
            return []
        with _connect(storage_config.sqlite_path) as connection:
            if not _table_exists(connection, "documents"):
                return []
            rows = connection.execute(
                """
                SELECT * FROM documents
                ORDER BY created_at DESC, document_id DESC
                """
            ).fetchall()
        return [_row_dict(row) for row in rows]

    def windows(
        self,
        *,
        document_id: str | None = None,
        status: str = "all",
        limit: int = 200,
    ) -> dict[str, Any]:
        _, storage_config, _ = self.configs
        store = SQLiteStore(storage_config.sqlite_path)
        store.initialize()
        store.initialize_extraction_tables()
        document = store.get_document(document_id)
        statuses = store.list_window_extraction_status(document.document_id)
        if status == "pending":
            statuses = [item for item in statuses if item["status"] == "pending"]
        elif status == "extracted":
            statuses = [item for item in statuses if item["status"] == "extracted"]
        elif status != "all":
            raise ValueError("status must be all, pending, or extracted")
        total = len(statuses)
        return {
            "document": jsonable(document),
            "status": status,
            "total": total,
            "windows": statuses[:limit],
        }

    def window_detail(self, window_id: str) -> dict[str, Any]:
        _, storage_config, _ = self.configs
        with _connect(storage_config.sqlite_path) as connection:
            window = connection.execute(
                """
                SELECT w.*, c.chapter_index, c.chapter_title
                FROM candidate_windows w
                JOIN chapters c ON c.chapter_id = w.chapter_id
                WHERE w.window_id = ?
                """,
                (window_id,),
            ).fetchone()
            if window is None:
                raise ValueError(f"Window not found: {window_id}")
            spans = []
            if _table_exists(connection, "spans"):
                spans = [
                    _decode_span(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM spans
                        WHERE window_id = ?
                        ORDER BY span_index_in_window, span_id
                        """,
                        (window_id,),
                    ).fetchall()
                ]
            failures = []
            if _table_exists(connection, "extraction_failures"):
                failures = [
                    _row_dict(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM extraction_failures
                        WHERE window_id = ?
                        ORDER BY id
                        """,
                        (window_id,),
                    ).fetchall()
                ]
        return {
            "window": _row_dict(window),
            "spans": spans,
            "failures": failures,
        }

    def spans(
        self,
        *,
        document_id: str | None = None,
        locator_status: str = "all",
        query: str = "",
        limit: int = 200,
    ) -> dict[str, Any]:
        _, storage_config, _ = self.configs
        store = SQLiteStore(storage_config.sqlite_path)
        store.initialize()
        store.initialize_extraction_tables()
        document = store.get_document(document_id)
        clauses = ["document_id = ?"]
        params: list[Any] = [document.document_id]
        if locator_status != "all":
            clauses.append("locator_status = ?")
            params.append(locator_status)
        if query:
            like = f"%{query}%"
            clauses.append(
                "(span_id LIKE ? OR summary LIKE ? "
                "OR entities_json LIKE ? OR topics_json LIKE ?)"
            )
            params.extend([like, like, like, like])
        params.append(limit)
        with _connect(storage_config.sqlite_path) as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM spans
                WHERE {' AND '.join(clauses)}
                ORDER BY salience_score DESC, span_start_idx ASC, span_id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return {
            "document": jsonable(document),
            "spans": [_decode_span(row) for row in rows],
        }

    def span_detail(self, span_id: str) -> dict[str, Any]:
        _, storage_config, _ = self.configs
        with _connect(storage_config.sqlite_path) as connection:
            span = connection.execute("SELECT * FROM spans WHERE span_id = ?", (span_id,)).fetchone()
            if span is None:
                raise ValueError(f"Span not found: {span_id}")
            candidates = []
            if _table_exists(connection, "locator_candidates"):
                candidates = [
                    _row_dict(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM locator_candidates
                        WHERE span_id = ?
                        ORDER BY confidence DESC, id
                        """,
                        (span_id,),
                    ).fetchall()
                ]
            failures = []
            if _table_exists(connection, "extraction_failures"):
                failures = [
                    _row_dict(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM extraction_failures
                        WHERE span_id = ?
                        ORDER BY id
                        """,
                        (span_id,),
                    ).fetchall()
                ]
        return {
            "span": _decode_span(span),
            "locator_candidates": candidates,
            "failures": failures,
        }

    def span_review(
        self,
        *,
        document_id: str | None = None,
        window_range: str = "",
        with_spans_only: bool = False,
        min_gap_chars: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Return per-window span coverage audits for manual review."""
        _, storage_config, _ = self.configs
        store = SQLiteStore(storage_config.sqlite_path)
        store.initialize()
        store.initialize_extraction_tables()
        document = store.get_document(document_id)
        with _connect(storage_config.sqlite_path) as connection:
            windows = connection.execute(
                """
                SELECT w.*, c.chapter_index, c.chapter_title
                FROM candidate_windows w
                JOIN chapters c ON c.chapter_id = w.chapter_id
                WHERE w.document_id = ?
                ORDER BY c.chapter_index, w.window_index
                """,
                (document.document_id,),
            ).fetchall()
            spans_by_window = _spans_by_window(connection, document.document_id)
            failures_by_window = _failures_by_window(connection, document.document_id)

        total_windows = len(windows)
        range_start, range_end = _parse_review_window_range(
            window_range,
            total_windows=total_windows,
        )
        windows = windows[range_start : range_end + 1] if total_windows else []

        audits = []
        for global_window_index, window_row in enumerate(windows, start=range_start):
            window = _row_dict(window_row)
            spans = spans_by_window.get(window["window_id"], [])
            failures = failures_by_window.get(window["window_id"], [])
            audit = _coverage_audit(
                window=window,
                spans=spans,
                failures=failures,
                global_window_index=global_window_index,
            )
            if with_spans_only and audit["span_count"] == 0:
                continue
            if audit["max_gap_chars"] >= min_gap_chars:
                audits.append(audit)

        audits.sort(
            key=lambda item: (
                item["max_gap_chars"],
                item["uncovered_chars"],
                item["failed_count"],
            ),
            reverse=True,
        )
        summary = _coverage_summary(audits)
        return {
            "document": jsonable(document),
            "summary": summary,
            "windows": audits[:limit],
            "total": len(audits),
            "total_windows": total_windows,
            "window_range": {
                "input": window_range,
                "start": range_start,
                "end": range_end,
            },
            "with_spans_only": with_spans_only,
            "limit": limit,
            "min_gap_chars": min_gap_chars,
        }

    def span_review_window(self, window_id: str) -> dict[str, Any]:
        _, storage_config, _ = self.configs
        with _connect(storage_config.sqlite_path) as connection:
            window = connection.execute(
                """
                SELECT w.*, c.chapter_index, c.chapter_title
                FROM candidate_windows w
                JOIN chapters c ON c.chapter_id = w.chapter_id
                WHERE w.window_id = ?
                """,
                (window_id,),
            ).fetchone()
            if window is None:
                raise ValueError(f"Window not found: {window_id}")
            spans = []
            if _table_exists(connection, "spans"):
                spans = [
                    _decode_span(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM spans
                        WHERE window_id = ?
                        ORDER BY span_index_in_window, span_id
                        """,
                        (window_id,),
                    ).fetchall()
                ]
            failures = []
            if _table_exists(connection, "extraction_failures"):
                failures = [
                    _row_dict(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM extraction_failures
                        WHERE window_id = ?
                        ORDER BY id
                        """,
                        (window_id,),
                    ).fetchall()
                ]
        window_payload = _row_dict(window)
        spans = _attach_failure_reasons(spans, failures)
        audit = _coverage_audit(
            window=window_payload,
            spans=spans,
            failures=failures,
        )
        return {
            "window": window_payload,
            "audit": audit,
            "spans": spans,
            "failures": failures,
            "gaps": _gap_details(window_payload, audit["intervals"], spans),
            "segments": _coverage_segments(window_payload, spans),
        }

    def graph_summary(self, *, document_id: str | None = None) -> dict[str, Any]:
        _, storage_config, _ = self.configs
        store = SQLiteStore(storage_config.sqlite_path)
        store.initialize()
        store.initialize_graph_tables()
        document = store.get_document(document_id)
        clusters = store.list_center_span_clusters(document.document_id)
        edges = store.list_span_edges(document.document_id)
        return {
            "document": jsonable(document),
            "cluster_count": len(clusters),
            "edge_count": len(edges),
            "clusters": jsonable(clusters[:40]),
            "edge_counts": _counts_by_key(jsonable(edges), "edge_type"),
        }

    def reports(self, *, limit: int = 80) -> list[dict[str, Any]]:
        config, _, _ = self.configs
        runs_dir = config.data_dir / "runs"
        if not runs_dir.exists():
            return []
        reports = []
        for path in runs_dir.glob("*.json"):
            stat = path.stat()
            kind = _report_kind(path.name)
            summary: dict[str, Any] = {}
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                summary = {
                    "run_id": payload.get("run_id"),
                    "query_id": payload.get("query_id"),
                    "document_id": payload.get("document_id")
                    or (payload.get("document") or {}).get("document_id"),
                    "question": payload.get("question"),
                }
            except (OSError, json.JSONDecodeError):
                summary = {"error": "failed_to_parse"}
            reports.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "kind": kind,
                    "bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "summary": summary,
                }
            )
        return sorted(reports, key=lambda item: item["mtime"], reverse=True)[:limit]

    def report(self, name: str) -> dict[str, Any]:
        config, _, _ = self.configs
        path = _safe_run_file(config.data_dir / "runs", name)
        return {
            "name": path.name,
            "path": str(path),
            "payload": json.loads(path.read_text(encoding="utf-8")),
        }


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _table_counts(path: Path) -> dict[str, int]:
    tables = [
        "documents",
        "chapters",
        "candidate_windows",
        "spans",
        "locator_candidates",
        "extraction_failures",
        "embedding_cache",
        "center_span_clusters",
        "span_edges",
        "query_runs",
        "evidence_packs",
    ]
    if not path.exists():
        return {table: 0 for table in tables}
    with _connect(path) as connection:
        return {
            table: _count_table(connection, table)
            for table in tables
        }


def _count_table(connection: sqlite3.Connection, table: str) -> int:
    if not _table_exists(connection, table):
        return 0
    row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"] or 0)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _decode_span(row: sqlite3.Row) -> dict[str, Any]:
    payload = _row_dict(row)
    payload["entities"] = _loads(payload.pop("entities_json", "[]"))
    payload["topics"] = _loads(payload.pop("topics_json", "[]"))
    return payload


def _spans_by_window(
    connection: sqlite3.Connection,
    document_id: str,
) -> dict[str, list[dict[str, Any]]]:
    if not _table_exists(connection, "spans"):
        return {}
    rows = connection.execute(
        """
        SELECT * FROM spans
        WHERE document_id = ?
        ORDER BY window_start, span_index_in_window, span_id
        """,
        (document_id,),
    ).fetchall()
    spans_by_window: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        span = _decode_span(row)
        spans_by_window.setdefault(span["window_id"], []).append(span)
    return spans_by_window


def _failures_by_window(
    connection: sqlite3.Connection,
    document_id: str,
) -> dict[str, list[dict[str, Any]]]:
    if not _table_exists(connection, "extraction_failures"):
        return {}
    rows = connection.execute(
        """
        SELECT f.*
        FROM extraction_failures f
        JOIN candidate_windows w ON w.window_id = f.window_id
        WHERE w.document_id = ?
        ORDER BY f.id
        """,
        (document_id,),
    ).fetchall()
    failures_by_window: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        failure = _row_dict(row)
        failures_by_window.setdefault(failure["window_id"], []).append(failure)
    return failures_by_window


def _attach_failure_reasons(
    spans: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reasons_by_span: dict[str, list[str]] = {}
    for failure in failures:
        span_id = failure.get("span_id")
        if span_id:
            reasons_by_span.setdefault(str(span_id), []).append(str(failure.get("reason") or ""))
    return [
        {
            **span,
            "failure_reasons": reasons_by_span.get(str(span.get("span_id")), []),
        }
        for span in spans
    ]


def _coverage_audit(
    *,
    window: dict[str, Any],
    spans: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    global_window_index: int | None = None,
) -> dict[str, Any]:
    intervals = _merged_located_intervals(window, spans)
    gaps = _gap_intervals(window, intervals)
    window_chars = max(0, int(window["window_end"]) - int(window["window_start"]))
    covered_chars = sum(end - start for start, end in intervals)
    uncovered_chars = max(0, window_chars - covered_chars)
    located_count = sum(1 for span in spans if span.get("locator_status") == "located")
    failed_count = len(spans) - located_count
    failure_reasons = [str(failure.get("reason") or "") for failure in failures]
    max_gap_chars = max((end - start for start, end in gaps), default=0)
    hint_tags = _coverage_hint_tags(
        failed_count=failed_count,
        failure_reasons=failure_reasons,
        max_gap_chars=max_gap_chars,
    )
    return {
        "window_id": window["window_id"],
        "global_window_index": global_window_index,
        "chapter_id": window["chapter_id"],
        "chapter_index": window.get("chapter_index"),
        "chapter_title": window.get("chapter_title", ""),
        "window_index": window["window_index"],
        "window_start": window["window_start"],
        "window_end": window["window_end"],
        "char_count": window_chars,
        "span_count": len(spans),
        "located_count": located_count,
        "failed_count": failed_count,
        "gap_count": len(gaps),
        "covered_chars": covered_chars,
        "uncovered_chars": uncovered_chars,
        "coverage_ratio": round(covered_chars / window_chars, 4) if window_chars else 0.0,
        "max_gap_chars": max_gap_chars,
        "hint_tags": hint_tags,
        "intervals": [{"start_idx": start, "end_idx": end} for start, end in intervals],
        "text_preview": _clip_text(str(window.get("text") or ""), 160),
        "uncovered_preview": _clip_text(str(window.get("uncovered_text") or ""), 160),
    }


def _merged_located_intervals(
    window: dict[str, Any],
    spans: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    window_start = int(window["window_start"])
    window_end = int(window["window_end"])
    intervals = []
    for span in spans:
        if span.get("locator_status") != "located":
            continue
        if span.get("span_start_idx") is None or span.get("span_end_idx") is None:
            continue
        start = max(window_start, int(span["span_start_idx"]))
        end = min(window_end, int(span["span_end_idx"]))
        if start < end:
            intervals.append((start, end))
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _gap_intervals(
    window: dict[str, Any],
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    cursor = int(window["window_start"])
    gaps = []
    for start, end in intervals:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < int(window["window_end"]):
        gaps.append((cursor, int(window["window_end"])))
    return gaps


def _coverage_hint_tags(
    *,
    failed_count: int,
    failure_reasons: list[str],
    max_gap_chars: int,
) -> list[str]:
    tags = []
    if failed_count:
        tags.append("locator_failed_likely")
    if any("outside" in reason or "length" in reason for reason in failure_reasons):
        tags.append("bounds_rejected_likely")
    if max_gap_chars >= 240 and not failed_count:
        tags.append("needs_manual_review")
    if not tags:
        tags.append("model_skipped_likely")
    return tags


def _gap_details(
    window: dict[str, Any],
    intervals: list[dict[str, int]],
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tuple_intervals = [(int(item["start_idx"]), int(item["end_idx"])) for item in intervals]
    gaps = _gap_intervals(window, tuple_intervals)
    located_spans = [
        span
        for span in spans
        if span.get("locator_status") == "located"
        and span.get("span_start_idx") is not None
        and span.get("span_end_idx") is not None
    ]
    details = []
    for index, (start, end) in enumerate(gaps):
        left = _nearest_left_span(located_spans, start)
        right = _nearest_right_span(located_spans, end)
        text = _slice_window_text(window, start, end)
        details.append(
            {
                "gap_index": index,
                "start_idx": start,
                "end_idx": end,
                "relative_start": start - int(window["window_start"]),
                "relative_end": end - int(window["window_start"]),
                "char_count": end - start,
                "text_preview": _clip_text(text, 240),
                "left_span_id": left.get("span_id") if left else "",
                "right_span_id": right.get("span_id") if right else "",
            }
        )
    return details


def _coverage_segments(
    window: dict[str, Any],
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    window_start = int(window["window_start"])
    window_end = int(window["window_end"])
    located_spans = [
        span
        for span in spans
        if span.get("locator_status") == "located"
        and span.get("span_start_idx") is not None
        and span.get("span_end_idx") is not None
    ]
    boundaries = {window_start, window_end}
    for span in located_spans:
        boundaries.add(max(window_start, int(span["span_start_idx"])))
        boundaries.add(min(window_end, int(span["span_end_idx"])))
    sorted_boundaries = sorted(boundaries)
    segments = []
    gap_index = 0
    for start, end in zip(sorted_boundaries, sorted_boundaries[1:]):
        if start >= end:
            continue
        active_spans = [
            span["span_id"]
            for span in located_spans
            if int(span["span_start_idx"]) < end and int(span["span_end_idx"]) > start
        ]
        segment: dict[str, Any] = {
            "start_idx": start,
            "end_idx": end,
            "relative_start": start - window_start,
            "relative_end": end - window_start,
            "text": _slice_window_text(window, start, end),
            "span_ids": active_spans,
        }
        if not active_spans:
            segment["gap_index"] = gap_index
            gap_index += 1
        segments.append(segment)
    return segments


def _nearest_left_span(spans: list[dict[str, Any]], start: int) -> dict[str, Any] | None:
    candidates = [span for span in spans if int(span["span_end_idx"]) <= start]
    if not candidates:
        return None
    return max(candidates, key=lambda span: int(span["span_end_idx"]))


def _nearest_right_span(spans: list[dict[str, Any]], end: int) -> dict[str, Any] | None:
    candidates = [span for span in spans if int(span["span_start_idx"]) >= end]
    if not candidates:
        return None
    return min(candidates, key=lambda span: int(span["span_start_idx"]))


def _slice_window_text(window: dict[str, Any], start: int, end: int) -> str:
    relative_start = start - int(window["window_start"])
    relative_end = end - int(window["window_start"])
    return str(window.get("text") or "")[relative_start:relative_end]


def _coverage_summary(audits: list[dict[str, Any]]) -> dict[str, Any]:
    total_chars = sum(int(item["char_count"]) for item in audits)
    covered_chars = sum(int(item["covered_chars"]) for item in audits)
    return {
        "window_count": len(audits),
        "covered_chars": covered_chars,
        "uncovered_chars": sum(int(item["uncovered_chars"]) for item in audits),
        "coverage_ratio": round(covered_chars / total_chars, 4) if total_chars else 0.0,
        "max_gap_chars": max((int(item["max_gap_chars"]) for item in audits), default=0),
        "failed_window_count": sum(1 for item in audits if int(item["failed_count"]) > 0),
        "review_window_count": sum(
            1 for item in audits if "needs_manual_review" in item["hint_tags"]
        ),
    }


def _parse_review_window_range(range_text: str, *, total_windows: int) -> tuple[int, int]:
    if total_windows <= 0:
        return 0, -1
    value = range_text.strip()
    if not value:
        return 0, total_windows - 1
    separator = "-" if "-" in value else ":"
    if separator not in value:
        start = end = int(value)
    else:
        raw_start, raw_end = value.split(separator, 1)
        start = int(raw_start.strip())
        end = int(raw_end.strip())
    if start < 0 or end < start or end >= total_windows:
        raise ValueError(
            f"Invalid window range {range_text!r}; expected 0-based range within 0-{total_windows - 1}."
        )
    return start, end


def _clip_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(0, limit - 3)]}..."


def _loads(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _counts_by_key(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _report_kind(name: str) -> str:
    suffixes = [
        ("_ingest_report.json", "ingest"),
        ("_window_report.json", "windows"),
        ("_extraction_report.json", "extract"),
        ("_index_report.json", "index"),
        ("_graph_report.json", "graph"),
        ("_retrieval_report.json", "retrieve"),
        ("_evidence_pack.json", "evidence"),
        ("_answer_report.json", "ask"),
    ]
    for suffix, kind in suffixes:
        if name.endswith(suffix):
            return kind
    return "report"


def _safe_run_file(runs_dir: Path, name: str) -> Path:
    path = (runs_dir / Path(name).name).resolve()
    root = runs_dir.resolve()
    if root not in path.parents or path.suffix != ".json" or not path.exists():
        raise ValueError(f"Report not found: {name}")
    return path

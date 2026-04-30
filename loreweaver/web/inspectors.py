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
                "(span_id LIKE ? OR micro_topic LIKE ? OR micro_summary LIKE ? "
                "OR entities_json LIKE ? OR topics_json LIKE ?)"
            )
            params.extend([like, like, like, like, like])
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

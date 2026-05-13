"""Background job runner for the local debugging UI."""

from __future__ import annotations

import os
import queue
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loreweaver.config import load_config
from loreweaver.evidence.assembler import assemble_evidence_pack_from_retrieval_report
from loreweaver.eval.corpus import build_chapter_corpus
from loreweaver.eval.generator import generate_question_set
from loreweaver.eval.runner import run_eval, summarize_eval_run
from loreweaver.extraction.extractor import extract_document_windows, list_extraction_windows
from loreweaver.graph.center_span import build_m15_graph, list_graph_clusters
from loreweaver.ingest.pipeline import ingest_text
from loreweaver.ingest.window_splitter import build_candidate_windows
from loreweaver.indexing.pipeline import build_m14_indexes, search_bm25_index, search_vector_index
from loreweaver.logging import new_run_id
from loreweaver.progress import ProgressEvent, ProgressReporter, ProgressSink
from loreweaver.qa.answerer import ask_m18
from loreweaver.retrieval.pipeline import retrieve_m16
from loreweaver.storage.sqlite_store import SQLiteStore
from loreweaver.web.inspectors import jsonable


JobEvent = dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WebJob:
    job_id: str
    command: str
    payload: dict[str, Any]
    raw_payload: dict[str, Any] = field(repr=False)
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] | None = None
    events: "queue.Queue[JobEvent]" = field(default_factory=queue.Queue)
    cancel_requested: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "command": self.command,
            "payload": self.payload,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
        }


class JobManager:
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
        self._jobs: dict[str, WebJob] = {}
        self._lock = threading.Lock()
        self._env_lock = threading.Lock()

    def start(self, command: str, payload: dict[str, Any]) -> WebJob:
        job = WebJob(
            job_id=new_run_id(f"web_{command}"),
            command=command,
            payload=_redact_payload(payload),
            raw_payload=dict(payload),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> WebJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [job.snapshot() for job in sorted(jobs, key=lambda item: item.created_at, reverse=True)]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.cancel_requested.set()
        _emit(job, "cancel_requested", {})
        return True

    def _run(self, job: WebJob) -> None:
        job.status = "running"
        job.started_at = _now()
        _emit(job, "started", {"command": job.command})
        progress = ProgressReporter(
            command=job.command,
            run_id=job.job_id,
            sinks=[WebProgressSink(job)],
        )
        try:
            result = _dispatch(
                command=job.command,
                payload=job.raw_payload,
                config_path=self.config_path,
                storage_config_path=self.storage_config_path,
                models_config_path=self.models_config_path,
                env_lock=self._env_lock,
                progress=progress,
            )
            job.result = jsonable(result)
            job.status = "completed"
            _emit(job, "completed", {"result": job.result})
        except Exception as error:  # pragma: no cover - surfaced in UI.
            job.error = str(error)
            job.status = "failed"
            _emit(
                job,
                "failed",
                {"error": str(error), "traceback": traceback.format_exc(limit=20)},
            )
        finally:
            job.completed_at = _now()
            _emit(job, "terminal", {"status": job.status})


def command_specs() -> dict[str, Any]:
    return {
        "status": {"label": "Status", "fields": []},
        "ingest": {
            "label": "Ingest",
            "fields": ["source", "title", "author", "max_chapters"],
        },
        "windows": {
            "label": "Windows",
            "fields": ["document_id", "window_size", "overlap_ratio", "min_chars", "max_chars", "by_chapter"],
        },
        "extract": {
            "label": "Extract",
            "fields": [
                "document_id",
                "limit",
                "offset",
                "window_id",
                "window_range",
                "list_windows",
                "only",
                "mock",
                "batch",
                "batch_id",
                "batch_wait",
                "batch_poll_interval",
                "batch_timeout",
                "batch_completion_window",
                "repair_failed",
                "span_chars_min",
                "span_chars_max",
            ],
        },
        "index": {"label": "Index", "fields": ["document_id", "limit", "mock_embeddings"]},
        "search-vector": {"label": "Search Vector", "fields": ["query", "document_id", "top_k", "mock_embeddings"]},
        "search-bm25": {"label": "Search BM25", "fields": ["query", "document_id", "top_k"]},
        "spans": {"label": "Spans", "fields": ["document_id", "top_salience"]},
        "graph": {
            "label": "Graph",
            "fields": [
                "document_id",
                "cluster_count",
                "members_per_cluster",
                "min_members",
                "sync_neo4j",
                "no_neo4j",
                "no_embeddings",
                "list",
                "cluster_id",
            ],
        },
        "retrieve": {"label": "Retrieve", "fields": ["question", "document_id", "mock_embeddings", "mock_reranker", "no_reranker"]},
        "evidence": {"label": "Evidence", "fields": ["question", "document_id", "mock_embeddings", "mock_reranker", "no_reranker"]},
        "ask": {
            "label": "Ask",
            "fields": ["question", "document_id", "mock_embeddings", "mock_reranker", "no_reranker", "mock_answer"],
        },
        "eval-build-corpus": {
            "label": "Eval Build Corpus",
            "fields": ["document_id", "chapter_start", "chapter_end", "output"],
        },
        "eval-generate": {
            "label": "Eval Generate",
            "fields": ["corpus", "question_count", "profile", "max_output_tokens", "output"],
        },
        "eval-run": {
            "label": "Eval Run",
            "fields": [
                "questions",
                "document_id",
                "output",
                "limit",
                "mock_embeddings",
                "mock_reranker",
                "no_reranker",
            ],
        },
        "eval-report": {
            "label": "Eval Report",
            "fields": ["predictions"],
        },
    }


def _dispatch(
    *,
    command: str,
    payload: dict[str, Any],
    config_path: str,
    storage_config_path: str,
    models_config_path: str,
    env_lock: threading.Lock,
    progress: ProgressReporter | None,
) -> dict[str, Any]:
    env_overrides = _env_overrides(payload)
    safe_payload = _payload_without_env(payload)
    with _temporary_env(env_overrides, env_lock):
        return _dispatch_inner(
            command=command,
            payload=safe_payload,
            config_path=config_path,
            storage_config_path=storage_config_path,
            models_config_path=models_config_path,
            progress=progress,
        )


def _dispatch_inner(
    *,
    command: str,
    payload: dict[str, Any],
    config_path: str,
    storage_config_path: str,
    models_config_path: str,
    progress: ProgressReporter | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    storage_config = load_config(storage_config_path)
    models_config = load_config(models_config_path)

    if command == "status":
        return {
            "run_id": new_run_id("status"),
            "stage": config.values.get("project", {}).get("stage", "unknown"),
            "config": str(config.path),
            "data_dir": str(config.data_dir),
            "sqlite_path": str(storage_config.sqlite_path),
        }
    if command == "ingest":
        run_id = new_run_id("ingest")
        source = _optional_path(payload.get("source")) or config.sample_source_path
        if source is None:
            raise ValueError("source is required when sample.source_path is not configured")
        return ingest_text(
            config=config,
            storage_config=storage_config,
            run_id=run_id,
            source_path=source,
            title=_optional_str(payload.get("title")),
            author=_optional_str(payload.get("author")),
            max_chapters=_optional_int(payload.get("max_chapters")),
            progress=progress.child(command="ingest", run_id=run_id) if progress is not None else None,
        )
    if command == "windows":
        return build_candidate_windows(
            config=config,
            storage_config=storage_config,
            run_id=new_run_id("windows"),
            document_id=_optional_str(payload.get("document_id")),
            window_size_chars=_optional_int(payload.get("window_size")),
            overlap_ratio=_optional_float(payload.get("overlap_ratio")),
            min_window_chars=_optional_int(payload.get("min_chars")),
            max_window_chars=_optional_int(payload.get("max_chars")),
            split_by_chapter=bool(payload.get("by_chapter")),
            progress=progress.child(command="windows") if progress is not None else None,
        )
    if command == "extract":
        if bool(payload.get("list_windows")):
            return list_extraction_windows(
                storage_config=storage_config,
                document_id=_optional_str(payload.get("document_id")),
                only=str(payload.get("only") or "all"),
                limit=_optional_int(payload.get("limit")),
            )
        return extract_document_windows(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            run_id=new_run_id("extract"),
            document_id=_optional_str(payload.get("document_id")),
            limit=_optional_int(payload.get("limit")),
            offset=int(payload.get("offset") or 0),
            window_ids=_list_value(payload.get("window_id")),
            window_ranges=_list_value(payload.get("window_range")),
            mock=bool(payload.get("mock")),
            batch=bool(payload.get("batch")),
            batch_id=_optional_str(payload.get("batch_id")),
            batch_model=_optional_str(payload.get("batch_model")),
            batch_wait=bool(payload.get("batch_wait")),
            batch_poll_interval_seconds=float(payload.get("batch_poll_interval") or 30.0),
            batch_timeout_seconds=_optional_float(payload.get("batch_timeout")),
            batch_completion_window=str(payload.get("batch_completion_window") or "24h"),
            repair_failed=bool(payload.get("repair_failed")),
            span_chars_min=_optional_int(payload.get("span_chars_min")),
            span_chars_max=_optional_int(payload.get("span_chars_max")),
            progress=progress,
        )
    if command == "index":
        return build_m14_indexes(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            run_id=new_run_id("index"),
            document_id=_optional_str(payload.get("document_id")),
            limit=_optional_int(payload.get("limit")),
            mock_embeddings=bool(payload.get("mock_embeddings")),
            progress=progress,
        )
    if command == "search-vector":
        return search_vector_index(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            query=_required(payload, "query"),
            document_id=_optional_str(payload.get("document_id")),
            top_k=int(payload.get("top_k") or 5),
            mock_embeddings=bool(payload.get("mock_embeddings")),
        )
    if command == "search-bm25":
        return search_bm25_index(
            storage_config=storage_config,
            query=_required(payload, "query"),
            document_id=_optional_str(payload.get("document_id")),
            top_k=int(payload.get("top_k") or 5),
        )
    if command == "spans":
        store = SQLiteStore(storage_config.sqlite_path)
        store.initialize()
        document = store.get_document(_optional_str(payload.get("document_id")))
        spans = store.list_top_salience_spans(
            document.document_id,
            limit=int(payload.get("top_salience") or 30),
        )
        return {"document": document, "spans": spans}
    if command == "graph":
        if bool(payload.get("list")):
            return list_graph_clusters(
                storage_config=storage_config,
                document_id=_optional_str(payload.get("document_id")),
                cluster_id=_optional_str(payload.get("cluster_id")),
            )
        return build_m15_graph(
            config=config,
            storage_config=storage_config,
            run_id=new_run_id("graph"),
            document_id=_optional_str(payload.get("document_id")),
            cluster_count=_optional_int(payload.get("cluster_count")),
            members_per_cluster=_optional_int(payload.get("members_per_cluster")),
            min_members=_optional_int(payload.get("min_members")),
            use_embeddings=False if bool(payload.get("no_embeddings")) else None,
            sync_neo4j=_sync_neo4j(payload),
            progress=progress,
        )
    if command == "retrieve":
        return retrieve_m16(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            question=_required(payload, "question"),
            document_id=_optional_str(payload.get("document_id")),
            mock_embeddings=bool(payload.get("mock_embeddings")),
            mock_reranker=bool(payload.get("mock_reranker")),
            no_reranker=bool(payload.get("no_reranker")),
            progress=progress,
        )
    if command == "evidence":
        retrieval_report = retrieve_m16(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            question=_required(payload, "question"),
            document_id=_optional_str(payload.get("document_id")),
            mock_embeddings=bool(payload.get("mock_embeddings")),
            mock_reranker=bool(payload.get("mock_reranker")),
            no_reranker=bool(payload.get("no_reranker")),
            progress=progress.child(command="retrieve") if progress is not None else None,
        )
        return assemble_evidence_pack_from_retrieval_report(
            config=config,
            storage_config=storage_config,
            retrieval_report=retrieval_report,
            progress=progress,
        )
    if command == "ask":
        return ask_m18(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            question=_required(payload, "question"),
            document_id=_optional_str(payload.get("document_id")),
            mock_embeddings=bool(payload.get("mock_embeddings")),
            mock_reranker=bool(payload.get("mock_reranker")),
            no_reranker=bool(payload.get("no_reranker")),
            mock_answer=bool(payload.get("mock_answer")),
            progress=progress,
        )
    if command == "eval-build-corpus":
        return build_chapter_corpus(
            config=config,
            storage_config=storage_config,
            document_id=_optional_str(payload.get("document_id")),
            chapter_start=int(payload.get("chapter_start") or 1),
            chapter_end=int(payload.get("chapter_end") or 100),
            output_path=_optional_str(payload.get("output")),
        )
    if command == "eval-generate":
        return generate_question_set(
            config=config,
            models_config=models_config,
            corpus_path=_required(payload, "corpus"),
            output_path=_optional_str(payload.get("output")),
            question_count=int(payload.get("question_count") or 200),
            profile=str(payload.get("profile") or "broad"),
            max_output_tokens=_optional_int(payload.get("max_output_tokens")),
        )
    if command == "eval-run":
        return run_eval(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            question_set_path=_required(payload, "questions"),
            document_id=_optional_str(payload.get("document_id")),
            output_path=_optional_str(payload.get("output")),
            limit=_optional_int(payload.get("limit")),
            mock_embeddings=bool(payload.get("mock_embeddings")),
            mock_reranker=bool(payload.get("mock_reranker")),
            no_reranker=bool(payload.get("no_reranker")),
            progress=progress,
        )
    if command == "eval-report":
        return summarize_eval_run(_required(payload, "predictions"))
    raise ValueError(f"Unsupported command: {command}")


class WebProgressSink(ProgressSink):
    def __init__(self, job: WebJob) -> None:
        self.job = job

    def emit(self, event: ProgressEvent) -> None:
        job = self.job
        if job.cancel_requested.is_set():
            raise RuntimeError("Job cancelled")
        payload = event.to_dict()
        job.progress = payload
        _emit(job, event.name, payload)


def _emit(job: WebJob, event: str, payload: dict[str, Any]) -> None:
    job.events.put(
        {
            "event": event,
            "payload": jsonable(payload),
            "time": _now(),
        }
    )


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    env = redacted.get("_env")
    if isinstance(env, dict) and env:
        redacted["_env"] = {str(key): "<redacted>" for key in env}
    return redacted


def _payload_without_env(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "_env"}


def _env_overrides(payload: dict[str, Any]) -> dict[str, str]:
    raw_env = payload.get("_env")
    if not isinstance(raw_env, dict):
        return {}
    allowed = {"DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY"}
    overrides = {}
    for key, value in raw_env.items():
        name = str(key)
        text = _optional_str(value)
        if name in allowed and text:
            overrides[name] = text
    return overrides


@contextmanager
def _temporary_env(overrides: dict[str, str], lock: threading.Lock):
    with lock:
        old_values = {key: os.environ.get(key) for key in overrides}
        try:
            if overrides:
                os.environ.update(overrides)
            yield
        finally:
            for key, old_value in old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    text = _optional_str(value)
    return int(text) if text is not None else None


def _optional_float(value: Any) -> float | None:
    text = _optional_str(value)
    return float(text) if text is not None else None


def _optional_path(value: Any) -> Path | None:
    text = _optional_str(value)
    return Path(text) if text is not None else None


def _list_value(value: Any) -> list[str] | None:
    text = _optional_str(value)
    if text is None:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def _required(payload: dict[str, Any], key: str) -> str:
    value = _optional_str(payload.get(key))
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def _sync_neo4j(payload: dict[str, Any]) -> bool | None:
    if bool(payload.get("sync_neo4j")):
        return True
    if bool(payload.get("no_neo4j")):
        return False
    return None

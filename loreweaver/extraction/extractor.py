"""LLM extraction runner for M1.3."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover - exercised in minimal bootstrap envs.
    ValidationError = ValueError  # type: ignore[misc, assignment]

from loreweaver.config import AppConfig
from loreweaver.extraction.locator import (
    LocatorResult,
    anchor_constraints_ok,
    locate_span_anchors,
)
from loreweaver.extraction.prompts import build_extraction_messages
from loreweaver.extraction.retry import RetryPolicy
from loreweaver.extraction.schemas import SpanCandidatePayload, WindowExtractionPayload
from loreweaver.models.span import Span
from loreweaver.models.window import CandidateWindow
from loreweaver.storage.sqlite_store import SQLiteStore


BATCH_MAX_REQUESTS_PER_FILE = 5000


class ChatClient(Protocol):
    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        """Return raw JSON text and token usage."""


@dataclass(frozen=True)
class BatchSubmission:
    batch_id: str
    input_file_id: str
    status: str
    output_file_id: str | None
    error_file_id: str | None
    request_counts: dict[str, int]


@dataclass(frozen=True)
class BatchStatus:
    batch_id: str
    status: str
    input_file_id: str | None
    output_file_id: str | None
    error_file_id: str | None
    request_counts: dict[str, int]


@dataclass(frozen=True)
class BatchOutput:
    custom_id: str
    raw_output: str | None
    usage: dict[str, int]
    error: str | None


ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class TokenPrice:
    input_yuan_per_1k: float = 0.0
    output_yuan_per_1k: float = 0.0


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_yuan_per_1k: float
    output_yuan_per_1k: float
    estimated_yuan: float


@dataclass(frozen=True)
class ExtractionResult:
    span: Span
    payload: SpanCandidatePayload | None
    locator_result: LocatorResult | None
    status: str
    failure_reason: str | None
    raw_output: str | None
    attempts: int
    usage: dict[str, int]
    cost: CostEstimate


class WindowPayloadParseError(ValueError):
    """Raised when a full window payload cannot be parsed or schema-validated."""


class OpenAIChatClient:
    """OpenAI-compatible chat client used for OpenAI and SiliconFlow."""

    def __init__(
        self,
        *,
        api_key_env: str,
        base_url: str | None = None,
        json_response_format: bool = False,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {api_key_env}")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "The openai package is required for live extraction. "
                "Install optional M1 dependencies first."
            ) from error

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._json_response_format = json_response_format

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if self._json_response_format:
            request["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage is not None:
            usage = {
                "input_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
            }
        return content, usage

    def submit_chat_batch(
        self,
        *,
        input_path: Path,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> BatchSubmission:
        with input_path.open("rb") as file_obj:
            uploaded = self._client.files.create(file=file_obj, purpose="batch")
        input_file_id = _uploaded_file_id(uploaded)
        batch = self._client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata or {},
        )
        return BatchSubmission(
            batch_id=str(batch.id),
            input_file_id=input_file_id,
            status=str(getattr(batch, "status", "")),
            output_file_id=_optional_str(getattr(batch, "output_file_id", None)),
            error_file_id=_optional_str(getattr(batch, "error_file_id", None)),
            request_counts=_request_counts_dict(getattr(batch, "request_counts", None)),
        )

    def retrieve_chat_batch(self, batch_id: str) -> BatchStatus:
        batch = self._client.batches.retrieve(batch_id)
        return BatchStatus(
            batch_id=str(batch.id),
            status=str(getattr(batch, "status", "")),
            input_file_id=_optional_str(getattr(batch, "input_file_id", None)),
            output_file_id=_optional_str(getattr(batch, "output_file_id", None)),
            error_file_id=_optional_str(getattr(batch, "error_file_id", None)),
            request_counts=_request_counts_dict(getattr(batch, "request_counts", None)),
        )

    def download_file_text(self, file_id: str) -> str:
        if file_id.startswith(("http://", "https://")):
            with urllib.request.urlopen(file_id, timeout=60) as response:
                return response.read().decode("utf-8")
        content = self._client.files.content(file_id)
        if hasattr(content, "text"):
            return str(content.text)
        if hasattr(content, "content"):
            raw_content = content.content
            if isinstance(raw_content, bytes):
                return raw_content.decode("utf-8")
            return str(raw_content)
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return str(content)


class MockChatClient:
    """Deterministic local extractor for tests and dry M1.3 plumbing checks."""

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        del model, temperature
        user_text = messages[-1]["content"]
        match = re.search(r"<<<WINDOW_TEXT\n(?P<text>.*)\nWINDOW_TEXT>>>", user_text, re.S)
        window_text = match.group("text") if match else user_text
        compact = " ".join(window_text.split())
        first_start = compact[: min(40, len(compact))]
        first_end = compact[min(len(compact), 80) : min(len(compact), 120)] or compact[-40:]
        second_start = compact[min(len(compact), 40) : min(len(compact), 80)] or first_start
        second_end = compact[min(len(compact), 120) : min(len(compact), 160)] or first_end
        payload = {
            "spans": [
                {
                    "micro_topic": "窗口开头事件",
                    "span_type": "event",
                    "micro_summary": compact[:100] or "空窗口",
                    "entities": [],
                    "topics": ["mock_extraction"],
                    "salience_score": 0.5,
                    "start_anchor_quote": first_start,
                    "end_anchor_quote": first_end,
                    "key_quote": first_start,
                    "overlap_reason": "",
                },
                {
                    "micro_topic": "窗口后续线索",
                    "span_type": "mystery_clue",
                    "micro_summary": compact[40:140] or compact[:100] or "空窗口",
                    "entities": [],
                    "topics": ["mock_extraction"],
                    "salience_score": 0.45,
                    "start_anchor_quote": second_start,
                    "end_anchor_quote": second_end,
                    "key_quote": second_start,
                    "overlap_reason": "mock 模式用于验证一个窗口多 Span 落库。",
                },
            ]
        }
        raw = json.dumps(payload, ensure_ascii=False)
        usage = {
            "input_tokens": estimate_tokens(user_text),
            "output_tokens": estimate_tokens(raw),
            "total_tokens": estimate_tokens(user_text) + estimate_tokens(raw),
        }
        return raw, usage


def extract_document_windows(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    models_config: AppConfig,
    run_id: str,
    document_id: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    window_id: str | None = None,
    window_ids: list[str] | None = None,
    window_ranges: list[str] | None = None,
    mock: bool = False,
    batch: bool = False,
    batch_id: str | None = None,
    batch_model: str | None = None,
    batch_wait: bool = False,
    batch_poll_interval_seconds: float = 30.0,
    batch_timeout_seconds: float | None = None,
    batch_completion_window: str = "24h",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run M1.3 extraction over persisted candidate windows and write reports."""
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_extraction_tables()

    document = store.get_document(document_id)
    windows = store.list_candidate_windows(document.document_id)
    if not windows:
        raise ValueError(f"No candidate windows found for document_id={document.document_id}")
    selected_window_ids = _normalize_window_ids(window_id=window_id, window_ids=window_ids)
    if batch_id:
        windows = windows[offset:]
        if limit is not None:
            windows = windows[:limit]
    elif selected_window_ids or window_ranges:
        windows = _select_windows(
            windows,
            window_ids=selected_window_ids,
            window_ranges=window_ranges or [],
        )
    else:
        windows = windows[offset:]
        if limit is not None:
            windows = windows[:limit]

    extraction_config = config.values.get("extraction", {})
    locator_config = config.values.get("locator", {})
    model_settings = _model_settings(models_config)
    model_name = str(extraction_config.get("model") or model_settings["model"])
    batch_mode = batch or bool(batch_id)
    if batch_mode:
        model_name = str(
            batch_model
            or extraction_config.get("batch_model")
            or model_settings.get("batch_model")
            or "deepseek-ai/DeepSeek-V3.1-Terminus"
        )
    temperature = float(extraction_config.get("temperature", model_settings["temperature"]))
    retry_policy = RetryPolicy(max_retries=int(extraction_config.get("max_retries", 2)))
    min_spans_per_window = int(extraction_config.get("min_spans_per_window", 2))
    max_spans_per_window = int(extraction_config.get("max_spans_per_window", 12))
    anchor_min_chars = int(extraction_config.get("anchor_min_chars", 8))
    anchor_max_chars = int(extraction_config.get("anchor_max_chars", 80))
    target_span_chars_min = int(extraction_config.get("target_span_chars_min", 80))
    target_span_chars_max = int(extraction_config.get("target_span_chars_max", 800))
    store_located_text = bool(extraction_config.get("store_located_text", True))
    store_uncovered_text = bool(extraction_config.get("store_uncovered_text", True))
    fuzzy_threshold = float(locator_config.get("fuzzy_threshold", 0.86))
    price = _token_price(extraction_config, model_settings, batch_mode=batch_mode)

    client: ChatClient
    if mock:
        client = MockChatClient()
    else:
        client = OpenAIChatClient(
            api_key_env=str(model_settings["api_key_env"]),
            base_url=model_settings.get("base_url"),
            json_response_format=bool(model_settings.get("json_response_format", False)),
        )

    if batch or batch_id:
        if mock:
            raise ValueError("Batch extraction is only supported for live OpenAI-compatible clients.")
        if not isinstance(client, OpenAIChatClient):
            raise ValueError("Batch extraction requires OpenAIChatClient.")
        batch_report = _run_batch_extraction(
            store=store,
            config=config,
            storage_config=storage_config,
            run_id=run_id,
            document=document,
            windows=windows,
            client=client,
            model_name=model_name,
            temperature=temperature,
            retry_policy=retry_policy,
            min_spans_per_window=min_spans_per_window,
            max_spans_per_window=max_spans_per_window,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            target_span_chars_min=target_span_chars_min,
            target_span_chars_max=target_span_chars_max,
            store_located_text=store_located_text,
            store_uncovered_text=store_uncovered_text,
            fuzzy_threshold=fuzzy_threshold,
            price=price,
            batch_id=batch_id,
            batch_wait=batch_wait,
            batch_poll_interval_seconds=batch_poll_interval_seconds,
            batch_timeout_seconds=batch_timeout_seconds,
            batch_completion_window=batch_completion_window,
            progress_callback=progress_callback,
        )
        return batch_report

    store.delete_spans_for_windows(window.window_id for window in windows)

    if progress_callback is not None:
        progress_callback(
            "planned",
            {
                "document_id": document.document_id,
                "model": model_name,
                "mock": mock,
                "total_windows": len(windows),
                "window_offset": offset,
                "window_ids": selected_window_ids,
                "window_ranges": window_ranges or [],
            },
        )

    results: list[ExtractionResult] = []
    for window_index, window in enumerate(windows, start=1):
        window_started_at = time.perf_counter()
        if progress_callback is not None:
            progress_callback(
                "window_start",
                {
                    "window_index": window_index,
                    "total_windows": len(windows),
                    "window_id": window.window_id,
                    "chapter_id": window.chapter_id,
                    "char_count": window.char_count,
                },
            )
        window_results = extract_window(
            window,
            client=client,
            model=model_name,
            temperature=temperature,
            retry_policy=retry_policy,
            min_spans_per_window=min_spans_per_window,
            max_spans_per_window=max_spans_per_window,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            target_span_chars_min=target_span_chars_min,
            target_span_chars_max=target_span_chars_max,
            store_located_text=store_located_text,
            fuzzy_threshold=fuzzy_threshold,
            token_price=price,
            progress_callback=progress_callback,
            progress_payload={
                "window_index": window_index,
                "total_windows": len(windows),
                "window_id": window.window_id,
            },
        )
        window_located = sum(1 for result in window_results if result.status == "located")
        window_failed = len(window_results) - window_located
        window_cost = round(sum(result.cost.estimated_yuan for result in window_results), 6)
        uncovered_text = (
            build_uncovered_text(window, [result.span for result in window_results])
            if store_uncovered_text
            else ""
        )
        db_started_at = time.perf_counter()
        store.update_window_uncovered_text(window.window_id, uncovered_text)
        for result in window_results:
            store.upsert_span(result.span)
            if result.locator_result is not None:
                store.insert_locator_candidates(
                    span_id=result.span.span_id,
                    candidates=result.locator_result.candidates,
                )
            if result.status != "located":
                store.insert_extraction_failure(
                    window_id=window.window_id,
                    span_id=result.span.span_id,
                    stage="locator" if result.payload is not None else "extraction",
                    reason=result.failure_reason or "unknown failure",
                    attempts=result.attempts,
                    raw_output=result.raw_output,
                )
            results.append(result)
        if progress_callback is not None:
            progress_callback(
                "db_write_done",
                {
                    "window_index": window_index,
                    "total_windows": len(windows),
                    "window_id": window.window_id,
                    "elapsed_seconds": round(time.perf_counter() - db_started_at, 2),
                },
            )
        if progress_callback is not None:
            progress_callback(
                "window_done",
                {
                    "window_index": window_index,
                    "total_windows": len(windows),
                    "window_id": window.window_id,
                    "span_count": len(window_results),
                    "located_count": window_located,
                    "failed_count": window_failed,
                    "estimated_cost_yuan": window_cost,
                    "uncovered_chars": len(uncovered_text),
                    "elapsed_seconds": round(time.perf_counter() - window_started_at, 2),
                },
            )

    report = _build_report(
        run_id=run_id,
        document=document,
        windows=windows,
        results=results,
        model_name=model_name,
        mock=mock,
        sqlite_path=storage_config.sqlite_path,
    )
    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{run_id}_extraction_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.insert_extraction_report(run_id, document.document_id, report)
    if progress_callback is not None:
        progress_callback(
            "completed",
            {
                "report_path": str(report_path),
                "span_count": report["span_count"],
                "located_count": report["locator_success_count"],
                "failed_count": report["locator_failed_count"],
                "estimated_cost_yuan": report["estimated_cost_yuan"],
            },
        )
    return report


def list_extraction_windows(
    *,
    storage_config: AppConfig,
    document_id: str | None = None,
    only: str = "all",
    limit: int | None = None,
) -> dict[str, Any]:
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_extraction_tables()
    document = store.get_document(document_id)
    statuses = store.list_window_extraction_status(document.document_id)
    if only == "extracted":
        statuses = [status for status in statuses if status["status"] == "extracted"]
    elif only == "pending":
        statuses = [status for status in statuses if status["status"] == "pending"]
    elif only != "all":
        raise ValueError("--only must be one of: all, extracted, pending")
    if limit is not None:
        statuses = statuses[:limit]
    return {
        "document_id": document.document_id,
        "sqlite_path": str(storage_config.sqlite_path),
        "only": only,
        "window_count": len(statuses),
        "extracted_total": sum(1 for status in statuses if status["status"] == "extracted"),
        "pending_total": sum(1 for status in statuses if status["status"] == "pending"),
        "windows": statuses,
    }


def _run_batch_extraction(
    *,
    store: SQLiteStore,
    config: AppConfig,
    storage_config: AppConfig,
    run_id: str,
    document: Any,
    windows: list[CandidateWindow],
    client: OpenAIChatClient,
    model_name: str,
    temperature: float,
    retry_policy: RetryPolicy,
    min_spans_per_window: int,
    max_spans_per_window: int,
    anchor_min_chars: int,
    anchor_max_chars: int,
    target_span_chars_min: int,
    target_span_chars_max: int,
    store_located_text: bool,
    store_uncovered_text: bool,
    fuzzy_threshold: float,
    price: TokenPrice,
    batch_id: str | None,
    batch_wait: bool,
    batch_poll_interval_seconds: float,
    batch_timeout_seconds: float | None,
    batch_completion_window: str,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    current_batch_id = batch_id
    input_path: Path | None = None
    submission: BatchSubmission | None = None
    if current_batch_id is None:
        if len(windows) > BATCH_MAX_REQUESTS_PER_FILE:
            raise ValueError(
                "Batch input exceeds SiliconFlow's 5000-request file limit. "
                "Use --limit, --offset, --window-range, or --window-id to submit smaller batches."
            )
        input_path = runs_dir / f"{run_id}_extraction_batch_input.jsonl"
        _write_batch_input_file(
            input_path=input_path,
            windows=windows,
            model=model_name,
            temperature=temperature,
            min_spans_per_window=min_spans_per_window,
            max_spans_per_window=max_spans_per_window,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            target_span_chars_min=target_span_chars_min,
            target_span_chars_max=target_span_chars_max,
            json_response_format=client._json_response_format,
        )
        if progress_callback is not None:
            progress_callback(
                "batch_upload_start",
                {
                    "input_path": str(input_path),
                    "window_count": len(windows),
                    "model": model_name,
                },
            )
        submission = client.submit_chat_batch(
            input_path=input_path,
            completion_window=batch_completion_window,
            metadata={
                "run_id": run_id,
                "document_id": document.document_id,
                "purpose": "loreweaver_extract",
            },
        )
        current_batch_id = submission.batch_id
        if progress_callback is not None:
            progress_callback(
                "batch_submitted",
                {
                    "batch_id": submission.batch_id,
                    "input_file_id": submission.input_file_id,
                    "status": submission.status,
                    "window_count": len(windows),
                },
            )
        if not batch_wait:
            report = _build_batch_status_report(
                run_id=run_id,
                document=document,
                windows=windows,
                model_name=model_name,
                sqlite_path=storage_config.sqlite_path,
                batch_id=submission.batch_id,
                batch_status=submission.status,
                input_file_id=submission.input_file_id,
                output_file_id=submission.output_file_id,
                error_file_id=submission.error_file_id,
                request_counts=submission.request_counts,
                input_path=input_path,
            )
            _persist_extraction_report(
                config=config,
                store=store,
                run_id=run_id,
                document_id=document.document_id,
                report=report,
            )
            return report

    assert current_batch_id is not None
    status = _wait_for_batch_status(
        client=client,
        batch_id=current_batch_id,
        wait=batch_wait,
        poll_interval_seconds=batch_poll_interval_seconds,
        timeout_seconds=batch_timeout_seconds,
        progress_callback=progress_callback,
    )
    if status.status != "completed":
        report = _build_batch_status_report(
            run_id=run_id,
            document=document,
            windows=windows,
            model_name=model_name,
            sqlite_path=storage_config.sqlite_path,
            batch_id=status.batch_id,
            batch_status=status.status,
            input_file_id=status.input_file_id or (submission.input_file_id if submission else None),
            output_file_id=status.output_file_id,
            error_file_id=status.error_file_id,
            request_counts=status.request_counts,
            input_path=input_path,
        )
        _persist_extraction_report(
            config=config,
            store=store,
            run_id=run_id,
            document_id=document.document_id,
            report=report,
        )
        return report

    output_text = ""
    error_text = ""
    output_path: Path | None = None
    error_path: Path | None = None
    if status.output_file_id:
        output_text = client.download_file_text(status.output_file_id)
        output_path = runs_dir / f"{run_id}_extraction_batch_output.jsonl"
        output_path.write_text(output_text, encoding="utf-8")
    if status.error_file_id:
        error_text = client.download_file_text(status.error_file_id)
        error_path = runs_dir / f"{run_id}_extraction_batch_errors.jsonl"
        error_path.write_text(error_text, encoding="utf-8")
    if progress_callback is not None:
        progress_callback(
            "batch_downloaded",
            {
                "batch_id": status.batch_id,
                "output_file_id": status.output_file_id,
                "error_file_id": status.error_file_id,
                "output_path": str(output_path) if output_path else "",
                "error_path": str(error_path) if error_path else "",
            },
        )

    outputs = _parse_batch_output_lines(output_text) + _parse_batch_output_lines(error_text)
    results = _apply_batch_outputs(
        store=store,
        windows=windows,
        outputs=outputs,
        client=client,
        model=model_name,
        temperature=temperature,
        retry_policy=retry_policy,
        min_spans_per_window=min_spans_per_window,
        max_spans_per_window=max_spans_per_window,
        anchor_min_chars=anchor_min_chars,
        anchor_max_chars=anchor_max_chars,
        target_span_chars_min=target_span_chars_min,
        target_span_chars_max=target_span_chars_max,
        store_located_text=store_located_text,
        store_uncovered_text=store_uncovered_text,
        fuzzy_threshold=fuzzy_threshold,
        token_price=price,
        progress_callback=progress_callback,
    )
    report = _build_report(
        run_id=run_id,
        document=document,
        windows=[
            window
            for window in windows
            if any(result.span.window_id == window.window_id for result in results)
        ],
        results=results,
        model_name=model_name,
        mock=False,
        sqlite_path=storage_config.sqlite_path,
    )
    report.update(
        {
            "mode": "batch",
            "batch_id": status.batch_id,
            "batch_status": status.status,
            "input_file_id": status.input_file_id,
            "output_file_id": status.output_file_id,
            "error_file_id": status.error_file_id,
            "request_counts": status.request_counts,
            "batch_output_path": str(output_path) if output_path else None,
            "batch_error_path": str(error_path) if error_path else None,
        }
    )
    _persist_extraction_report(
        config=config,
        store=store,
        run_id=run_id,
        document_id=document.document_id,
        report=report,
    )
    if progress_callback is not None:
        progress_callback(
            "completed",
            {
                "report_path": report["report_path"],
                "span_count": report["span_count"],
                "located_count": report["locator_success_count"],
                "failed_count": report["locator_failed_count"],
                "estimated_cost_yuan": report["estimated_cost_yuan"],
            },
        )
    return report


def _wait_for_batch_status(
    *,
    client: OpenAIChatClient,
    batch_id: str,
    wait: bool,
    poll_interval_seconds: float,
    timeout_seconds: float | None,
    progress_callback: ProgressCallback | None,
) -> BatchStatus:
    started_at = time.perf_counter()
    terminal_statuses = {"completed", "failed", "expired", "cancelled", "cancelling"}
    while True:
        status = client.retrieve_chat_batch(batch_id)
        if progress_callback is not None:
            progress_callback(
                "batch_status",
                {
                    "batch_id": status.batch_id,
                    "status": status.status,
                    "request_counts": status.request_counts,
                    "output_file_id": status.output_file_id,
                    "error_file_id": status.error_file_id,
                },
            )
        if not wait or status.status in terminal_statuses:
            return status
        if timeout_seconds is not None and time.perf_counter() - started_at >= timeout_seconds:
            return status
        time.sleep(max(1.0, poll_interval_seconds))


def _write_batch_input_file(
    *,
    input_path: Path,
    windows: list[CandidateWindow],
    model: str,
    temperature: float,
    min_spans_per_window: int,
    max_spans_per_window: int,
    anchor_min_chars: int,
    anchor_max_chars: int,
    target_span_chars_min: int,
    target_span_chars_max: int,
    json_response_format: bool,
) -> None:
    with input_path.open("w", encoding="utf-8") as file_obj:
        for window in windows:
            line = _build_batch_request_line(
                window=window,
                model=model,
                temperature=temperature,
                min_spans_per_window=min_spans_per_window,
                max_spans_per_window=max_spans_per_window,
                anchor_min_chars=anchor_min_chars,
                anchor_max_chars=anchor_max_chars,
                target_span_chars_min=target_span_chars_min,
                target_span_chars_max=target_span_chars_max,
                json_response_format=json_response_format,
            )
            file_obj.write(json.dumps(line, ensure_ascii=False) + "\n")


def _build_batch_request_line(
    *,
    window: CandidateWindow,
    model: str,
    temperature: float,
    min_spans_per_window: int,
    max_spans_per_window: int,
    anchor_min_chars: int,
    anchor_max_chars: int,
    target_span_chars_min: int,
    target_span_chars_max: int,
    json_response_format: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": build_extraction_messages(
            window,
            min_spans_per_window=min_spans_per_window,
            max_spans_per_window=max_spans_per_window,
            target_span_chars_min=target_span_chars_min,
            target_span_chars_max=target_span_chars_max,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
        ),
        "temperature": temperature,
    }
    if json_response_format:
        body["response_format"] = {"type": "json_object"}
    return {
        "custom_id": window.window_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _apply_batch_outputs(
    *,
    store: SQLiteStore,
    windows: list[CandidateWindow],
    outputs: list[BatchOutput],
    client: ChatClient,
    model: str,
    temperature: float,
    retry_policy: RetryPolicy,
    min_spans_per_window: int,
    max_spans_per_window: int,
    anchor_min_chars: int,
    anchor_max_chars: int,
    target_span_chars_min: int,
    target_span_chars_max: int,
    store_located_text: bool,
    store_uncovered_text: bool,
    fuzzy_threshold: float,
    token_price: TokenPrice,
    progress_callback: ProgressCallback | None,
) -> list[ExtractionResult]:
    by_id = {window.window_id: window for window in windows}
    output_by_id = {output.custom_id: output for output in outputs}
    selected_window_ids = [window_id for window_id in output_by_id if window_id in by_id]
    store.delete_spans_for_windows(selected_window_ids)
    results: list[ExtractionResult] = []
    total_windows = len(selected_window_ids)
    for window_index, window_id in enumerate(selected_window_ids, start=1):
        window = by_id[window_id]
        output = output_by_id[window_id]
        if progress_callback is not None:
            progress_callback(
                "window_start",
                {
                    "window_index": window_index,
                    "total_windows": total_windows,
                    "window_id": window.window_id,
                    "chapter_id": window.chapter_id,
                    "char_count": window.char_count,
                },
            )
        if output.error:
            window_results = [
                _failed_window_result(
                    window,
                    reason=output.error,
                    raw_output=output.raw_output,
                    attempts=1,
                    usage_total=output.usage,
                    token_price=token_price,
                )
            ]
        else:
            try:
                window_results = _results_from_raw_window_output(
                    window,
                    raw_output=output.raw_output or "",
                    attempts=1,
                    usage=output.usage,
                    token_price=token_price,
                    max_spans_per_window=max_spans_per_window,
                    anchor_min_chars=anchor_min_chars,
                    anchor_max_chars=anchor_max_chars,
                    target_span_chars_min=target_span_chars_min,
                    target_span_chars_max=target_span_chars_max,
                    store_located_text=store_located_text,
                    fuzzy_threshold=fuzzy_threshold,
                    raise_parse_errors=True,
                )
            except WindowPayloadParseError as error:
                if progress_callback is not None:
                    progress_callback(
                        "batch_window_retry",
                        {
                            "window_index": window_index,
                            "total_windows": total_windows,
                            "window_id": window.window_id,
                            "reason": str(error),
                        },
                    )
                window_results = extract_window(
                    window,
                    client=client,
                    model=model,
                    temperature=temperature,
                    retry_policy=retry_policy,
                    min_spans_per_window=min_spans_per_window,
                    max_spans_per_window=max_spans_per_window,
                    anchor_min_chars=anchor_min_chars,
                    anchor_max_chars=anchor_max_chars,
                    target_span_chars_min=target_span_chars_min,
                    target_span_chars_max=target_span_chars_max,
                    store_located_text=store_located_text,
                    fuzzy_threshold=fuzzy_threshold,
                    token_price=token_price,
                    progress_callback=progress_callback,
                    progress_payload={
                        "window_index": window_index,
                        "total_windows": total_windows,
                        "window_id": window.window_id,
                    },
                )
                window_results = _add_batch_usage_to_first_result(
                    window_results,
                    batch_usage=output.usage,
                    token_price=token_price,
                )
        window_located = sum(1 for result in window_results if result.status == "located")
        uncovered_text = (
            build_uncovered_text(window, [result.span for result in window_results])
            if store_uncovered_text
            else ""
        )
        store.update_window_uncovered_text(window.window_id, uncovered_text)
        for result in window_results:
            store.upsert_span(result.span)
            if result.locator_result is not None:
                store.insert_locator_candidates(
                    span_id=result.span.span_id,
                    candidates=result.locator_result.candidates,
                )
            if result.status != "located":
                store.insert_extraction_failure(
                    window_id=window.window_id,
                    span_id=result.span.span_id,
                    stage="locator" if result.payload is not None else "extraction",
                    reason=result.failure_reason or "unknown failure",
                    attempts=result.attempts,
                    raw_output=result.raw_output,
                )
            results.append(result)
        if progress_callback is not None:
            progress_callback(
                "window_done",
                {
                    "window_index": window_index,
                    "total_windows": total_windows,
                    "window_id": window.window_id,
                    "span_count": len(window_results),
                    "located_count": window_located,
                    "failed_count": len(window_results) - window_located,
                    "estimated_cost_yuan": round(
                        sum(result.cost.estimated_yuan for result in window_results),
                        6,
                    ),
                    "uncovered_chars": len(uncovered_text),
                    "elapsed_seconds": 0,
                },
            )
    return results


def _results_from_raw_window_output(
    window: CandidateWindow,
    *,
    raw_output: str,
    attempts: int,
    usage: dict[str, int],
    token_price: TokenPrice,
    max_spans_per_window: int,
    anchor_min_chars: int,
    anchor_max_chars: int,
    target_span_chars_min: int,
    target_span_chars_max: int,
    store_located_text: bool,
    fuzzy_threshold: float,
    raise_parse_errors: bool = False,
) -> list[ExtractionResult]:
    messages = build_extraction_messages(
        window,
        min_spans_per_window=1,
        max_spans_per_window=max_spans_per_window,
        target_span_chars_min=target_span_chars_min,
        target_span_chars_max=target_span_chars_max,
        anchor_min_chars=anchor_min_chars,
        anchor_max_chars=anchor_max_chars,
    )
    usage_total = _usage_or_estimate(usage, messages, raw_output)
    try:
        payload = _parse_payload(raw_output)
        results = _results_from_window_payload(
            window,
            payload=payload,
            raw_output=raw_output,
            attempts=attempts,
            usage_total=usage_total,
            token_price=token_price,
            max_spans_per_window=max_spans_per_window,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            target_span_chars_min=target_span_chars_min,
            target_span_chars_max=target_span_chars_max,
            store_located_text=store_located_text,
            fuzzy_threshold=fuzzy_threshold,
        )
        if results:
            return results
    except (ValidationError, json.JSONDecodeError, ValueError) as error:
        if raise_parse_errors:
            raise WindowPayloadParseError(str(error)) from error
        return [
            _failed_window_result(
                window,
                reason=str(error),
                raw_output=raw_output,
                attempts=attempts,
                usage_total=usage_total,
                token_price=token_price,
            )
        ]
    return [
        _failed_window_result(
            window,
            reason="window extraction produced no valid spans",
            raw_output=raw_output,
            attempts=attempts,
            usage_total=usage_total,
            token_price=token_price,
        )
    ]


def _add_batch_usage_to_first_result(
    results: list[ExtractionResult],
    *,
    batch_usage: dict[str, int],
    token_price: TokenPrice,
) -> list[ExtractionResult]:
    if not results:
        return results
    merged_usage = _merge_usage(results[0].usage, batch_usage)
    return [
        replace(results[0], usage=merged_usage, cost=estimate_cost(merged_usage, token_price)),
        *results[1:],
    ]


def _parse_batch_output_lines(text: str) -> list[BatchOutput]:
    outputs: list[BatchOutput] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        outputs.append(_batch_output_from_line(json.loads(line)))
    return outputs


def _batch_output_from_line(data: dict[str, Any]) -> BatchOutput:
    custom_id = str(data.get("custom_id") or "")
    error = data.get("error")
    response = data.get("response") or {}
    if error:
        return BatchOutput(
            custom_id=custom_id,
            raw_output=json.dumps(data, ensure_ascii=False),
            usage=_empty_usage(),
            error=_format_batch_error(error),
        )
    if int(response.get("status_code") or 0) >= 400:
        return BatchOutput(
            custom_id=custom_id,
            raw_output=json.dumps(data, ensure_ascii=False),
            usage=_empty_usage(),
            error=f"batch response status_code={response.get('status_code')}",
        )
    body = response.get("body") or {}
    return BatchOutput(
        custom_id=custom_id,
        raw_output=_batch_body_content(body),
        usage=_batch_body_usage(body),
        error=None,
    )


def _batch_body_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        return "".join(str(item.get("text", item)) for item in content)
    return str(content)


def _batch_body_usage(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _format_batch_error(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error.get("type")
        return str(message or error)
    return str(error)


def _build_batch_status_report(
    *,
    run_id: str,
    document: Any,
    windows: list[CandidateWindow],
    model_name: str,
    sqlite_path: Path,
    batch_id: str,
    batch_status: str,
    input_file_id: str | None,
    output_file_id: str | None,
    error_file_id: str | None,
    request_counts: dict[str, int],
    input_path: Path | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": "batch",
        "document": {
            "document_id": document.document_id,
            "title": document.title,
            "normalized_path": document.normalized_path,
        },
        "model": model_name,
        "mock": False,
        "window_count": len(windows),
        "span_count": 0,
        "extraction_success_count": 0,
        "extraction_failed_count": 0,
        "locator_success_count": 0,
        "locator_failed_count": 0,
        "structured_success_rate": 0,
        "locator_success_rate": 0,
        "usage": _empty_usage(),
        "estimated_cost_yuan": 0,
        "sqlite_path": str(sqlite_path),
        "failed_windows": [],
        "spans_preview": [],
        "batch_id": batch_id,
        "batch_status": batch_status,
        "input_file_id": input_file_id,
        "output_file_id": output_file_id,
        "error_file_id": error_file_id,
        "request_counts": request_counts,
        "batch_input_path": str(input_path) if input_path else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _persist_extraction_report(
    *,
    config: AppConfig,
    store: SQLiteStore,
    run_id: str,
    document_id: str,
    report: dict[str, Any],
) -> None:
    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{run_id}_extraction_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.insert_extraction_report(run_id, document_id, report)


def _normalize_window_ids(
    *,
    window_id: str | None,
    window_ids: list[str] | None,
) -> list[str]:
    raw_values: list[str] = []
    if window_id:
        raw_values.append(window_id)
    raw_values.extend(window_ids or [])
    normalized: list[str] = []
    for raw_value in raw_values:
        for item in raw_value.split(","):
            value = item.strip()
            if value and value not in normalized:
                normalized.append(value)
    return normalized


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _uploaded_file_id(uploaded: Any) -> str:
    file_id = getattr(uploaded, "id", None)
    if file_id:
        return str(file_id)
    data = getattr(uploaded, "data", None)
    if isinstance(data, dict) and data.get("id"):
        return str(data["id"])
    if isinstance(uploaded, dict):
        if uploaded.get("id"):
            return str(uploaded["id"])
        nested_data = uploaded.get("data")
        if isinstance(nested_data, dict) and nested_data.get("id"):
            return str(nested_data["id"])
    raise ValueError("Provider file upload response did not include a file id.")


def _request_counts_dict(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        items = value.items()
    else:
        items = (
            (name, getattr(value, name, 0))
            for name in ("total", "completed", "failed")
            if hasattr(value, name)
        )
    counts: dict[str, int] = {}
    for key, raw_count in items:
        try:
            counts[str(key)] = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
    return counts


def _select_windows(
    windows: list[CandidateWindow],
    *,
    window_ids: list[str],
    window_ranges: list[str],
) -> list[CandidateWindow]:
    by_id = {window.window_id: window for window in windows}
    selected_ids: list[str] = []
    missing_ids = [window_id for window_id in window_ids if window_id not in by_id]
    if missing_ids:
        raise ValueError(f"Candidate window not found: {', '.join(missing_ids)}")
    selected_ids.extend(window_id for window_id in window_ids if window_id not in selected_ids)

    for range_text in window_ranges:
        start, end = _parse_window_range(range_text, total_windows=len(windows))
        for window in windows[start - 1 : end]:
            if window.window_id not in selected_ids:
                selected_ids.append(window.window_id)
    if not selected_ids:
        raise ValueError("No candidate windows selected.")
    return [by_id[window_id] for window_id in selected_ids]


def _parse_window_range(range_text: str, *, total_windows: int) -> tuple[int, int]:
    value = range_text.strip()
    separator = "-" if "-" in value else ":"
    if separator not in value:
        index = int(value)
        start = index
        end = index
    else:
        raw_start, raw_end = value.split(separator, 1)
        start = int(raw_start.strip())
        end = int(raw_end.strip())
    if start < 1 or end < start or end > total_windows:
        raise ValueError(
            f"Invalid window range {range_text!r}; expected 1-based range within 1-{total_windows}."
        )
    return start, end


def extract_window(
    window: CandidateWindow,
    *,
    client: ChatClient,
    model: str,
    temperature: float,
    retry_policy: RetryPolicy,
    min_spans_per_window: int,
    max_spans_per_window: int,
    anchor_min_chars: int,
    anchor_max_chars: int,
    target_span_chars_min: int,
    target_span_chars_max: int,
    store_located_text: bool,
    fuzzy_threshold: float,
    token_price: TokenPrice,
    progress_callback: ProgressCallback | None = None,
    progress_payload: dict[str, Any] | None = None,
) -> list[ExtractionResult]:
    messages = build_extraction_messages(
        window,
        min_spans_per_window=min_spans_per_window,
        max_spans_per_window=max_spans_per_window,
        target_span_chars_min=target_span_chars_min,
        target_span_chars_max=target_span_chars_max,
        anchor_min_chars=anchor_min_chars,
        anchor_max_chars=anchor_max_chars,
    )
    raw_output: str | None = None
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    last_reason: str | None = None

    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            api_started_at = time.perf_counter()
            if progress_callback is not None:
                progress_callback(
                    "api_start",
                    {
                        **(progress_payload or {}),
                        "attempt": attempt,
                    },
                )
            raw_output, usage = client.complete_json(
                messages=messages,
                model=model,
                temperature=temperature,
            )
            if progress_callback is not None:
                progress_callback(
                    "api_done",
                    {
                        **(progress_payload or {}),
                        "attempt": attempt,
                        "elapsed_seconds": round(time.perf_counter() - api_started_at, 2),
                        "input_tokens": int(usage.get("input_tokens", 0)),
                        "output_tokens": int(usage.get("output_tokens", 0)),
                    },
                )
            parse_locate_started_at = time.perf_counter()
            usage_total = _merge_usage(usage_total, _usage_or_estimate(usage, messages, raw_output))
            payload = _parse_payload(raw_output)
            results = _results_from_window_payload(
                window,
                payload=payload,
                raw_output=raw_output,
                attempts=attempt,
                usage_total=usage_total,
                token_price=token_price,
                max_spans_per_window=max_spans_per_window,
                anchor_min_chars=anchor_min_chars,
                anchor_max_chars=anchor_max_chars,
                target_span_chars_min=target_span_chars_min,
                target_span_chars_max=target_span_chars_max,
                store_located_text=store_located_text,
                fuzzy_threshold=fuzzy_threshold,
            )
            if progress_callback is not None:
                progress_callback(
                    "parse_locate_done",
                    {
                        **(progress_payload or {}),
                        "attempt": attempt,
                        "elapsed_seconds": round(time.perf_counter() - parse_locate_started_at, 2),
                        "span_count": len(results),
                        "located_count": sum(1 for result in results if result.status == "located"),
                        "failed_count": sum(1 for result in results if result.status != "located"),
                    },
                )
            if any(result.status == "located" for result in results):
                return results
            last_reason = "; ".join(
                result.failure_reason or "unknown span failure" for result in results[:3]
            )
        except (ValidationError, json.JSONDecodeError, ValueError) as error:
            last_reason = str(error)

    failed_payload = _best_effort_payload(raw_output)
    if failed_payload is not None:
        return _results_from_window_payload(
            window,
            payload=failed_payload,
            raw_output=raw_output,
            attempts=retry_policy.max_attempts,
            usage_total=usage_total,
            token_price=token_price,
            max_spans_per_window=max_spans_per_window,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            target_span_chars_min=target_span_chars_min,
            target_span_chars_max=target_span_chars_max,
            store_located_text=store_located_text,
            fuzzy_threshold=fuzzy_threshold,
            force_failure_reason=last_reason,
        )
    return [
        _failed_window_result(
            window,
            reason=last_reason or "window extraction produced no valid spans",
            raw_output=raw_output,
            attempts=retry_policy.max_attempts,
            usage_total=usage_total,
            token_price=token_price,
        )
    ]


def estimate_tokens(text: str) -> int:
    """Cheap provider-agnostic token estimate used when API usage is unavailable."""
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    non_cjk_text = "".join(" " if "\u4e00" <= char <= "\u9fff" else char for char in text)
    word_like = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", non_cjk_text)
    return max(1, cjk_chars + len(word_like))


def estimate_cost(usage: dict[str, int], price: TokenPrice) -> CostEstimate:
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    estimated_yuan = (
        input_tokens / 1000 * price.input_yuan_per_1k
        + output_tokens / 1000 * price.output_yuan_per_1k
    )
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_yuan_per_1k=price.input_yuan_per_1k,
        output_yuan_per_1k=price.output_yuan_per_1k,
        estimated_yuan=round(estimated_yuan, 6),
    )


def build_uncovered_text(window: CandidateWindow, spans: list[Span]) -> str:
    """Return merged window fragments not covered by any located span."""
    intervals = _merged_located_intervals(window, spans)
    cursor = window.window_start
    blocks: list[str] = []
    for start, end in intervals:
        if cursor < start:
            blocks.append(_format_uncovered_block(window, cursor, start))
        cursor = max(cursor, end)
    if cursor < window.window_end:
        blocks.append(_format_uncovered_block(window, cursor, window.window_end))
    return "\n\n".join(block for block in blocks if block)


def _merged_located_intervals(window: CandidateWindow, spans: list[Span]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for span in spans:
        if span.locator_status != "located":
            continue
        if span.span_start_idx is None or span.span_end_idx is None:
            continue
        start = max(window.window_start, span.span_start_idx)
        end = min(window.window_end, span.span_end_idx)
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


def _format_uncovered_block(window: CandidateWindow, start: int, end: int) -> str:
    text = window.text[start - window.window_start : end - window.window_start].strip()
    if not text:
        return ""
    return f"[{start}, {end})\n{text}"


def _model_settings(models_config: AppConfig) -> dict[str, Any]:
    model_values = models_config.values
    extraction_model = model_values.get("models", {}).get("extraction", {})
    provider_name = extraction_model.get("provider", "openai")
    provider = model_values.get("providers", {}).get(provider_name, {})
    return {
        "provider": provider_name,
        "model": extraction_model.get("name", "gpt-4o-mini"),
        "temperature": extraction_model.get("temperature", 0),
        "api_key_env": provider.get("api_key_env", "OPENAI_API_KEY"),
        "base_url": provider.get("base_url"),
        "input_yuan_per_1k": extraction_model.get("input_yuan_per_1k", 0.0),
        "output_yuan_per_1k": extraction_model.get("output_yuan_per_1k", 0.0),
        "json_response_format": extraction_model.get("json_response_format", False),
        "batch_model": extraction_model.get("batch_name"),
        "batch_input_yuan_per_1k": extraction_model.get("batch_input_yuan_per_1k", 0.0),
        "batch_output_yuan_per_1k": extraction_model.get("batch_output_yuan_per_1k", 0.0),
    }


def _token_price(
    extraction_config: dict[str, Any],
    model_settings: dict[str, Any],
    *,
    batch_mode: bool,
) -> TokenPrice:
    input_key = "batch_input_yuan_per_1k" if batch_mode else "input_yuan_per_1k"
    output_key = "batch_output_yuan_per_1k" if batch_mode else "output_yuan_per_1k"
    return TokenPrice(
        input_yuan_per_1k=float(
            extraction_config.get(
                input_key,
                model_settings.get(input_key, extraction_config.get("input_yuan_per_1k", 0.0)),
            )
        ),
        output_yuan_per_1k=float(
            extraction_config.get(
                output_key,
                model_settings.get(output_key, extraction_config.get("output_yuan_per_1k", 0.0)),
            )
        ),
    )


def _parse_payload(raw_output: str) -> WindowExtractionPayload:
    raw_output = raw_output.strip()
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_output, re.S)
        if match is None:
            raise
        data = json.loads(match.group(0))
    return WindowExtractionPayload.model_validate(data)


def _best_effort_payload(raw_output: str | None) -> WindowExtractionPayload | None:
    if not raw_output:
        return None
    try:
        return _parse_payload(raw_output)
    except (ValidationError, json.JSONDecodeError, ValueError):
        return None


def _results_from_window_payload(
    window: CandidateWindow,
    *,
    payload: WindowExtractionPayload,
    raw_output: str | None,
    attempts: int,
    usage_total: dict[str, int],
    token_price: TokenPrice,
    max_spans_per_window: int,
    anchor_min_chars: int,
    anchor_max_chars: int,
    target_span_chars_min: int,
    target_span_chars_max: int,
    store_located_text: bool,
    fuzzy_threshold: float,
    force_failure_reason: str | None = None,
) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    for span_index, span_payload in enumerate(payload.spans[:max_spans_per_window], start=1):
        locator_result: LocatorResult | None = None
        status = "located"
        failure_reason = force_failure_reason
        start_anchor = _anchor_for_location(
            span_payload.start_anchor_quote,
            role="start",
            max_chars=anchor_max_chars,
        )
        end_anchor = _anchor_for_location(
            span_payload.end_anchor_quote,
            role="end",
            max_chars=anchor_max_chars,
        )
        if failure_reason is None:
            start_ok, start_reason = anchor_constraints_ok(
                start_anchor,
                min_chars=anchor_min_chars,
                max_chars=anchor_max_chars,
                label="start_anchor_quote",
            )
            end_ok, end_reason = anchor_constraints_ok(
                end_anchor,
                min_chars=anchor_min_chars,
                max_chars=anchor_max_chars,
                label="end_anchor_quote",
            )
            if not start_ok or not end_ok:
                status = "failed"
                failure_reason = start_reason or end_reason
            else:
                locator_result = locate_span_anchors(
                    window,
                    start_anchor_quote=start_anchor,
                    end_anchor_quote=end_anchor,
                    fuzzy_threshold=fuzzy_threshold,
                    min_span_chars=target_span_chars_min,
                    max_span_chars=target_span_chars_max,
                )
                if locator_result.status != "located":
                    status = "failed"
                    failure_reason = locator_result.failure_reason
        else:
            status = "failed"

        span = _span_from_payload(
            window,
            span_index=span_index,
            payload=span_payload,
            locator_result=locator_result,
            status=status,
            store_located_text=store_located_text,
        )
        result_usage = usage_total if span_index == 1 else _empty_usage()
        results.append(
            ExtractionResult(
                span=span,
                payload=span_payload,
                locator_result=locator_result,
                status=status,
                failure_reason=failure_reason,
                raw_output=raw_output,
                attempts=attempts,
                usage=result_usage,
                cost=estimate_cost(result_usage, token_price),
            )
        )
    return results


def _anchor_for_location(anchor: str, *, role: str, max_chars: int) -> str:
    """Trim overlong LLM anchors while preserving the side closest to the span boundary."""
    cleaned = anchor.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if role == "end":
        return cleaned[-max_chars:]
    return cleaned[:max_chars]


def _failed_window_result(
    window: CandidateWindow,
    *,
    reason: str,
    raw_output: str | None,
    attempts: int,
    usage_total: dict[str, int],
    token_price: TokenPrice,
) -> ExtractionResult:
    span = _span_from_payload(
        window,
        span_index=1,
        payload=None,
        locator_result=None,
        status="failed",
        store_located_text=False,
    )
    return ExtractionResult(
        span=span,
        payload=None,
        locator_result=None,
        status="failed",
        failure_reason=reason,
        raw_output=raw_output,
        attempts=attempts,
        usage=usage_total,
        cost=estimate_cost(usage_total, token_price),
    )


def _span_from_payload(
    window: CandidateWindow,
    *,
    span_index: int,
    payload: SpanCandidatePayload | None,
    locator_result: LocatorResult | None,
    status: str,
    store_located_text: bool,
) -> Span:
    span_start = locator_result.start_idx if locator_result else None
    span_end = locator_result.end_idx if locator_result else None
    locator_confidence = locator_result.confidence if locator_result else 0.0
    locator_status = locator_result.status if locator_result else "failed"
    if status == "failed" and locator_status == "located":
        locator_status = "failed"
    located_text = ""
    if status == "located" and store_located_text and span_start is not None and span_end is not None:
        located_text = window.text[span_start - window.window_start : span_end - window.window_start]
    return Span(
        span_id=f"span_{window.window_id}_{span_index:03d}",
        document_id=window.document_id,
        chapter_id=window.chapter_id,
        window_id=window.window_id,
        span_index_in_window=span_index,
        window_start=window.window_start,
        window_end=window.window_end,
        micro_topic=payload.micro_topic if payload else "",
        span_type=payload.span_type if payload else "other",
        micro_summary=payload.micro_summary if payload else "",
        entities=payload.entities if payload else [],
        topics=payload.topics if payload else [],
        salience_score=payload.salience_score if payload else 0.0,
        start_anchor_quote=payload.start_anchor_quote if payload else "",
        end_anchor_quote=payload.end_anchor_quote if payload else "",
        key_quote=payload.key_quote if payload else "",
        overlap_reason=payload.overlap_reason if payload else "",
        span_start_idx=span_start if status == "located" else None,
        span_end_idx=span_end if status == "located" else None,
        located_text=located_text,
        locator_confidence=locator_confidence if status == "located" else 0.0,
        locator_status=locator_status if status == "located" else "failed",
        created_at=datetime.now(timezone.utc),
    )


def _usage_or_estimate(
    usage: dict[str, int],
    messages: list[dict[str, str]],
    raw_output: str,
) -> dict[str, int]:
    if usage.get("input_tokens") or usage.get("output_tokens"):
        return usage
    input_tokens = estimate_tokens("\n".join(message["content"] for message in messages))
    output_tokens = estimate_tokens(raw_output)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": int(left.get("input_tokens", 0)) + int(right.get("input_tokens", 0)),
        "output_tokens": int(left.get("output_tokens", 0)) + int(right.get("output_tokens", 0)),
        "total_tokens": int(left.get("total_tokens", 0)) + int(right.get("total_tokens", 0)),
    }


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _build_report(
    *,
    run_id: str,
    document: Any,
    windows: list[CandidateWindow],
    results: list[ExtractionResult],
    model_name: str,
    mock: bool,
    sqlite_path: Path,
) -> dict[str, Any]:
    located = [result for result in results if result.status == "located"]
    failed = [result for result in results if result.status != "located"]
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_cost = 0.0
    for result in results:
        total_usage = _merge_usage(total_usage, result.usage)
        total_cost += result.cost.estimated_yuan
    return {
        "run_id": run_id,
        "document": {
            "document_id": document.document_id,
            "title": document.title,
            "normalized_path": document.normalized_path,
        },
        "model": model_name,
        "mock": mock,
        "window_count": len(windows),
        "span_count": len(results),
        "extraction_success_count": len(located),
        "extraction_failed_count": len(failed),
        "locator_success_count": len(located),
        "locator_failed_count": len(failed),
        "structured_success_rate": round(len(located) / len(results), 4) if results else 0,
        "locator_success_rate": round(len(located) / len(results), 4) if results else 0,
        "usage": total_usage,
        "estimated_cost_yuan": round(total_cost, 6),
        "sqlite_path": str(sqlite_path),
        "failed_windows": [
            {
                "window_id": result.span.window_id,
                "span_id": result.span.span_id,
                "micro_topic": result.span.micro_topic,
                "reason": result.failure_reason,
            }
            for result in failed[:50]
        ],
        "spans_preview": [
            {
                "span_id": result.span.span_id,
                "window_id": result.span.window_id,
                "span_index_in_window": result.span.span_index_in_window,
                "micro_topic": result.span.micro_topic,
                "span_type": result.span.span_type,
                "span_start_idx": result.span.span_start_idx,
                "span_end_idx": result.span.span_end_idx,
                "locator_confidence": result.span.locator_confidence,
                "micro_summary": result.span.micro_summary,
                "start_anchor_quote": result.span.start_anchor_quote,
                "end_anchor_quote": result.span.end_anchor_quote,
                "key_quote": result.span.key_quote,
                "located_text": result.span.located_text,
            }
            for result in results[:5]
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _asdict_cost(cost: CostEstimate) -> dict[str, Any]:
    return asdict(cost)

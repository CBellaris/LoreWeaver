"""Top-level extraction runners for M1.3.

Call chain:
- extract_document_windows -> extract_window for live per-window extraction.
- extract_document_windows -> _run_batch_extraction for batch submit/download/apply/retry.
- extract_window -> payload parsing -> locator -> Span construction.
"""

from __future__ import annotations

import time
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.extraction.batch import (
    BATCH_LIVE_RETRY_THRESHOLD_DEFAULT,
    BATCH_MAX_REQUESTS_PER_FILE,
    BATCH_RETRY_MAX_ROUNDS_DEFAULT,
    _apply_batch_outputs,
    _batch_output_from_line,
    _build_batch_request_line,
    _run_batch_extraction,
)
from loreweaver.extraction.mock_client import MockChatClient
from loreweaver.extraction.persistence import _persist_window_results
from loreweaver.extraction.reports import _build_report, _persist_extraction_report
from loreweaver.extraction.retry import RetryPolicy
from loreweaver.extraction.selection import _normalize_window_ids, _select_windows
from loreweaver.extraction.settings import _model_settings, _token_price
from loreweaver.extraction.types import (
    BatchApplyOutcome,
    BatchOutput,
    ChatClient,
    CostEstimate,
    ExtractionResult,
    TokenPrice,
    WindowPayloadParseError,
)
from loreweaver.extraction.usage import estimate_cost, estimate_tokens
from loreweaver.extraction.window_processing import (
    _results_from_raw_window_output,
    build_uncovered_text,
    extract_window,
)
from loreweaver.model_services import ModelServiceFactory
from loreweaver.progress import ProgressReporter
from loreweaver.storage.sqlite_store import SQLiteStore


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
    progress: ProgressReporter | None = None,
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
    model_name = str(model_settings["model"])
    batch_mode = batch or bool(batch_id)
    if batch_mode:
        model_name = str(
            batch_model
            or model_settings.get("batch_model")
            or "deepseek-ai/DeepSeek-V3.1-Terminus"
        )
    temperature = float(model_settings["temperature"])
    retry_policy = RetryPolicy(max_retries=int(extraction_config.get("max_retries", 2)))
    batch_live_retry_threshold = max(
        0,
        int(
            extraction_config.get(
                "batch_live_retry_threshold",
                BATCH_LIVE_RETRY_THRESHOLD_DEFAULT,
            )
        ),
    )
    batch_retry_max_rounds = max(
        0,
        int(
            extraction_config.get(
                "batch_retry_max_rounds",
                BATCH_RETRY_MAX_ROUNDS_DEFAULT,
            )
        ),
    )
    anchor_min_chars = int(extraction_config.get("anchor_min_chars", 8))
    anchor_max_chars = int(extraction_config.get("anchor_max_chars", 80))
    store_located_text = bool(extraction_config.get("store_located_text", True))
    store_uncovered_text = bool(extraction_config.get("store_uncovered_text", True))
    fuzzy_threshold = float(locator_config.get("fuzzy_threshold", 0.86))
    price = _token_price(model_settings, batch_mode=batch_mode)

    factory = ModelServiceFactory.from_configs(models_config=models_config)
    client: ChatClient
    if mock:
        client = MockChatClient()
    else:
        client = factory.chat("extraction")

    if batch or batch_id:
        if mock:
            raise ValueError(
                "Batch extraction is only supported for live OpenAI-compatible clients."
            )
        if not all(
            hasattr(client, name)
            for name in ("submit_chat_batch", "retrieve_chat_batch", "download_file_text")
        ):
            raise ValueError("Batch extraction requires a batch-capable chat client.")
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
            json_response_format=bool(model_settings.get("json_response_format", False)),
            retry_policy=retry_policy,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            store_located_text=store_located_text,
            store_uncovered_text=store_uncovered_text,
            fuzzy_threshold=fuzzy_threshold,
            price=price,
            batch_id=batch_id,
            batch_wait=batch_wait,
            batch_poll_interval_seconds=batch_poll_interval_seconds,
            batch_timeout_seconds=batch_timeout_seconds,
            batch_completion_window=batch_completion_window,
            batch_live_retry_threshold=batch_live_retry_threshold,
            batch_retry_max_rounds=batch_retry_max_rounds,
            progress=progress,
        )
        return batch_report

    store.delete_spans_for_windows(window.window_id for window in windows)

    if progress is not None:
        progress.emit(
            "planned",
            stage="extract.plan",
            label=f"Plan extraction for {len(windows)} windows",
            current=0,
            total=len(windows),
            unit="windows",
            detail={
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
        if progress is not None:
            progress.emit(
                "window_start",
                stage="extract.window",
                label=f"Extract {window.window_id}",
                current=window_index - 1,
                total=len(windows),
                unit="windows",
                detail={
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
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            store_located_text=store_located_text,
            fuzzy_threshold=fuzzy_threshold,
            token_price=price,
            progress=progress,
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
        _persist_window_results(
            store=store,
            window=window,
            results=window_results,
            store_uncovered_text=store_uncovered_text,
            uncovered_text=uncovered_text,
        )
        results.extend(window_results)
        if progress is not None:
            progress.emit(
                "db_write_done",
                stage="extract.db",
                label=f"Persist {window.window_id}",
                current=window_index,
                total=len(windows),
                unit="windows",
                detail={
                    "window_index": window_index,
                    "total_windows": len(windows),
                    "window_id": window.window_id,
                    "elapsed_seconds": round(time.perf_counter() - db_started_at, 2),
                },
            )
        if progress is not None:
            progress.emit(
                "window_done",
                stage="extract.window",
                label=f"Completed {window.window_id}",
                current=window_index,
                total=len(windows),
                unit="windows",
                detail={
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
    _persist_extraction_report(
        config=config,
        store=store,
        run_id=run_id,
        document_id=document.document_id,
        report=report,
    )
    if progress is not None:
        progress.emit(
            "completed",
            stage="extract.completed",
            label="Extraction completed",
            current=len(windows),
            total=len(windows),
            unit="windows",
            status="completed",
            detail={
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


__all__ = [
    "BATCH_LIVE_RETRY_THRESHOLD_DEFAULT",
    "BATCH_MAX_REQUESTS_PER_FILE",
    "BATCH_RETRY_MAX_ROUNDS_DEFAULT",
    "BatchApplyOutcome",
    "BatchOutput",
    "ChatClient",
    "CostEstimate",
    "ExtractionResult",
    "MockChatClient",
    "TokenPrice",
    "WindowPayloadParseError",
    "_apply_batch_outputs",
    "_batch_output_from_line",
    "_build_batch_request_line",
    "_results_from_raw_window_output",
    "build_uncovered_text",
    "estimate_cost",
    "estimate_tokens",
    "extract_document_windows",
    "extract_window",
    "list_extraction_windows",
]

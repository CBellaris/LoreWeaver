"""Batch extraction submission, application, and retry orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.extraction.persistence import _persist_window_results
from loreweaver.extraction.prompts import build_extraction_messages
from loreweaver.extraction.reports import (
    _build_batch_status_report,
    _build_report,
    _persist_extraction_report,
)
from loreweaver.extraction.retry import RetryPolicy
from loreweaver.extraction.types import (
    BatchApplyOutcome,
    BatchOutput,
    ChatClient,
    ExtractionResult,
    TokenPrice,
    WindowPayloadParseError,
)
from loreweaver.extraction.usage import _empty_usage, _merge_usage
from loreweaver.extraction.window_processing import (
    _add_usage_to_first_result,
    _failed_window_result,
    _has_retryable_locator_failures,
    _results_from_raw_window_output,
    extract_window,
)
from loreweaver.model_services.types import BatchStatus, BatchSubmission
from loreweaver.models.window import CandidateWindow
from loreweaver.progress import ProgressReporter
from loreweaver.storage.sqlite_store import SQLiteStore


BATCH_MAX_REQUESTS_PER_FILE = 5000
BATCH_LIVE_RETRY_THRESHOLD_DEFAULT = 8
BATCH_RETRY_MAX_ROUNDS_DEFAULT = 3


def _run_batch_extraction(
    *,
    store: SQLiteStore,
    config: AppConfig,
    storage_config: AppConfig,
    run_id: str,
    document: Any,
    windows: list[CandidateWindow],
    client: Any,
    model_name: str,
    temperature: float,
    json_response_format: bool,
    retry_policy: RetryPolicy,
    anchor_min_chars: int,
    anchor_max_chars: int,
    store_located_text: bool,
    store_uncovered_text: bool,
    fuzzy_threshold: float,
    price: TokenPrice,
    batch_id: str | None,
    batch_wait: bool,
    batch_poll_interval_seconds: float,
    batch_timeout_seconds: float | None,
    batch_completion_window: str,
    batch_live_retry_threshold: int,
    batch_retry_max_rounds: int,
    progress: ProgressReporter | None,
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
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            json_response_format=json_response_format,
        )
        if progress is not None:
            progress.emit(
                "batch_upload_start",
                stage="extract.batch.upload",
                label="Upload batch input",
                current=0,
                total=len(windows),
                unit="windows",
                detail={
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
        if progress is not None:
            progress.emit(
                "batch_submitted",
                stage="extract.batch.submit",
                label=f"Batch submitted {submission.batch_id}",
                status=submission.status,
                detail={
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
        progress=progress,
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
            input_file_id=(
                status.input_file_id or (submission.input_file_id if submission else None)
            ),
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
    if progress is not None:
        progress.emit(
            "batch_downloaded",
            stage="extract.batch.download",
            label=f"Downloaded batch {status.batch_id}",
            detail={
                "batch_id": status.batch_id,
                "output_file_id": status.output_file_id,
                "error_file_id": status.error_file_id,
                "output_path": str(output_path) if output_path else "",
                "error_path": str(error_path) if error_path else "",
            },
        )

    outputs = _parse_batch_output_lines(output_text) + _parse_batch_output_lines(error_text)
    apply_outcome = _apply_batch_outputs(
        store=store,
        windows=windows,
        outputs=outputs,
        anchor_min_chars=anchor_min_chars,
        anchor_max_chars=anchor_max_chars,
        store_located_text=store_located_text,
        store_uncovered_text=store_uncovered_text,
        fuzzy_threshold=fuzzy_threshold,
        token_price=price,
        prior_usage_by_window={},
        progress=progress,
    )
    results = list(apply_outcome.results)
    pending_windows = list(apply_outcome.retry_windows)
    pending_usage_by_window = dict(apply_outcome.retry_usage_by_window)
    pending_reasons = dict(apply_outcome.retry_reasons)
    retry_batch_summaries: list[dict[str, Any]] = []
    retry_round = 0
    retry_batch_blocked = False

    while (
        pending_windows
        and len(pending_windows) > batch_live_retry_threshold
        and retry_round < batch_retry_max_rounds
    ):
        retry_round += 1
        retry_status, retry_outputs, retry_summary = _run_batch_retry_round(
            config=config,
            document=document,
            run_id=run_id,
            retry_round=retry_round,
            windows=pending_windows,
            client=client,
                model_name=model_name,
                temperature=temperature,
                json_response_format=json_response_format,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            batch_poll_interval_seconds=batch_poll_interval_seconds,
            batch_timeout_seconds=batch_timeout_seconds,
            batch_completion_window=batch_completion_window,
            progress=progress,
        )
        retry_batch_summaries.append(retry_summary)
        if retry_status.status != "completed":
            pending_reasons = {
                window.window_id: (
                    f"retry batch {retry_status.batch_id} status={retry_status.status}"
                )
                for window in pending_windows
            }
            retry_batch_blocked = True
            break
        apply_outcome = _apply_batch_outputs(
            store=store,
            windows=pending_windows,
            outputs=retry_outputs,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            store_located_text=store_located_text,
            store_uncovered_text=store_uncovered_text,
            fuzzy_threshold=fuzzy_threshold,
            token_price=price,
            prior_usage_by_window=pending_usage_by_window,
            progress=progress,
        )
        results.extend(apply_outcome.results)
        pending_windows = list(apply_outcome.retry_windows)
        pending_usage_by_window = dict(apply_outcome.retry_usage_by_window)
        pending_reasons = dict(apply_outcome.retry_reasons)

    if pending_windows and len(pending_windows) <= batch_live_retry_threshold:
        live_results = _retry_windows_live(
            store=store,
            windows=pending_windows,
            retry_reasons=pending_reasons,
            prior_usage_by_window=pending_usage_by_window,
            client=client,
            model=model_name,
            temperature=temperature,
            retry_policy=retry_policy,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            store_located_text=store_located_text,
            store_uncovered_text=store_uncovered_text,
            fuzzy_threshold=fuzzy_threshold,
            token_price=price,
            progress=progress,
        )
        results.extend(live_results)
        pending_windows = []
        pending_usage_by_window = {}
        pending_reasons = {}

    if pending_windows:
        deferred_results = _record_deferred_batch_retry_windows(
            store=store,
            windows=pending_windows,
            retry_reasons=pending_reasons,
            prior_usage_by_window=pending_usage_by_window,
            retry_batch_blocked=retry_batch_blocked,
            retry_round=retry_round,
            retry_max_rounds=batch_retry_max_rounds,
            token_price=price,
            store_uncovered_text=store_uncovered_text,
        )
        results.extend(deferred_results)
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
            "batch_retry_rounds": retry_batch_summaries,
            "batch_live_retry_threshold": batch_live_retry_threshold,
            "batch_retry_max_rounds": batch_retry_max_rounds,
            "batch_retry_deferred_window_count": len(pending_windows),
        }
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
            label="Batch extraction completed",
            status="completed",
            detail={
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
    client: Any,
    batch_id: str,
    wait: bool,
    poll_interval_seconds: float,
    timeout_seconds: float | None,
    progress: ProgressReporter | None,
) -> BatchStatus:
    started_at = time.perf_counter()
    terminal_statuses = {"completed", "failed", "expired", "cancelled", "cancelling"}
    while True:
        status = client.retrieve_chat_batch(batch_id)
        if progress is not None:
            progress.emit(
                "batch_status",
                stage="extract.batch.wait",
                label=f"Batch status {status.status}",
                status=status.status,
                detail={
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


def _run_batch_retry_round(
    *,
    config: AppConfig,
    document: Any,
    run_id: str,
    retry_round: int,
    windows: list[CandidateWindow],
    client: Any,
    model_name: str,
    temperature: float,
    json_response_format: bool,
    anchor_min_chars: int,
    anchor_max_chars: int,
    batch_poll_interval_seconds: float,
    batch_timeout_seconds: float | None,
    batch_completion_window: str,
    progress: ProgressReporter | None,
) -> tuple[BatchStatus, list[BatchOutput], dict[str, Any]]:
    runs_dir = config.data_dir / "runs"
    retry_tag = f"retry{retry_round:02d}"
    input_path = runs_dir / f"{run_id}_extraction_batch_{retry_tag}_input.jsonl"
    _write_batch_input_file(
        input_path=input_path,
        windows=windows,
        model=model_name,
        temperature=temperature,
        anchor_min_chars=anchor_min_chars,
        anchor_max_chars=anchor_max_chars,
        json_response_format=json_response_format,
    )
    if progress is not None:
        progress.emit(
            "batch_upload_start",
            stage="extract.batch.upload",
            label=f"Upload retry batch {retry_round}",
            current=0,
            total=len(windows),
            unit="windows",
            detail={
                "retry_round": retry_round,
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
            "purpose": "loreweaver_extract_retry",
            "retry_round": str(retry_round),
        },
    )
    if progress is not None:
        progress.emit(
            "batch_submitted",
            stage="extract.batch.submit",
            label=f"Retry batch submitted {submission.batch_id}",
            status=submission.status,
            detail={
                "retry_round": retry_round,
                "batch_id": submission.batch_id,
                "input_file_id": submission.input_file_id,
                "status": submission.status,
                "window_count": len(windows),
            },
        )
    status = _wait_for_batch_status(
        client=client,
        batch_id=submission.batch_id,
        wait=True,
        poll_interval_seconds=batch_poll_interval_seconds,
        timeout_seconds=batch_timeout_seconds,
        progress=progress,
    )

    output_text = ""
    error_text = ""
    output_path: Path | None = None
    error_path: Path | None = None
    if status.status == "completed":
        if status.output_file_id:
            output_text = client.download_file_text(status.output_file_id)
            output_path = runs_dir / f"{run_id}_extraction_batch_{retry_tag}_output.jsonl"
            output_path.write_text(output_text, encoding="utf-8")
        if status.error_file_id:
            error_text = client.download_file_text(status.error_file_id)
            error_path = runs_dir / f"{run_id}_extraction_batch_{retry_tag}_errors.jsonl"
            error_path.write_text(error_text, encoding="utf-8")
        if progress is not None:
            progress.emit(
                "batch_downloaded",
                stage="extract.batch.download",
                label=f"Downloaded retry batch {status.batch_id}",
                detail={
                    "retry_round": retry_round,
                    "batch_id": status.batch_id,
                    "output_file_id": status.output_file_id,
                    "error_file_id": status.error_file_id,
                    "output_path": str(output_path) if output_path else "",
                    "error_path": str(error_path) if error_path else "",
                },
            )

    outputs = _parse_batch_output_lines(output_text) + _parse_batch_output_lines(error_text)
    summary = {
        "retry_round": retry_round,
        "batch_id": status.batch_id,
        "batch_status": status.status,
        "input_file_id": status.input_file_id or submission.input_file_id,
        "output_file_id": status.output_file_id,
        "error_file_id": status.error_file_id,
        "request_counts": status.request_counts,
        "batch_input_path": str(input_path),
        "batch_output_path": str(output_path) if output_path else None,
        "batch_error_path": str(error_path) if error_path else None,
        "window_count": len(windows),
    }
    return status, outputs, summary


def _write_batch_input_file(
    *,
    input_path: Path,
    windows: list[CandidateWindow],
    model: str,
    temperature: float,
    anchor_min_chars: int,
    anchor_max_chars: int,
    json_response_format: bool,
) -> None:
    with input_path.open("w", encoding="utf-8") as file_obj:
        for window in windows:
            line = _build_batch_request_line(
                window=window,
                model=model,
                temperature=temperature,
                anchor_min_chars=anchor_min_chars,
                anchor_max_chars=anchor_max_chars,
                json_response_format=json_response_format,
            )
            file_obj.write(json.dumps(line, ensure_ascii=False) + "\n")


def _build_batch_request_line(
    *,
    window: CandidateWindow,
    model: str,
    temperature: float,
    anchor_min_chars: int,
    anchor_max_chars: int,
    json_response_format: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": build_extraction_messages(
            window,
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
    anchor_min_chars: int,
    anchor_max_chars: int,
    store_located_text: bool,
    store_uncovered_text: bool,
    fuzzy_threshold: float,
    token_price: TokenPrice,
    prior_usage_by_window: dict[str, dict[str, int]],
    progress: ProgressReporter | None,
) -> BatchApplyOutcome:
    by_id = {window.window_id: window for window in windows}
    output_by_id = {output.custom_id: output for output in outputs}
    selected_window_ids = [window.window_id for window in windows]
    store.delete_spans_for_windows(selected_window_ids)
    results: list[ExtractionResult] = []
    retry_windows: list[CandidateWindow] = []
    retry_usage_by_window: dict[str, dict[str, int]] = {}
    retry_reasons: dict[str, str] = {}
    total_windows = len(windows)
    for window_index, window_id in enumerate(selected_window_ids, start=1):
        window = by_id[window_id]
        output = output_by_id.get(window_id)
        if progress is not None:
            progress.emit(
                "window_start",
                stage="extract.window",
                label=f"Apply batch output {window.window_id}",
                current=window_index - 1,
                total=total_windows,
                unit="windows",
                detail={
                    "window_index": window_index,
                    "total_windows": total_windows,
                    "window_id": window.window_id,
                    "chapter_id": window.chapter_id,
                    "char_count": window.char_count,
                },
            )
        if output is None:
            retry_windows.append(window)
            retry_usage_by_window[window.window_id] = dict(
                prior_usage_by_window.get(window.window_id, _empty_usage())
            )
            retry_reasons[window.window_id] = "batch output missing"
            _emit_batch_retry_queued(
                progress=progress,
                window=window,
                window_index=window_index,
                total_windows=total_windows,
                reason=retry_reasons[window.window_id],
            )
            continue

        combined_usage = _merge_usage(
            prior_usage_by_window.get(window.window_id, _empty_usage()),
            output.usage,
        )
        retry_reason: str | None = None
        window_results: list[ExtractionResult] | None = None
        if output.error:
            retry_reason = output.error
        else:
            try:
                window_results = _results_from_raw_window_output(
                    window,
                    raw_output=output.raw_output or "",
                    attempts=1,
                    usage=output.usage,
                    token_price=token_price,
                    anchor_min_chars=anchor_min_chars,
                    anchor_max_chars=anchor_max_chars,
                    store_located_text=store_located_text,
                    fuzzy_threshold=fuzzy_threshold,
                    raise_parse_errors=True,
                )
            except WindowPayloadParseError as error:
                retry_reason = str(error)
            if window_results is not None and _has_retryable_locator_failures(window_results):
                retry_reason = "; ".join(
                    result.failure_reason or "unknown span failure"
                    for result in window_results
                    if result.status != "located"
                )
        if retry_reason is not None:
            retry_windows.append(window)
            retry_usage_by_window[window.window_id] = combined_usage
            retry_reasons[window.window_id] = retry_reason
            _emit_batch_retry_queued(
                progress=progress,
                window=window,
                window_index=window_index,
                total_windows=total_windows,
                reason=retry_reason,
            )
            continue

        assert window_results is not None
        prior_usage = prior_usage_by_window.get(window.window_id)
        if prior_usage:
            window_results = _add_usage_to_first_result(
                window_results,
                usage=prior_usage,
                token_price=token_price,
            )
        window_located = sum(1 for result in window_results if result.status == "located")
        uncovered_text = _persist_window_results(
            store=store,
            window=window,
            results=window_results,
            store_uncovered_text=store_uncovered_text,
        )
        results.extend(window_results)
        if progress is not None:
            progress.emit(
                "window_done",
                stage="extract.window",
                label=f"Completed {window.window_id}",
                current=window_index,
                total=total_windows,
                unit="windows",
                detail={
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
    return BatchApplyOutcome(
        results=results,
        retry_windows=retry_windows,
        retry_usage_by_window=retry_usage_by_window,
        retry_reasons=retry_reasons,
    )


def _emit_batch_retry_queued(
    *,
    progress: ProgressReporter | None,
    window: CandidateWindow,
    window_index: int,
    total_windows: int,
    reason: str,
) -> None:
    if progress is None:
        return
    progress.emit(
        "batch_window_retry",
        stage="extract.batch.retry",
        label=f"Queue retry {window.window_id}",
        current=window_index,
        total=total_windows,
        unit="windows",
        message=reason.splitlines()[0],
        detail={
            "window_index": window_index,
            "total_windows": total_windows,
            "window_id": window.window_id,
            "reason": reason,
        },
    )


def _retry_windows_live(
    *,
    store: SQLiteStore,
    windows: list[CandidateWindow],
    retry_reasons: dict[str, str],
    prior_usage_by_window: dict[str, dict[str, int]],
    client: ChatClient,
    model: str,
    temperature: float,
    retry_policy: RetryPolicy,
    anchor_min_chars: int,
    anchor_max_chars: int,
    store_located_text: bool,
    store_uncovered_text: bool,
    fuzzy_threshold: float,
    token_price: TokenPrice,
    progress: ProgressReporter | None,
) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    total_windows = len(windows)
    store.delete_spans_for_windows(window.window_id for window in windows)
    for window_index, window in enumerate(windows, start=1):
        if progress is not None:
            progress.emit(
                "window_start",
                stage="extract.window",
                label=f"Live retry {window.window_id}",
                current=window_index - 1,
                total=total_windows,
                unit="windows",
                message=retry_reasons.get(window.window_id),
                detail={
                    "window_index": window_index,
                    "total_windows": total_windows,
                    "window_id": window.window_id,
                    "chapter_id": window.chapter_id,
                    "char_count": window.char_count,
                    "retry_reason": retry_reasons.get(window.window_id),
                },
            )
        window_results = extract_window(
            window,
            client=client,
            model=model,
            temperature=temperature,
            retry_policy=retry_policy,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
            store_located_text=store_located_text,
            fuzzy_threshold=fuzzy_threshold,
            token_price=token_price,
            progress=progress,
            progress_payload={
                "window_index": window_index,
                "total_windows": total_windows,
                "window_id": window.window_id,
            },
        )
        prior_usage = prior_usage_by_window.get(window.window_id)
        if prior_usage:
            window_results = _add_usage_to_first_result(
                window_results,
                usage=prior_usage,
                token_price=token_price,
            )
        window_located = sum(1 for result in window_results if result.status == "located")
        uncovered_text = _persist_window_results(
            store=store,
            window=window,
            results=window_results,
            store_uncovered_text=store_uncovered_text,
        )
        results.extend(window_results)
        if progress is not None:
            progress.emit(
                "window_done",
                stage="extract.window",
                label=f"Completed {window.window_id}",
                current=window_index,
                total=total_windows,
                unit="windows",
                detail={
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


def _record_deferred_batch_retry_windows(
    *,
    store: SQLiteStore,
    windows: list[CandidateWindow],
    retry_reasons: dict[str, str],
    prior_usage_by_window: dict[str, dict[str, int]],
    retry_batch_blocked: bool,
    retry_round: int,
    retry_max_rounds: int,
    token_price: TokenPrice,
    store_uncovered_text: bool,
) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    store.delete_spans_for_windows(window.window_id for window in windows)
    for window in windows:
        base_reason = retry_reasons.get(window.window_id, "batch retry still required")
        if retry_batch_blocked:
            reason = f"{base_reason}; deferred before live fallback"
        else:
            reason = (
                f"{base_reason}; deferred after {retry_round}/{retry_max_rounds} "
                "batch retry rounds"
            )
        usage_total = prior_usage_by_window.get(window.window_id, _empty_usage())
        window_results = [
            _failed_window_result(
                window,
                reason=reason,
                raw_output=None,
                attempts=max(1, retry_round),
                usage_total=usage_total,
                token_price=token_price,
            )
        ]
        uncovered_text = window.text if store_uncovered_text else ""
        _persist_window_results(
            store=store,
            window=window,
            results=window_results,
            store_uncovered_text=store_uncovered_text,
            uncovered_text=uncovered_text,
        )
        results.extend(window_results)
    return results

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

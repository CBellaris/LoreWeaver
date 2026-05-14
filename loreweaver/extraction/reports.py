"""Extraction report persistence and summary helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.extraction.types import ExtractionResult
from loreweaver.extraction.usage import _empty_usage, _merge_usage
from loreweaver.models.window import CandidateWindow
from loreweaver.storage.sqlite_store import SQLiteStore


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
                "summary": result.span.summary,
                "reason": result.failure_reason,
            }
            for result in failed[:50]
        ],
        "spans_preview": [
            {
                "span_id": result.span.span_id,
                "window_id": result.span.window_id,
                "span_index_in_window": result.span.span_index_in_window,
                "span_type": result.span.span_type,
                "span_start_idx": result.span.span_start_idx,
                "span_end_idx": result.span.span_end_idx,
                "locator_confidence": result.span.locator_confidence,
                "summary": result.span.summary,
                "start_anchor_quote": result.span.start_anchor_quote,
                "end_anchor_quote": result.span.end_anchor_quote,
                "key_quote": result.span.key_quote,
                "located_text": result.span.located_text,
            }
            for result in results[:5]
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

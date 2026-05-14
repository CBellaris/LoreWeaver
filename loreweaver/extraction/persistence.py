"""Persistence helpers shared by live and batch extraction flows."""

from __future__ import annotations

from loreweaver.extraction.types import ExtractionResult
from loreweaver.extraction.window_processing import build_uncovered_text
from loreweaver.models.window import CandidateWindow
from loreweaver.storage.sqlite_store import SQLiteStore


def _persist_window_results(
    *,
    store: SQLiteStore,
    window: CandidateWindow,
    results: list[ExtractionResult],
    store_uncovered_text: bool,
    uncovered_text: str | None = None,
) -> str:
    if uncovered_text is None:
        uncovered_text = (
            build_uncovered_text(window, [result.span for result in results])
            if store_uncovered_text
            else ""
        )
    store.update_window_uncovered_text(window.window_id, uncovered_text)
    for result in results:
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
    return uncovered_text

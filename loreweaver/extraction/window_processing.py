"""Single-window extraction, parsing, locating, and span construction."""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover - exercised in minimal bootstrap envs.
    ValidationError = ValueError  # type: ignore[misc, assignment]

from loreweaver.extraction.locator import (
    LocatorResult,
    anchor_constraints_ok,
    locate_span_anchors,
)
from loreweaver.extraction.prompts import build_extraction_messages
from loreweaver.extraction.retry import RetryPolicy
from loreweaver.extraction.schemas import SpanCandidatePayload, WindowExtractionPayload
from loreweaver.extraction.types import (
    ChatClient,
    ExtractionResult,
    TokenPrice,
    WindowPayloadParseError,
)
from loreweaver.extraction.usage import (
    _empty_usage,
    _merge_usage,
    _usage_or_estimate,
    estimate_cost,
)
from loreweaver.model_services import ChatRequest
from loreweaver.models.span import Span
from loreweaver.models.window import CandidateWindow
from loreweaver.progress import ProgressReporter


SMALL_UNCOVERED_GAP_MAX_CHARS = 120


def _results_from_raw_window_output(
    window: CandidateWindow,
    *,
    raw_output: str,
    attempts: int,
    usage: dict[str, int],
    token_price: TokenPrice,
    anchor_min_chars: int,
    anchor_max_chars: int,
    store_located_text: bool,
    fuzzy_threshold: float,
    raise_parse_errors: bool = False,
) -> list[ExtractionResult]:
    messages = build_extraction_messages(
        window,
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
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
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


def _add_usage_to_first_result(
    results: list[ExtractionResult],
    *,
    usage: dict[str, int],
    token_price: TokenPrice,
) -> list[ExtractionResult]:
    if not results:
        return results
    merged_usage = _merge_usage(results[0].usage, usage)
    return [
        replace(results[0], usage=merged_usage, cost=estimate_cost(merged_usage, token_price)),
        *results[1:],
    ]

def extract_window(
    window: CandidateWindow,
    *,
    client: ChatClient,
    model: str,
    temperature: float,
    retry_policy: RetryPolicy,
    anchor_min_chars: int,
    anchor_max_chars: int,
    store_located_text: bool,
    fuzzy_threshold: float,
    token_price: TokenPrice,
    progress: ProgressReporter | None = None,
    progress_payload: dict[str, Any] | None = None,
) -> list[ExtractionResult]:
    messages = build_extraction_messages(
        window,
        anchor_min_chars=anchor_min_chars,
        anchor_max_chars=anchor_max_chars,
    )
    raw_output: str | None = None
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    last_reason: str | None = None
    last_results: list[ExtractionResult] | None = None

    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            api_started_at = time.perf_counter()
            if progress is not None:
                progress.emit(
                    "api_start",
                    stage="extract.api",
                    label=f"Call extraction model for {window.window_id}",
                    current=(progress_payload or {}).get("window_index"),
                    total=(progress_payload or {}).get("total_windows"),
                    unit="windows",
                    detail={
                        **(progress_payload or {}),
                        "attempt": attempt,
                    },
                )
            result = client.complete(
                ChatRequest(
                    messages=messages,
                    temperature=temperature,
                    extra={"model": model},
                )
            )
            raw_output = result.content
            usage = result.usage
            if progress is not None:
                progress.emit(
                    "api_done",
                    stage="extract.api",
                    label=f"Model response for {window.window_id}",
                    current=(progress_payload or {}).get("window_index"),
                    total=(progress_payload or {}).get("total_windows"),
                    unit="windows",
                    detail={
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
                anchor_min_chars=anchor_min_chars,
                anchor_max_chars=anchor_max_chars,
                store_located_text=store_located_text,
                fuzzy_threshold=fuzzy_threshold,
            )
            if progress is not None:
                progress.emit(
                    "parse_locate_done",
                    stage="extract.locate",
                    label=f"Parsed and located {window.window_id}",
                    current=(progress_payload or {}).get("window_index"),
                    total=(progress_payload or {}).get("total_windows"),
                    unit="windows",
                    detail={
                        **(progress_payload or {}),
                        "attempt": attempt,
                        "elapsed_seconds": round(time.perf_counter() - parse_locate_started_at, 2),
                        "span_count": len(results),
                        "located_count": sum(1 for result in results if result.status == "located"),
                        "failed_count": sum(1 for result in results if result.status != "located"),
                    },
                )
            last_results = results
            if not _has_retryable_locator_failures(results):
                return results
            last_reason = "; ".join(
                result.failure_reason or "unknown span failure" for result in results[:3]
            )
        except (ValidationError, json.JSONDecodeError, ValueError) as error:
            last_reason = str(error)

    if last_results is not None:
        return last_results

    failed_payload = _best_effort_payload(raw_output)
    if failed_payload is not None:
        return _results_from_window_payload(
            window,
            payload=failed_payload,
            raw_output=raw_output,
            attempts=retry_policy.max_attempts,
            usage_total=usage_total,
            token_price=token_price,
            anchor_min_chars=anchor_min_chars,
            anchor_max_chars=anchor_max_chars,
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


def _has_retryable_locator_failures(results: list[ExtractionResult]) -> bool:
    return any(
        result.payload is not None
        and result.status != "located"
        and _is_retryable_locator_failure(result.failure_reason)
        for result in results
    )


def _is_retryable_locator_failure(reason: str | None) -> bool:
    return reason in {
        "start anchor not found",
        "end anchor not found",
        "no ordered start/end anchor pair found",
    }

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
    anchor_min_chars: int,
    anchor_max_chars: int,
    store_located_text: bool,
    fuzzy_threshold: float,
    force_failure_reason: str | None = None,
) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    for span_index, span_payload in enumerate(payload.spans, start=1):
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
    return absorb_small_uncovered_gaps(
        window,
        results,
        max_gap_chars=SMALL_UNCOVERED_GAP_MAX_CHARS,
        store_located_text=store_located_text,
    )


def absorb_small_uncovered_gaps(
    window: CandidateWindow,
    results: list[ExtractionResult],
    *,
    max_gap_chars: int = SMALL_UNCOVERED_GAP_MAX_CHARS,
    store_located_text: bool = True,
) -> list[ExtractionResult]:
    """Attach small uncovered fragments to adjacent located spans.

    Base spans are expected to cover the full window. The locator can still leave
    tiny gaps around headings, quotes, short reactions, or paragraph seams. Those
    fragments are not useful standalone evidence, so we absorb them into the
    nearest located span while leaving larger gaps visible for inspection.
    """
    if max_gap_chars <= 0 or not results:
        return results

    located_positions = [
        index
        for index, result in enumerate(results)
        if result.status == "located"
        and result.span.span_start_idx is not None
        and result.span.span_end_idx is not None
    ]
    if not located_positions:
        return results

    bounds: dict[int, list[int]] = {
        index: [results[index].span.span_start_idx, results[index].span.span_end_idx]  # type: ignore[list-item]
        for index in located_positions
    }
    ordered = sorted(located_positions, key=lambda index: (bounds[index][0], bounds[index][1]))

    _absorb_gap_into_neighbor(
        window,
        bounds,
        gap_start=window.window_start,
        gap_end=bounds[ordered[0]][0],
        neighbor=ordered[0],
        side="before",
        max_gap_chars=max_gap_chars,
    )
    for left, right in zip(ordered, ordered[1:]):
        gap_start = bounds[left][1]
        gap_end = bounds[right][0]
        if gap_start >= gap_end:
            continue
        neighbor, side = _choose_gap_neighbor(window, gap_start, gap_end, left, right)
        _absorb_gap_into_neighbor(
            window,
            bounds,
            gap_start=gap_start,
            gap_end=gap_end,
            neighbor=neighbor,
            side=side,
            max_gap_chars=max_gap_chars,
        )
    _absorb_gap_into_neighbor(
        window,
        bounds,
        gap_start=bounds[ordered[-1]][1],
        gap_end=window.window_end,
        neighbor=ordered[-1],
        side="after",
        max_gap_chars=max_gap_chars,
    )

    updated: list[ExtractionResult] = []
    for index, result in enumerate(results):
        if index not in bounds:
            updated.append(result)
            continue
        start, end = bounds[index]
        span = result.span
        if start == span.span_start_idx and end == span.span_end_idx:
            updated.append(result)
            continue
        located_text = span.located_text
        if store_located_text:
            located_text = window.text[start - window.window_start : end - window.window_start]
        updated.append(
            replace(
                result,
                span=replace(
                    span,
                    span_start_idx=start,
                    span_end_idx=end,
                    located_text=located_text,
                ),
            )
        )
    return updated


def _absorb_gap_into_neighbor(
    window: CandidateWindow,
    bounds: dict[int, list[int]],
    *,
    gap_start: int,
    gap_end: int,
    neighbor: int,
    side: str,
    max_gap_chars: int,
) -> None:
    if gap_start >= gap_end or neighbor not in bounds:
        return
    if not _is_absorbable_gap(window, gap_start, gap_end, max_gap_chars=max_gap_chars):
        return
    if side == "before":
        bounds[neighbor][0] = min(bounds[neighbor][0], gap_start)
    else:
        bounds[neighbor][1] = max(bounds[neighbor][1], gap_end)


def _choose_gap_neighbor(
    window: CandidateWindow,
    gap_start: int,
    gap_end: int,
    left: int,
    right: int,
) -> tuple[int, str]:
    gap_text = _window_slice(window, gap_start, gap_end).strip()
    if gap_text.startswith(("”", "’", "」", "』", "）", ")", "，", "。", "！", "？", "、", "；", "：")):
        return left, "after"
    if gap_text.endswith(("“", "‘", "「", "『", "（", "(")):
        return right, "before"
    if "\n" in gap_text and len(gap_text) <= 30:
        return right, "before"
    return left, "after"


def _is_absorbable_gap(
    window: CandidateWindow,
    start: int,
    end: int,
    *,
    max_gap_chars: int,
) -> bool:
    text = _window_slice(window, start, end).strip()
    return not text or len(text) <= max_gap_chars


def _window_slice(window: CandidateWindow, start: int, end: int) -> str:
    return window.text[start - window.window_start : end - window.window_start]


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
    if (
        status == "located"
        and store_located_text
        and span_start is not None
        and span_end is not None
    ):
        located_text = window.text[
            span_start - window.window_start : span_end - window.window_start
        ]
    return Span(
        span_id=f"span_{window.window_id}_{span_index:03d}",
        document_id=window.document_id,
        chapter_id=window.chapter_id,
        window_id=window.window_id,
        span_index_in_window=span_index,
        window_start=window.window_start,
        window_end=window.window_end,
        span_type=payload.span_type if payload else "other",
        summary=payload.summary if payload else "",
        entities=payload.entities if payload else [],
        salience_score=payload.salience_score if payload else 0.0,
        start_anchor_quote=payload.start_anchor_quote if payload else "",
        end_anchor_quote=payload.end_anchor_quote if payload else "",
        key_quote=payload.key_quote if payload else "",
        span_start_idx=span_start if status == "located" else None,
        span_end_idx=span_end if status == "located" else None,
        located_text=located_text,
        locator_confidence=locator_confidence if status == "located" else 0.0,
        locator_status=locator_status if status == "located" else "failed",
        created_at=datetime.now(timezone.utc),
    )

"""Quote-to-coordinate location utilities for M1.3."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata

from loreweaver.models.window import CandidateWindow


@dataclass(frozen=True)
class LocatorCandidate:
    start_idx: int
    end_idx: int
    confidence: float
    strategy: str
    matched_text: str


@dataclass(frozen=True)
class LocatorResult:
    status: str
    start_idx: int | None
    end_idx: int | None
    confidence: float
    strategy: str
    candidates: list[LocatorCandidate]
    failure_reason: str | None = None


def locate_quote(
    window: CandidateWindow,
    quote_text: str,
    *,
    fuzzy_threshold: float = 0.86,
) -> LocatorResult:
    """Locate an LLM quote inside a candidate window and map it to global coordinates."""
    quote = quote_text.strip()
    if not quote:
        return LocatorResult("failed", None, None, 0.0, "none", [], "quote is empty")

    candidates: list[LocatorCandidate] = []
    candidates.extend(_exact_candidates(window, quote))
    if candidates:
        return _select_best_candidate(window, candidates)

    candidates.extend(_normalized_candidates(window, quote))
    if candidates:
        return _select_best_candidate(window, candidates)

    candidates.extend(_fuzzy_candidates(window, quote, fuzzy_threshold=fuzzy_threshold))
    if candidates:
        return _select_best_candidate(window, candidates)

    return LocatorResult(
        status="failed",
        start_idx=None,
        end_idx=None,
        confidence=0.0,
        strategy="none",
        candidates=[],
        failure_reason="quote was not found in the candidate window",
    )


def anchor_constraints_ok(
    anchor: str,
    *,
    min_chars: int = 8,
    max_chars: int = 80,
    label: str = "anchor",
) -> tuple[bool, str | None]:
    anchor_length = len(anchor.strip())
    if anchor_length < min_chars:
        return False, f"{label} is shorter than {min_chars} chars"
    if anchor_length > max_chars:
        return False, f"{label} is longer than {max_chars} chars"
    return True, None


def locate_span_anchors(
    window: CandidateWindow,
    *,
    start_anchor_quote: str,
    end_anchor_quote: str,
    fuzzy_threshold: float = 0.86,
) -> LocatorResult:
    """Locate a span interval from ordered start/end anchors inside a window."""
    start_anchor = start_anchor_quote.strip()
    end_anchor = end_anchor_quote.strip()
    if not start_anchor:
        return LocatorResult("failed", None, None, 0.0, "anchors", [], "start anchor is empty")
    if not end_anchor:
        return LocatorResult("failed", None, None, 0.0, "anchors", [], "end anchor is empty")

    start_candidates = _find_anchor_candidates(
        window,
        start_anchor,
        fuzzy_threshold=fuzzy_threshold,
        role="start",
    )
    end_candidates = _find_anchor_candidates(
        window,
        end_anchor,
        fuzzy_threshold=fuzzy_threshold,
        role="end",
    )
    if not start_candidates:
        return LocatorResult("failed", None, None, 0.0, "anchors", [], "start anchor not found")
    if not end_candidates:
        return LocatorResult("failed", None, None, 0.0, "anchors", [], "end anchor not found")

    interval_candidates: list[LocatorCandidate] = []
    for start_candidate in start_candidates:
        for end_candidate in end_candidates:
            if end_candidate.end_idx < start_candidate.start_idx:
                continue
            span_start = start_candidate.start_idx
            span_end = end_candidate.end_idx
            confidence = round((start_candidate.confidence + end_candidate.confidence) / 2, 4)
            strategy = f"anchors:{start_candidate.strategy}+{end_candidate.strategy}"
            matched_text = window.text[
                span_start - window.window_start : span_end - window.window_start
            ]
            candidate = LocatorCandidate(
                start_idx=span_start,
                end_idx=span_end,
                confidence=confidence,
                strategy=strategy,
                matched_text=matched_text,
            )
            interval_candidates.append(candidate)

    if interval_candidates:
        return _select_best_interval_candidate(window, interval_candidates)

    return LocatorResult(
        status="failed",
        start_idx=None,
        end_idx=None,
        confidence=0.0,
        strategy="anchors",
        candidates=[],
        failure_reason="no ordered start/end anchor pair found",
    )


def _exact_candidates(window: CandidateWindow, quote: str) -> list[LocatorCandidate]:
    candidates: list[LocatorCandidate] = []
    start = window.text.find(quote)
    while start >= 0:
        end = start + len(quote)
        candidates.append(
            LocatorCandidate(
                start_idx=window.window_start + start,
                end_idx=window.window_start + end,
                confidence=1.0,
                strategy="exact",
                matched_text=window.text[start:end],
            )
        )
        start = window.text.find(quote, start + 1)
    return candidates


def _find_anchor_candidates(
    window: CandidateWindow,
    quote: str,
    *,
    fuzzy_threshold: float,
    role: str,
) -> list[LocatorCandidate]:
    candidates = _exact_candidates(window, quote)
    if candidates:
        return [_with_strategy_prefix(candidate, role) for candidate in candidates]

    candidates = _normalized_candidates(window, quote)
    if candidates:
        return [_with_strategy_prefix(candidate, role) for candidate in candidates]

    candidates = _fuzzy_candidates(window, quote, fuzzy_threshold=fuzzy_threshold)
    return [_with_strategy_prefix(candidate, role) for candidate in candidates]


def _with_strategy_prefix(candidate: LocatorCandidate, role: str) -> LocatorCandidate:
    return LocatorCandidate(
        start_idx=candidate.start_idx,
        end_idx=candidate.end_idx,
        confidence=candidate.confidence,
        strategy=f"{role}_{candidate.strategy}",
        matched_text=candidate.matched_text,
    )


def _normalized_candidates(window: CandidateWindow, quote: str) -> list[LocatorCandidate]:
    normalized_window, window_map = _normalize_for_match(window.text)
    normalized_quote, _ = _normalize_for_match(quote)
    if not normalized_quote:
        return []

    candidates: list[LocatorCandidate] = []
    start = normalized_window.find(normalized_quote)
    while start >= 0:
        end = start + len(normalized_quote)
        original_start = window_map[start]
        original_end = window_map[end - 1] + 1
        candidates.append(
            LocatorCandidate(
                start_idx=window.window_start + original_start,
                end_idx=window.window_start + original_end,
                confidence=0.96,
                strategy="normalized",
                matched_text=window.text[original_start:original_end],
            )
        )
        start = normalized_window.find(normalized_quote, start + 1)
    return candidates


def _fuzzy_candidates(
    window: CandidateWindow,
    quote: str,
    *,
    fuzzy_threshold: float,
) -> list[LocatorCandidate]:
    normalized_window, window_map = _normalize_for_match(window.text)
    normalized_quote, _ = _normalize_for_match(quote)
    quote_length = len(normalized_quote)
    if quote_length < 8 or not normalized_window:
        return []

    min_length = max(4, int(quote_length * 0.75))
    max_length = min(len(normalized_window), max(quote_length + 24, int(quote_length * 1.25)))
    best_by_start: dict[int, LocatorCandidate] = {}

    for candidate_length in range(min_length, max_length + 1):
        for start in range(0, len(normalized_window) - candidate_length + 1):
            fragment = normalized_window[start : start + candidate_length]
            score = SequenceMatcher(None, normalized_quote, fragment, autojunk=False).ratio()
            if score < fuzzy_threshold:
                continue
            original_start = window_map[start]
            original_end = window_map[start + candidate_length - 1] + 1
            candidate = LocatorCandidate(
                start_idx=window.window_start + original_start,
                end_idx=window.window_start + original_end,
                confidence=round(score, 4),
                strategy="fuzzy",
                matched_text=window.text[original_start:original_end],
            )
            current = best_by_start.get(original_start)
            if current is None or candidate.confidence > current.confidence:
                best_by_start[original_start] = candidate

    return sorted(best_by_start.values(), key=lambda item: item.confidence, reverse=True)[:10]


def _normalize_for_match(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": ".",
        "　": " ",
    }
    previous_space = False
    for index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", replacements.get(char, char))
        for normalized_char in normalized:
            if normalized_char.isspace():
                if previous_space:
                    continue
                normalized_chars.append(" ")
                index_map.append(index)
                previous_space = True
                continue
            normalized_chars.append(normalized_char)
            index_map.append(index)
            previous_space = False
    normalized_text = "".join(normalized_chars).strip()
    if normalized_text and normalized_chars and normalized_chars[0].isspace():
        first_non_space = next(i for i, char in enumerate(normalized_chars) if not char.isspace())
        index_map = index_map[first_non_space:]
    if normalized_text and normalized_chars and normalized_chars[-1].isspace():
        trailing_spaces = len(normalized_chars) - len("".join(normalized_chars).rstrip())
        if trailing_spaces:
            index_map = index_map[:-trailing_spaces]
    return normalized_text, index_map


def _select_best_candidate(
    window: CandidateWindow,
    candidates: list[LocatorCandidate],
) -> LocatorResult:
    window_center = (window.window_start + window.window_end) / 2
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.confidence,
            -abs(((candidate.start_idx + candidate.end_idx) / 2) - window_center),
            -(candidate.end_idx - candidate.start_idx),
        ),
        reverse=True,
    )
    best = sorted_candidates[0]
    return LocatorResult(
        status="located",
        start_idx=best.start_idx,
        end_idx=best.end_idx,
        confidence=best.confidence,
        strategy=best.strategy,
        candidates=sorted_candidates,
    )


def _select_best_interval_candidate(
    window: CandidateWindow,
    candidates: list[LocatorCandidate],
) -> LocatorResult:
    window_center = (window.window_start + window.window_end) / 2
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.confidence,
            -abs(((candidate.start_idx + candidate.end_idx) / 2) - window_center),
            -(candidate.end_idx - candidate.start_idx),
        ),
        reverse=True,
    )
    best = sorted_candidates[0]
    return LocatorResult(
        status="located",
        start_idx=best.start_idx,
        end_idx=best.end_idx,
        confidence=best.confidence,
        strategy=best.strategy,
        candidates=sorted_candidates,
    )


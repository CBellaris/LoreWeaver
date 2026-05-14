"""Candidate-window selection helpers."""

from __future__ import annotations

from loreweaver.models.window import CandidateWindow


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

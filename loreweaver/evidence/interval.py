"""Evidence interval merge utilities for M1.7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loreweaver.models.chapter import Chapter
from loreweaver.models.evidence import MergedEvidenceInterval


@dataclass(frozen=True)
class SpanEvidenceSeed:
    span_id: str
    document_id: str
    chapter_id: str
    span_start_idx: int
    span_end_idx: int
    retrieval_sources: list[str]
    rerank_score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvidenceIntervalWarning:
    span_id: str
    reason: str


def build_span_evidence_seeds(top_results: list[dict[str, Any]]) -> tuple[list[SpanEvidenceSeed], list[EvidenceIntervalWarning]]:
    seeds: list[SpanEvidenceSeed] = []
    warnings: list[EvidenceIntervalWarning] = []
    for item in top_results:
        span_id = str(item.get("span_id", ""))
        start_idx = item.get("span_start_idx")
        end_idx = item.get("span_end_idx")
        if not span_id:
            warnings.append(EvidenceIntervalWarning(span_id="<missing>", reason="missing span_id"))
            continue
        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            warnings.append(EvidenceIntervalWarning(span_id=span_id, reason="missing coordinates"))
            continue
        if start_idx < 0 or end_idx <= start_idx:
            warnings.append(EvidenceIntervalWarning(span_id=span_id, reason="invalid coordinate range"))
            continue
        sources = [str(source) for source in item.get("sources", []) if str(source)]
        seeds.append(
            SpanEvidenceSeed(
                span_id=span_id,
                document_id=str(item.get("document_id", "")),
                chapter_id=str(item.get("chapter_id", "")),
                span_start_idx=start_idx,
                span_end_idx=end_idx,
                retrieval_sources=sources,
                rerank_score=float(item.get("rerank_score", 0.0)),
                metadata=dict(item),
            )
        )
    return seeds, warnings


def expand_seeds_to_intervals(
    seeds: list[SpanEvidenceSeed],
    *,
    chapters_by_id: dict[str, Chapter],
    pre_context_chars: int,
    post_context_chars: int,
) -> tuple[list[MergedEvidenceInterval], list[EvidenceIntervalWarning]]:
    intervals: list[MergedEvidenceInterval] = []
    warnings: list[EvidenceIntervalWarning] = []
    for seed in seeds:
        chapter = chapters_by_id.get(seed.chapter_id)
        if chapter is None:
            warnings.append(EvidenceIntervalWarning(span_id=seed.span_id, reason="chapter not found"))
            continue
        if seed.span_start_idx < chapter.start_idx or seed.span_end_idx > chapter.end_idx:
            warnings.append(
                EvidenceIntervalWarning(span_id=seed.span_id, reason="coordinates outside chapter")
            )
            continue
        start_idx = max(chapter.start_idx, seed.span_start_idx - max(0, pre_context_chars))
        end_idx = min(chapter.end_idx, seed.span_end_idx + max(0, post_context_chars))
        intervals.append(
            MergedEvidenceInterval(
                document_id=seed.document_id,
                chapter_id=seed.chapter_id,
                chapter_title=chapter.chapter_title,
                start_idx=start_idx,
                end_idx=end_idx,
                source_span_ids=[seed.span_id],
                retrieval_sources=sorted(set(seed.retrieval_sources)),
                rerank_score=seed.rerank_score,
                priority_score=_priority_score(seed.rerank_score, seed.retrieval_sources),
            )
        )
    return intervals, warnings


def merge_evidence_intervals(
    intervals: list[MergedEvidenceInterval],
    *,
    merge_gap_chars: int,
) -> list[MergedEvidenceInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item.chapter_id, item.start_idx, item.end_idx))
    merged: list[MergedEvidenceInterval] = [ordered[0]]
    for interval in ordered[1:]:
        previous = merged[-1]
        if (
            interval.chapter_id == previous.chapter_id
            and interval.start_idx <= previous.end_idx + max(0, merge_gap_chars)
        ):
            merged[-1] = _merge_two_intervals(previous, interval)
            continue
        merged.append(interval)
    return merged


def select_intervals_for_budget(
    intervals: list[MergedEvidenceInterval],
    *,
    max_evidence_chars: int,
    max_blocks: int,
) -> list[MergedEvidenceInterval]:
    if max_blocks <= 0 or max_evidence_chars <= 0:
        return []
    if not intervals:
        return []

    selected: list[MergedEvidenceInterval] = []
    selected_keys: set[tuple[str, int, int]] = set()
    selected_chapters: set[str] = set()
    used_chars = 0
    ordered = sorted(
        intervals,
        key=lambda item: (item.priority_score, item.rerank_score, len(item.retrieval_sources)),
        reverse=True,
    )

    def try_add(interval: MergedEvidenceInterval) -> None:
        nonlocal used_chars
        if len(selected) >= max_blocks:
            return
        key = (interval.chapter_id, interval.start_idx, interval.end_idx)
        if key in selected_keys:
            return
        remaining_chars = max_evidence_chars - used_chars
        if remaining_chars <= 0:
            return
        item = interval
        interval_chars = item.end_idx - item.start_idx
        if interval_chars > remaining_chars:
            item = MergedEvidenceInterval(
                document_id=item.document_id,
                chapter_id=item.chapter_id,
                chapter_title=item.chapter_title,
                start_idx=item.start_idx,
                end_idx=item.start_idx + remaining_chars,
                source_span_ids=item.source_span_ids,
                retrieval_sources=item.retrieval_sources,
                rerank_score=item.rerank_score,
                priority_score=item.priority_score,
            )
        selected.append(item)
        selected_keys.add(key)
        selected_chapters.add(item.chapter_id)
        used_chars += item.end_idx - item.start_idx

    for interval in ordered:
        if interval.chapter_id not in selected_chapters:
            try_add(interval)
    for interval in ordered:
        try_add(interval)

    return sorted(selected, key=lambda item: (item.start_idx, item.end_idx, item.chapter_id))


def interval_payload(interval: MergedEvidenceInterval) -> dict[str, Any]:
    return {
        "document_id": interval.document_id,
        "chapter_id": interval.chapter_id,
        "chapter_title": interval.chapter_title,
        "start_idx": interval.start_idx,
        "end_idx": interval.end_idx,
        "source_span_ids": interval.source_span_ids,
        "retrieval_sources": interval.retrieval_sources,
        "rerank_score": interval.rerank_score,
        "priority_score": interval.priority_score,
    }


def _merge_two_intervals(
    left: MergedEvidenceInterval,
    right: MergedEvidenceInterval,
) -> MergedEvidenceInterval:
    source_span_ids = list(dict.fromkeys([*left.source_span_ids, *right.source_span_ids]))
    retrieval_sources = sorted(set(left.retrieval_sources).union(right.retrieval_sources))
    rerank_score = max(left.rerank_score, right.rerank_score)
    return MergedEvidenceInterval(
        document_id=left.document_id or right.document_id,
        chapter_id=left.chapter_id,
        chapter_title=left.chapter_title,
        start_idx=min(left.start_idx, right.start_idx),
        end_idx=max(left.end_idx, right.end_idx),
        source_span_ids=source_span_ids,
        retrieval_sources=retrieval_sources,
        rerank_score=rerank_score,
        priority_score=max(
            left.priority_score,
            right.priority_score,
            _priority_score(rerank_score, retrieval_sources),
        ),
    )


def _priority_score(rerank_score: float, retrieval_sources: list[str]) -> float:
    source_bonus = 0.08 * max(0, len(set(retrieval_sources)) - 1)
    return round(float(rerank_score) + source_bonus, 8)

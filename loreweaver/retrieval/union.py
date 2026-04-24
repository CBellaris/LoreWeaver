"""Multi-source retrieval union logic for M1.6."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from loreweaver.models.span import Span
from loreweaver.retrieval.models import RetrievalHit, UnionCandidate
from loreweaver.storage.bm25_store import tokenize_for_bm25


def merge_retrieval_hits(
    hits: list[RetrievalHit],
    *,
    question: str,
    spans_by_id: dict[str, Span] | None = None,
    max_candidates: int = 80,
) -> list[UnionCandidate]:
    usable_hits = [
        hit for hit in hits if hit.span is not None or (spans_by_id and hit.span_id in spans_by_id)
    ]
    if not usable_hits:
        return []

    normalized = _normalize_by_source(usable_hits)
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in usable_hits:
        grouped[hit.span_id].append(hit)

    candidates: list[UnionCandidate] = []
    for span_id, span_hits in grouped.items():
        span = span_hits[0].span or (spans_by_id or {})[span_id]
        sources = sorted({hit.source for hit in span_hits})
        source_scores = {
            source: max(hit.score for hit in span_hits if hit.source == source)
            for source in sources
        }
        normalized_scores = {
            source: max(normalized[(span_id, source)] for hit in span_hits if hit.source == source)
            for source in sources
        }
        source_count_bonus = min(0.3, 0.1 * (len(sources) - 1))
        graph_bonus = 0.08 if "graph" in sources else 0.0
        entity_coverage = _entity_coverage(question, span)
        fused_score = min(
            1.0,
            0.62 * max(normalized_scores.values(), default=0.0)
            + 0.18 * _mean(list(normalized_scores.values()))
            + source_count_bonus
            + graph_bonus
            + 0.12 * entity_coverage,
        )
        cluster_ids = sorted(
            {
                str(hit.metadata["cluster_id"])
                for hit in span_hits
                if hit.metadata.get("cluster_id")
            }
        )
        candidates.append(
            UnionCandidate(
                span_id=span_id,
                span=span,
                sources=sources,
                source_scores=source_scores,
                normalized_scores=normalized_scores,
                fused_score=round(fused_score, 6),
                metadata={"cluster_ids": cluster_ids},
            )
        )

    return sorted(candidates, key=lambda item: item.fused_score, reverse=True)[:max_candidates]


def union_report(candidates: list[UnionCandidate], *, source_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "source_counts": dict(source_counts),
        "candidate_count": len(candidates),
        "multi_source_count": sum(1 for candidate in candidates if len(candidate.sources) > 1),
    }


def _normalize_by_source(hits: list[RetrievalHit]) -> dict[tuple[str, str], float]:
    by_source: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in hits:
        by_source[hit.source].append(hit)

    normalized: dict[tuple[str, str], float] = {}
    for source, source_hits in by_source.items():
        scores = [hit.score for hit in source_hits]
        min_score = min(scores)
        max_score = max(scores)
        span_best: dict[str, float] = {}
        for hit in source_hits:
            if max_score == min_score:
                value = 1.0
            else:
                value = (hit.score - min_score) / (max_score - min_score)
            span_best[hit.span_id] = max(value, span_best.get(hit.span_id, 0.0))
        for span_id, value in span_best.items():
            normalized[(span_id, source)] = value
    return normalized


def _entity_coverage(question: str, span: Span) -> float:
    if not span.entities:
        return 0.0
    question_tokens = set(tokenize_for_bm25(question))
    entity_tokens = {
        token
        for entity in span.entities
        for token in tokenize_for_bm25(entity)
    }
    if not entity_tokens:
        return 0.0
    return len(question_tokens.intersection(entity_tokens)) / len(entity_tokens)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

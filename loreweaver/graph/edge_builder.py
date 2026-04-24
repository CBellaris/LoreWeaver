"""Graph edge construction helpers for M1.5."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Protocol

from loreweaver.models.chapter import Chapter
from loreweaver.models.cluster import CenterSpanCluster, SpanEdge
from loreweaver.models.span import Span


class MemberLike(Protocol):
    span: Span
    score: float


def build_graph_edges(
    *,
    document_id: str,
    clusters: list[CenterSpanCluster],
    member_candidates_by_cluster: dict[str, list[MemberLike]],
    spans: list[Span],
    chapters: list[Chapter],
) -> list[SpanEdge]:
    now = datetime.now(timezone.utc)
    edges: dict[str, SpanEdge] = {}
    spans_by_id = {span.span_id: span for span in spans}

    for cluster in clusters:
        member_candidates = member_candidates_by_cluster.get(cluster.cluster_id, [])
        for member in member_candidates:
            _put_edge(
                edges,
                document_id=document_id,
                from_id=cluster.cluster_id,
                to_id=member.span.span_id,
                from_type="CenterSpanCluster",
                to_type="Span",
                edge_type="SUPPORTS",
                weight=member.score,
                source="rule",
                created_at=now,
            )
            if member.span.span_id != cluster.center_span_id:
                _put_edge(
                    edges,
                    document_id=document_id,
                    from_id=cluster.center_span_id,
                    to_id=member.span.span_id,
                    from_type="Span",
                    to_type="Span",
                    edge_type="RELATED_TO",
                    weight=min(1.0, member.score),
                    source="rule",
                    created_at=now,
                )

        for span_id in cluster.member_span_ids:
            span = spans_by_id.get(span_id)
            if span is None:
                continue
            for entity in span.entities:
                entity_id = f"entity::{document_id}::{entity}"
                _put_edge(
                    edges,
                    document_id=document_id,
                    from_id=span.span_id,
                    to_id=entity_id,
                    from_type="Span",
                    to_type="Entity",
                    edge_type="MENTIONS_ENTITY",
                    weight=1.0,
                    source="rule",
                    created_at=now,
                )

    ordered_chapters = sorted(chapters, key=lambda chapter: chapter.chapter_index)
    for left, right in zip(ordered_chapters, ordered_chapters[1:]):
        _put_edge(
            edges,
            document_id=document_id,
            from_id=left.chapter_id,
            to_id=right.chapter_id,
            from_type="Chapter",
            to_type="Chapter",
            edge_type="ADJACENT_CHAPTER",
            weight=1.0,
            source="rule",
            created_at=now,
        )

    return list(edges.values())


def _put_edge(
    edges: dict[str, SpanEdge],
    *,
    document_id: str,
    from_id: str,
    to_id: str,
    from_type: str,
    to_type: str,
    edge_type: str,
    weight: float,
    source: str,
    created_at: datetime,
) -> None:
    edge_id = _stable_edge_id(document_id, from_id, to_id, edge_type)
    edges[edge_id] = SpanEdge(
        edge_id=edge_id,
        document_id=document_id,
        from_id=from_id,
        to_id=to_id,
        from_type=from_type,
        to_type=to_type,
        edge_type=edge_type,
        weight=round(float(weight), 6),
        source=source,
        created_at=created_at,
    )


def _stable_edge_id(document_id: str, from_id: str, to_id: str, edge_type: str) -> str:
    raw = "::".join([document_id, from_id, to_id, edge_type])
    return f"edge_{uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:20]}"

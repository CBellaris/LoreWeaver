"""Graph-guided retrieval for M1.6."""

from __future__ import annotations

from typing import Any

from loreweaver.models.cluster import CenterSpanCluster
from loreweaver.retrieval.models import RetrievalHit
from loreweaver.storage.bm25_store import tokenize_for_bm25
from loreweaver.storage.sqlite_store import SQLiteStore


CLUSTER_TYPE_HINTS = {
    "character_relation": {"character"},
    "faction_history": {"faction", "history"},
    "location": {"location"},
    "power_system": {"power_system"},
    "timeline": {"history"},
}


def retrieve_graph(
    *,
    store: SQLiteStore,
    document_id: str,
    question: str,
    query_type: str,
    cluster_top_k: int,
    span_per_cluster: int,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    clusters = store.list_center_span_clusters(document_id)
    if not clusters:
        return [], {"source": "graph", "status": "no_clusters", "count": 0}

    scored_clusters = _rank_clusters(clusters, question, query_type)
    selected = scored_clusters[:cluster_top_k]
    span_ids: list[str] = []
    hit_specs: list[tuple[str, float, dict[str, Any]]] = []

    for cluster, cluster_score in selected:
        cluster_meta = {
            "cluster_id": cluster.cluster_id,
            "cluster_name": cluster.cluster_name,
            "cluster_type": cluster.cluster_type,
            "cluster_score": cluster_score,
        }
        span_ids.append(cluster.center_span_id)
        hit_specs.append(
            (
                cluster.center_span_id,
                min(1.0, 0.65 + cluster_score * 0.35),
                {**cluster_meta, "role": "center_span"},
            )
        )

        support_edges = store.list_span_edges(
            document_id,
            edge_type="SUPPORTS",
            from_id=cluster.cluster_id,
        )
        support_edges = sorted(support_edges, key=lambda edge: edge.weight, reverse=True)
        member_ids = [edge.to_id for edge in support_edges[:span_per_cluster]]
        if not member_ids:
            member_ids = cluster.member_span_ids[:span_per_cluster]
        for rank, span_id in enumerate(member_ids, start=1):
            span_ids.append(span_id)
            edge_weight = next(
                (edge.weight for edge in support_edges if edge.to_id == span_id),
                cluster.confidence,
            )
            hit_specs.append(
                (
                    span_id,
                    min(1.0, 0.5 * cluster_score + 0.5 * edge_weight),
                    {**cluster_meta, "role": "member_span", "member_rank": rank},
                )
            )

    spans_by_id = {span.span_id: span for span in store.list_spans_by_ids(span_ids)}
    hits = [
        RetrievalHit(
            span_id=span_id,
            source="graph",
            score=score,
            span=spans_by_id.get(span_id),
            metadata=metadata,
        )
        for span_id, score, metadata in hit_specs
        if span_id in spans_by_id
    ]
    return hits, {
        "source": "graph",
        "status": "ok",
        "cluster_count": len(selected),
        "count": len(hits),
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.cluster_name,
                "cluster_type": cluster.cluster_type,
                "score": score,
            }
            for cluster, score in selected
        ],
    }


def _rank_clusters(
    clusters: list[CenterSpanCluster],
    question: str,
    query_type: str,
) -> list[tuple[CenterSpanCluster, float]]:
    query_tokens = set(tokenize_for_bm25(question))
    preferred_types = CLUSTER_TYPE_HINTS.get(query_type, set())
    scored: list[tuple[CenterSpanCluster, float]] = []
    for cluster in clusters:
        cluster_text = "\n".join(
            [
                cluster.cluster_name,
                cluster.cluster_type,
                cluster.summary,
            ]
        )
        cluster_tokens = set(tokenize_for_bm25(cluster_text))
        overlap = len(query_tokens.intersection(cluster_tokens))
        coverage = overlap / max(1, len(query_tokens))
        type_boost = 0.25 if cluster.cluster_type in preferred_types else 0.0
        confidence_boost = min(0.2, cluster.confidence * 0.2)
        score = min(1.0, coverage + type_boost + confidence_boost)
        scored.append((cluster, score))

    return sorted(
        scored,
        key=lambda item: (item[1], item[0].confidence, item[0].cluster_name),
        reverse=True,
    )

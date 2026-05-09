"""Center Span Cluster construction for M1.5."""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.graph.edge_builder import build_graph_edges
from loreweaver.models.cluster import CenterSpanCluster, SpanEdge
from loreweaver.models.span import Span
from loreweaver.progress import ProgressReporter
from loreweaver.storage.bm25_store import tokenize_for_bm25
from loreweaver.storage.neo4j_store import Neo4jGraphStore
from loreweaver.storage.qdrant_store import QdrantVectorStore
from loreweaver.storage.sqlite_store import SQLiteStore


ALLOWED_CLUSTER_TYPES = (
    "character",
    "faction",
    "location",
    "power_system",
    "history",
    "mystery",
)

CLUSTER_TYPE_LABELS = {
    "character": "角色关系",
    "faction": "核心势力",
    "location": "关键地点",
    "power_system": "力量体系",
    "history": "历史事件",
    "mystery": "悬疑异常",
}


@dataclass(frozen=True)
class MemberCandidate:
    span: Span
    score: float
    reasons: list[str]
    component_scores: dict[str, float]


@dataclass(frozen=True)
class GraphScoringWeights:
    vector: float = 0.4
    entity: float = 0.2
    topic: float = 0.15
    bm25: float = 0.1
    chapter: float = 0.1
    salience: float = 0.05


@dataclass(frozen=True)
class GraphVectorLoad:
    enabled: bool
    source: str
    requested_count: int
    loaded_count: int
    coverage: float
    error: str | None = None


def build_m15_graph(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    run_id: str,
    document_id: str | None = None,
    cluster_count: int | None = None,
    members_per_cluster: int | None = None,
    min_members: int | None = None,
    use_embeddings: bool | None = None,
    sync_neo4j: bool | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Build the lightweight M1.5 center-span graph and write a debug report."""
    if progress is not None:
        progress = progress.child(command="graph", run_id=run_id)
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_graph_tables()
    document = store.get_document(document_id)
    if progress is not None:
        progress.emit(
            "planned",
            stage="graph.plan",
            label="Plan graph build",
            current=0,
            total=6,
            unit="steps",
            detail={"document_id": document.document_id},
        )
        progress.emit("stage_start", stage="graph.load_spans", label="Load located spans", current=1, total=6, unit="steps")
    spans = store.list_spans(document.document_id, located_only=True)
    if not spans:
        raise ValueError(
            f"No located spans found for document_id={document.document_id}. "
            "Run loreweaver extract first."
        )

    graph_config = config.values.get("graph", {})
    effective_cluster_count = int(cluster_count or graph_config.get("cluster_count", 4))
    effective_members_per_cluster = int(
        members_per_cluster or graph_config.get("members_per_cluster", 8)
    )
    effective_min_members = int(min_members or graph_config.get("min_members_per_cluster", 5))
    effective_use_embeddings = (
        bool(graph_config.get("use_embeddings", True))
        if use_embeddings is None
        else use_embeddings
    )
    weights = _weights_from_config(graph_config)
    if not effective_use_embeddings:
        weights = _normalize_weights(
            GraphScoringWeights(
                vector=0.0,
                entity=weights.entity,
                topic=weights.topic,
                bm25=weights.bm25,
                chapter=weights.chapter,
                salience=weights.salience,
            )
        )
    if progress is not None:
        progress.emit(
            "stage_start",
            stage="graph.load_vectors",
            label="Load span vectors",
            current=2,
            total=6,
            unit="steps",
            detail={"span_count": len(spans), "enabled": effective_use_embeddings},
        )
    span_vectors, vector_load = _load_span_vectors(
        storage_config=storage_config,
        document_id=document.document_id,
        spans=spans,
        enabled=effective_use_embeddings,
    )
    effective_sync_neo4j = bool(
        storage_config.values.get("neo4j", {}).get("enabled", False)
        if sync_neo4j is None
        else sync_neo4j
    )

    if progress is not None:
        progress.emit(
            "stage_start",
            stage="graph.clusters",
            label="Build center-span clusters",
            current=3,
            total=6,
            unit="steps",
            detail={
                "cluster_count": effective_cluster_count,
                "members_per_cluster": effective_members_per_cluster,
            },
        )
    clusters, member_candidates_by_cluster = build_center_span_clusters(
        document_id=document.document_id,
        spans=spans,
        cluster_count=effective_cluster_count,
        members_per_cluster=effective_members_per_cluster,
        min_members=effective_min_members,
        span_vectors=span_vectors,
        weights=weights,
    )
    if not clusters:
        raise ValueError(
            "Unable to build any CenterSpanCluster from located spans. "
            "Try extracting more windows first."
        )

    chapters = store.list_chapters(document.document_id)
    if progress is not None:
        progress.emit(
            "stage_start",
            stage="graph.edges",
            label="Build graph edges",
            current=4,
            total=6,
            unit="steps",
            detail={"cluster_count": len(clusters)},
        )
    edges = build_graph_edges(
        document_id=document.document_id,
        clusters=clusters,
        member_candidates_by_cluster=member_candidates_by_cluster,
        spans=spans,
        chapters=chapters,
    )
    if progress is not None:
        progress.emit(
            "stage_start",
            stage="graph.sqlite",
            label="Persist graph to SQLite",
            current=5,
            total=6,
            unit="steps",
            detail={"cluster_count": len(clusters), "edge_count": len(edges)},
        )
    store.replace_graph(document_id=document.document_id, clusters=clusters, edges=edges)

    neo4j_report = {"enabled": effective_sync_neo4j, "synced": False, "message": "disabled"}
    if effective_sync_neo4j:
        if progress is not None:
            progress.emit(
                "stage_start",
                stage="graph.neo4j",
                label="Sync graph to Neo4j",
                current=5,
                total=6,
                unit="steps",
            )
        neo4j_store = Neo4jGraphStore.from_config(storage_config)
        try:
            neo4j_report = neo4j_store.replace_graph(
                document_id=document.document_id,
                clusters=clusters,
                spans=spans,
                edges=edges,
            )
        finally:
            neo4j_store.close()

    report = _build_report(
        run_id=run_id,
        document_id=document.document_id,
        sqlite_path=str(storage_config.sqlite_path),
        clusters=clusters,
        edges=edges,
        member_candidates_by_cluster=member_candidates_by_cluster,
        vector_load=vector_load,
        weights=weights,
        neo4j_report=neo4j_report,
    )
    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{run_id}_graph_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.insert_graph_report(run_id, document.document_id, report)
    if progress is not None:
        progress.emit(
            "completed",
            stage="graph.completed",
            label="Graph build completed",
            current=6,
            total=6,
            unit="steps",
            status="completed",
            detail={
                "report_path": str(report_path),
                "cluster_count": len(clusters),
                "edge_count": len(edges),
                "neo4j": neo4j_report,
            },
        )
    return report


def build_center_span_clusters(
    *,
    document_id: str,
    spans: list[Span],
    cluster_count: int,
    members_per_cluster: int,
    min_members: int,
    span_vectors: dict[str, list[float]] | None = None,
    weights: GraphScoringWeights | None = None,
) -> tuple[list[CenterSpanCluster], dict[str, list[MemberCandidate]]]:
    located_spans = [
        span
        for span in spans
        if span.locator_status == "located"
        and span.span_start_idx is not None
        and span.span_end_idx is not None
    ]
    if not located_spans:
        return [], {}

    effective_vectors = span_vectors or {}
    effective_weights = weights or GraphScoringWeights()
    entity_counts = Counter(entity for span in located_spans for entity in set(span.entities))
    topic_counts = Counter(topic for span in located_spans for topic in set(span.topics))
    now = datetime.now(timezone.utc)
    by_type: dict[str, list[Span]] = defaultdict(list)
    for span in located_spans:
        by_type[classify_cluster_type(span)].append(span)

    type_order = sorted(
        by_type,
        key=lambda cluster_type: (
            len(by_type[cluster_type]),
            max(span.salience_score for span in by_type[cluster_type]),
        ),
        reverse=True,
    )

    clusters: list[CenterSpanCluster] = []
    member_candidates_by_cluster: dict[str, list[MemberCandidate]] = {}
    used_centers: set[str] = set()
    for cluster_type in type_order:
        if len(clusters) >= cluster_count:
            break
        center = _choose_center(
            by_type[cluster_type],
            used_centers,
            effective_vectors,
            cluster_type,
        )
        if center is None:
            continue
        members = _rank_member_candidates(
            center=center,
            spans=located_spans,
            preferred_type=cluster_type,
            limit=members_per_cluster,
            min_members=min_members,
            span_vectors=effective_vectors,
            entity_counts=entity_counts,
            topic_counts=topic_counts,
            weights=effective_weights,
        )
        if len(members) < min_members:
            continue
        cluster_id = _stable_id("cluster", document_id, cluster_type, center.span_id)
        cluster = CenterSpanCluster(
            cluster_id=cluster_id,
            document_id=document_id,
            center_span_id=center.span_id,
            cluster_name=_cluster_name(cluster_type, center, [member.span for member in members]),
            cluster_type=cluster_type,
            micro_summary=_cluster_summary(cluster_type, center, [member.span for member in members]),
            member_span_ids=[member.span.span_id for member in members],
            confidence=round(_cluster_confidence(members), 4),
            status="active",
            created_at=now,
        )
        clusters.append(cluster)
        member_candidates_by_cluster[cluster.cluster_id] = members
        used_centers.add(center.span_id)
    return clusters, member_candidates_by_cluster


def classify_cluster_type(span: Span) -> str:
    span_type = span.span_type.lower()
    combined = " ".join([span.micro_topic, span.micro_summary, *span.topics, *span.entities])
    if span_type == "location_lore" or _contains_any(combined, ("地点", "地理", "城堡", "遗迹")):
        return "location"
    if span_type == "faction_lore" or _contains_any(combined, ("势力", "家族", "王国", "贵族")):
        return "faction"
    if span_type == "power_rule" or _contains_any(combined, ("魔法", "能力", "规则", "精神")):
        return "power_system"
    if span_type == "mystery_clue" or _contains_any(combined, ("异常", "伏笔", "秘密", "谜")):
        return "mystery"
    if span_type in {"event", "scene_action"} or _contains_any(combined, ("历史", "事件", "战争")):
        return "history"
    if span.entities or span_type in {"dialogue_exchange", "relationship_signal"}:
        return "character"
    return "history"


def _choose_center(
    spans: list[Span],
    used_centers: set[str],
    span_vectors: dict[str, list[float]],
    cluster_type: str,
) -> Span | None:
    candidates = [span for span in spans if span.span_id not in used_centers]
    if not candidates:
        return None
    if span_vectors:
        vector_candidates = [span for span in candidates if span.span_id in span_vectors]
        if vector_candidates:
            return sorted(
                vector_candidates,
                key=lambda span: (
                    _center_type_bonus(span, cluster_type),
                    _medoid_score(span, spans, span_vectors),
                    span.salience_score,
                    len(span.entities),
                    -(span.span_start_idx or 0),
                ),
                reverse=True,
            )[0]
    return sorted(
        candidates,
        key=lambda span: (
            _center_type_bonus(span, cluster_type),
            span.salience_score,
            len(span.entities),
            -(span.span_start_idx or 0),
        ),
        reverse=True,
    )[0]


def _rank_member_candidates(
    *,
    center: Span,
    spans: list[Span],
    preferred_type: str,
    limit: int,
    min_members: int,
    span_vectors: dict[str, list[float]],
    entity_counts: Counter[str],
    topic_counts: Counter[str],
    weights: GraphScoringWeights,
) -> list[MemberCandidate]:
    ranked = [
        _score_member(
            center=center,
            span=span,
            preferred_type=preferred_type,
            span_vectors=span_vectors,
            entity_counts=entity_counts,
            topic_counts=topic_counts,
            weights=weights,
        )
        for span in spans
    ]
    ranked.sort(
        key=lambda candidate: (
            candidate.span.span_id == center.span_id,
            candidate.score,
            candidate.span.salience_score,
            -(candidate.span.span_start_idx or 0),
        ),
        reverse=True,
    )
    selected = ranked[: max(limit, min_members)]
    if selected and selected[0].span.span_id != center.span_id:
        center_candidate = next(
            candidate for candidate in ranked if candidate.span.span_id == center.span_id
        )
        selected = [center_candidate, *[item for item in selected if item.span.span_id != center.span_id]]
    return selected[: max(limit, min_members)]


def _score_member(
    *,
    center: Span,
    span: Span,
    preferred_type: str,
    span_vectors: dict[str, list[float]],
    entity_counts: Counter[str],
    topic_counts: Counter[str],
    weights: GraphScoringWeights,
) -> MemberCandidate:
    reasons: list[str] = []
    if span.span_id == center.span_id:
        reasons.append("center_span")

    vector_score = _vector_score(center, span, span_vectors)
    entity_score, shared_entities = _weighted_overlap_score(
        center.entities,
        span.entities,
        entity_counts,
        total_items=max(1, len(span_vectors) or sum(entity_counts.values()) or 1),
    )
    topic_score, shared_topics = _weighted_overlap_score(
        center.topics,
        span.topics,
        topic_counts,
        total_items=max(1, len(span_vectors) or sum(topic_counts.values()) or 1),
    )
    bm25_score = min(1.0, _lexical_overlap(center, span) / 20.0)
    chapter_score = _chapter_proximity_score(center, span)
    salience_score = max(0.0, min(1.0, span.salience_score))

    component_scores = {
        "vector": round(vector_score, 6),
        "entity": round(entity_score, 6),
        "topic": round(topic_score, 6),
        "bm25": round(bm25_score, 6),
        "chapter": round(chapter_score, 6),
        "salience": round(salience_score, 6),
    }

    score = (
        weights.vector * vector_score
        + weights.entity * entity_score
        + weights.topic * topic_score
        + weights.bm25 * bm25_score
        + weights.chapter * chapter_score
        + weights.salience * salience_score
    )
    if span.span_id == center.span_id:
        score = 1.0
    if classify_cluster_type(span) == preferred_type:
        reasons.append("same_cluster_type")
    if shared_entities:
        reasons.append("shared_entities:" + ",".join(shared_entities[:4]))
    if shared_topics:
        reasons.append("shared_topics:" + ",".join(shared_topics[:4]))
    if chapter_score == 1.0:
        reasons.append("same_chapter")
    elif chapter_score >= 0.5:
        reasons.append("adjacent_chapter")
    if bm25_score:
        reasons.append(f"bm25_overlap:{bm25_score:.3f}")
    if vector_score:
        reasons.append(f"vector_similarity:{vector_score:.3f}")
    if not reasons:
        reasons.append("salience_fallback")
    return MemberCandidate(
        span=span,
        score=round(score, 6),
        reasons=reasons,
        component_scores=component_scores,
    )


def _cluster_name(cluster_type: str, center: Span, members: list[Span]) -> str:
    entity_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    for span in members:
        entity_counts.update(span.entities)
        topic_counts.update(span.topics)
    label = CLUSTER_TYPE_LABELS.get(cluster_type, cluster_type)
    representative_entity = _representative_entity(cluster_type, entity_counts)
    if representative_entity:
        return f"{label}：{representative_entity}"
    if topic_counts:
        return f"{label}：{topic_counts.most_common(1)[0][0]}"
    return f"{label}：{center.micro_topic}"


def _cluster_summary(cluster_type: str, center: Span, members: list[Span]) -> str:
    topics = Counter(topic for span in members for topic in span.topics).most_common(3)
    topic_text = "、".join(topic for topic, _ in topics) if topics else center.micro_topic
    label = CLUSTER_TYPE_LABELS.get(cluster_type, cluster_type)
    return f"{label}聚合，以中心 Span「{center.micro_topic}」为锚点，成员主要覆盖：{topic_text}。"


def _cluster_confidence(members: list[MemberCandidate]) -> float:
    if not members:
        return 0.0
    non_center_scores = [
        member.score for member in members if "center_span" not in member.reasons
    ] or [members[0].score]
    return min(1.0, sum(non_center_scores) / len(non_center_scores))


def _build_report(
    *,
    run_id: str,
    document_id: str,
    sqlite_path: str,
    clusters: list[CenterSpanCluster],
    edges: list[SpanEdge],
    member_candidates_by_cluster: dict[str, list[MemberCandidate]],
    vector_load: GraphVectorLoad,
    weights: GraphScoringWeights,
    neo4j_report: dict[str, Any],
) -> dict[str, Any]:
    edge_counts = Counter(edge.edge_type for edge in edges)
    return {
        "run_id": run_id,
        "document_id": document_id,
        "sqlite_path": sqlite_path,
        "cluster_count": len(clusters),
        "edge_count": len(edges),
        "edge_counts": dict(edge_counts),
        "scoring": {
            "weights": {
                "vector": weights.vector,
                "entity": weights.entity,
                "topic": weights.topic,
                "bm25": weights.bm25,
                "chapter": weights.chapter,
                "salience": weights.salience,
            },
            "vector_load": {
                "enabled": vector_load.enabled,
                "source": vector_load.source,
                "requested_count": vector_load.requested_count,
                "loaded_count": vector_load.loaded_count,
                "coverage": vector_load.coverage,
                "error": vector_load.error,
            },
        },
        "neo4j": neo4j_report,
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.cluster_name,
                "cluster_type": cluster.cluster_type,
                "center_span_id": cluster.center_span_id,
                "member_count": len(cluster.member_span_ids),
                "confidence": cluster.confidence,
                "summary": cluster.micro_summary,
                "members": [
                    {
                        "span_id": member.span.span_id,
                        "score": member.score,
                        "component_scores": member.component_scores,
                        "reasons": member.reasons,
                        "micro_topic": member.span.micro_topic,
                        "span_type": member.span.span_type,
                        "chapter_id": member.span.chapter_id,
                        "range": [member.span.span_start_idx, member.span.span_end_idx],
                    }
                    for member in member_candidates_by_cluster[cluster.cluster_id]
                ],
            }
            for cluster in clusters
        ],
    }


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "::".join(parts)
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:16]}"


def _center_type_bonus(span: Span, cluster_type: str) -> float:
    span_type = span.span_type.lower()
    preferred_types = {
        "character": {"dialogue_exchange", "relationship_signal"},
        "faction": {"faction_lore"},
        "location": {"location_lore"},
        "power_system": {"power_rule"},
        "history": {"event", "scene_action"},
        "mystery": {"mystery_clue"},
    }
    if span_type in preferred_types.get(cluster_type, set()):
        return 1.0
    return 0.0


def _representative_entity(cluster_type: str, entity_counts: Counter[str]) -> str | None:
    if not entity_counts:
        return None
    type_keywords = {
        "faction": ("家族", "帝国", "王国", "领", "组织", "骑士团", "教会"),
        "location": ("领", "城堡", "隧道", "大陆", "塔", "遗迹", "营地", "陵寝", "山", "河"),
        "character": ("·",),
    }
    keywords = type_keywords.get(cluster_type, ())
    ranked = sorted(
        entity_counts,
        key=lambda entity: (
            any(keyword in entity for keyword in keywords),
            entity_counts[entity],
            len(entity),
        ),
        reverse=True,
    )
    return ranked[0]


def _weights_from_config(graph_config: dict[str, Any]) -> GraphScoringWeights:
    raw_weights = graph_config.get("scoring_weights", {})
    weights = GraphScoringWeights(
        vector=float(raw_weights.get("vector", 0.4)),
        entity=float(raw_weights.get("entity", 0.2)),
        topic=float(raw_weights.get("topic", 0.15)),
        bm25=float(raw_weights.get("bm25", 0.1)),
        chapter=float(raw_weights.get("chapter", 0.1)),
        salience=float(raw_weights.get("salience", 0.05)),
    )
    return _normalize_weights(weights)


def _normalize_weights(weights: GraphScoringWeights) -> GraphScoringWeights:
    total = (
        weights.vector
        + weights.entity
        + weights.topic
        + weights.bm25
        + weights.chapter
        + weights.salience
    )
    if total <= 0:
        return GraphScoringWeights()
    return GraphScoringWeights(
        vector=weights.vector / total,
        entity=weights.entity / total,
        topic=weights.topic / total,
        bm25=weights.bm25 / total,
        chapter=weights.chapter / total,
        salience=weights.salience / total,
    )


def _load_span_vectors(
    *,
    storage_config: AppConfig,
    document_id: str,
    spans: list[Span],
    enabled: bool,
) -> tuple[dict[str, list[float]], GraphVectorLoad]:
    requested_count = len(spans)
    if not enabled:
        return {}, GraphVectorLoad(
            enabled=False,
            source="disabled",
            requested_count=requested_count,
            loaded_count=0,
            coverage=0.0,
        )
    try:
        vector_store = QdrantVectorStore.from_config(storage_config, document_id=document_id)
        try:
            vectors = vector_store.retrieve_vectors([span.span_id for span in spans])
        finally:
            vector_store.close()
    except Exception as error:  # noqa: BLE001 - graph build must keep its local fallback.
        return {}, GraphVectorLoad(
            enabled=True,
            source="qdrant",
            requested_count=requested_count,
            loaded_count=0,
            coverage=0.0,
            error=str(error),
        )
    return vectors, GraphVectorLoad(
        enabled=True,
        source="qdrant",
        requested_count=requested_count,
        loaded_count=len(vectors),
        coverage=round((len(vectors) / requested_count) if requested_count else 0.0, 6),
    )


def _medoid_score(span: Span, candidates: list[Span], span_vectors: dict[str, list[float]]) -> float:
    similarities = [
        _cosine_similarity(span_vectors[span.span_id], span_vectors[other.span_id])
        for other in candidates
        if other.span_id != span.span_id and other.span_id in span_vectors
    ]
    if not similarities:
        return span.salience_score * 0.3
    top_similarities = sorted(similarities, reverse=True)[: min(8, len(similarities))]
    average_similarity = sum(top_similarities) / len(top_similarities)
    return 0.7 * max(0.0, average_similarity) + 0.3 * span.salience_score


def _vector_score(center: Span, span: Span, span_vectors: dict[str, list[float]]) -> float:
    center_vector = span_vectors.get(center.span_id)
    span_vector = span_vectors.get(span.span_id)
    if center_vector is None or span_vector is None:
        return 0.0
    return max(0.0, min(1.0, _cosine_similarity(center_vector, span_vector)))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _weighted_overlap_score(
    center_items: list[str],
    span_items: list[str],
    item_counts: Counter[str],
    *,
    total_items: int,
) -> tuple[float, list[str]]:
    shared_items = sorted(set(center_items).intersection(span_items))
    if not shared_items:
        return 0.0, []
    max_idf = max((_idf(count, total_items) for count in item_counts.values()), default=1.0)
    score = sum(_idf(item_counts[item], total_items) for item in shared_items)
    return min(1.0, score / max(1.0, max_idf * 2.0)), shared_items


def _idf(item_count: int, total_items: int) -> float:
    return 1.0 + math.log((total_items + 1.0) / (item_count + 1.0))


def _chapter_proximity_score(center: Span, span: Span) -> float:
    chapter_gap = abs(_chapter_number(center.chapter_id) - _chapter_number(span.chapter_id))
    if chapter_gap == 0:
        return 1.0
    if chapter_gap == 1:
        return 0.6
    if chapter_gap == 2:
        return 0.35
    return max(0.0, 0.2 / chapter_gap)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _lexical_overlap(center: Span, span: Span) -> int:
    center_tokens = set(tokenize_for_bm25(_span_text(center)))
    span_tokens = set(tokenize_for_bm25(_span_text(span)))
    return len(center_tokens.intersection(span_tokens))


def _span_text(span: Span) -> str:
    return "\n".join([span.micro_topic, span.micro_summary, *span.entities, *span.topics])


def _chapter_number(chapter_id: str) -> int:
    match = re.search(r"ch(\d+)", chapter_id)
    if not match:
        return 0
    return int(match.group(1))


def list_graph_clusters(
    *,
    storage_config: AppConfig,
    document_id: str | None = None,
    cluster_id: str | None = None,
) -> dict[str, Any]:
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_graph_tables()
    document = store.get_document(document_id)
    clusters = store.list_center_span_clusters(document.document_id, cluster_id=cluster_id)
    edges = store.list_span_edges(document.document_id)
    member_ids = [span_id for cluster in clusters for span_id in cluster.member_span_ids]
    spans_by_id = {span.span_id: span for span in store.list_spans_by_ids(member_ids)}
    return {
        "document_id": document.document_id,
        "cluster_count": len(clusters),
        "edge_count": len(edges),
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.cluster_name,
                "cluster_type": cluster.cluster_type,
                "center_span_id": cluster.center_span_id,
                "member_count": len(cluster.member_span_ids),
                "confidence": cluster.confidence,
                "summary": cluster.micro_summary,
                "members": [
                    _span_preview(spans_by_id[span_id])
                    for span_id in cluster.member_span_ids
                    if span_id in spans_by_id
                ],
            }
            for cluster in clusters
        ],
    }


def _span_preview(span: Span) -> dict[str, Any]:
    return {
        "span_id": span.span_id,
        "chapter_id": span.chapter_id,
        "micro_topic": span.micro_topic,
        "span_type": span.span_type,
        "salience_score": span.salience_score,
        "range": [span.span_start_idx, span.span_end_idx],
        "key_quote": span.key_quote,
    }

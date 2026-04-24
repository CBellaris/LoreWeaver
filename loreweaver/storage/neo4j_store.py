"""Neo4j graph store adapter for M1.5."""

from __future__ import annotations

import os
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.models.cluster import CenterSpanCluster, SpanEdge
from loreweaver.models.span import Span


class Neo4jGraphStore:
    """Optional Neo4j sink for the lightweight M1.5 graph."""

    def __init__(self, driver: Any | None) -> None:
        self.driver = driver

    @classmethod
    def from_config(cls, storage_config: AppConfig) -> "Neo4jGraphStore":
        neo4j_config = storage_config.values.get("neo4j", {})
        uri = _env_value(neo4j_config.get("uri_env"))
        username = _env_value(neo4j_config.get("username_env"))
        password = _env_value(neo4j_config.get("password_env"))
        if not uri or not username or not password:
            raise ValueError(
                "Neo4j sync is enabled, but one or more Neo4j environment variables are missing."
            )
        try:
            from neo4j import GraphDatabase
        except ImportError as error:
            raise RuntimeError(
                "The neo4j package is required when storage.neo4j.enabled is true."
            ) from error
        return cls(GraphDatabase.driver(uri, auth=(username, password)))

    def replace_graph(
        self,
        *,
        document_id: str,
        clusters: list[CenterSpanCluster],
        spans: list[Span],
        edges: list[SpanEdge],
    ) -> dict[str, Any]:
        if self.driver is None:
            return {"enabled": False, "synced": False, "message": "driver not configured"}
        span_ids = {span_id for cluster in clusters for span_id in cluster.member_span_ids}
        cluster_spans = [span for span in spans if span.span_id in span_ids]
        with self.driver.session() as session:
            session.execute_write(_replace_graph_tx, document_id, clusters, cluster_spans, edges)
        return {
            "enabled": True,
            "synced": True,
            "cluster_count": len(clusters),
            "span_count": len(cluster_spans),
            "edge_count": len(edges),
        }

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()


def _replace_graph_tx(
    tx: Any,
    document_id: str,
    clusters: list[CenterSpanCluster],
    spans: list[Span],
    edges: list[SpanEdge],
) -> None:
    tx.run(
        """
        MATCH (n {document_id: $document_id})
        WHERE n:LoreWeaverCluster OR n:LoreWeaverSpan OR n:LoreWeaverEntity
        DETACH DELETE n
        """,
        document_id=document_id,
    )
    tx.run(
        """
        MATCH ()-[r:SUPPORTS|RELATED_TO|MENTIONS_ENTITY|ADJACENT_CHAPTER]->()
        WHERE r.document_id = $document_id
        DELETE r
        """,
        document_id=document_id,
    )
    for span in spans:
        tx.run(
            """
            MERGE (s:LoreWeaverSpan {span_id: $span_id})
            SET s.document_id = $document_id,
                s.chapter_id = $chapter_id,
                s.micro_topic = $micro_topic,
                s.span_type = $span_type,
                s.micro_summary = $micro_summary,
                s.span_start_idx = $span_start_idx,
                s.span_end_idx = $span_end_idx
            """,
            span_id=span.span_id,
            document_id=span.document_id,
            chapter_id=span.chapter_id,
            micro_topic=span.micro_topic,
            span_type=span.span_type,
            micro_summary=span.micro_summary,
            span_start_idx=span.span_start_idx,
            span_end_idx=span.span_end_idx,
        )
    for cluster in clusters:
        tx.run(
            """
            MERGE (c:LoreWeaverCluster {cluster_id: $cluster_id})
            SET c.document_id = $document_id,
                c.center_span_id = $center_span_id,
                c.cluster_name = $cluster_name,
                c.cluster_type = $cluster_type,
                c.micro_summary = $micro_summary,
                c.confidence = $confidence,
                c.status = $status
            """,
            cluster_id=cluster.cluster_id,
            document_id=cluster.document_id,
            center_span_id=cluster.center_span_id,
            cluster_name=cluster.cluster_name,
            cluster_type=cluster.cluster_type,
            micro_summary=cluster.micro_summary,
            confidence=cluster.confidence,
            status=cluster.status,
        )
    for edge in edges:
        if edge.to_type == "Entity":
            tx.run(
                """
                MERGE (e:LoreWeaverEntity {entity_id: $entity_id})
                SET e.document_id = $document_id,
                    e.name = $name
                """,
                entity_id=edge.to_id,
                document_id=edge.document_id,
                name=edge.to_id.rsplit("::", 1)[-1],
            )
        if edge.from_type == "Chapter":
            _merge_chapter_node(tx, document_id=edge.document_id, chapter_id=edge.from_id)
        if edge.to_type == "Chapter":
            _merge_chapter_node(tx, document_id=edge.document_id, chapter_id=edge.to_id)
        _merge_edge(tx, edge)


def _merge_edge(tx: Any, edge: SpanEdge) -> None:
    from_label = _node_label(edge.from_type)
    to_label = _node_label(edge.to_type)
    from_key = _node_key(edge.from_type)
    to_key = _node_key(edge.to_type)
    cypher = f"""
    MATCH (a:{from_label} {{{from_key}: $from_id}})
    MATCH (b:{to_label} {{{to_key}: $to_id}})
    MERGE (a)-[r:{edge.edge_type} {{edge_id: $edge_id}}]->(b)
    SET r.document_id = $document_id,
        r.weight = $weight,
        r.source = $source
    """
    tx.run(
        cypher,
        from_id=edge.from_id,
        to_id=edge.to_id,
        edge_id=edge.edge_id,
        document_id=edge.document_id,
        weight=edge.weight,
        source=edge.source,
    )


def _merge_chapter_node(tx: Any, *, document_id: str, chapter_id: str) -> None:
    tx.run(
        """
        MERGE (c:LoreWeaverChapter {chapter_id: $chapter_id})
        SET c.document_id = $document_id
        """,
        chapter_id=chapter_id,
        document_id=document_id,
    )


def _node_label(node_type: str) -> str:
    if node_type == "CenterSpanCluster":
        return "LoreWeaverCluster"
    if node_type == "Entity":
        return "LoreWeaverEntity"
    if node_type == "Chapter":
        return "LoreWeaverChapter"
    return "LoreWeaverSpan"


def _node_key(node_type: str) -> str:
    if node_type == "CenterSpanCluster":
        return "cluster_id"
    if node_type == "Entity":
        return "entity_id"
    if node_type == "Chapter":
        return "chapter_id"
    return "span_id"


def _env_value(env_name: object) -> str | None:
    if not env_name:
        return None
    value = os.environ.get(str(env_name))
    return value or None

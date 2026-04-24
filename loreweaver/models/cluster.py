"""Cluster data models for the M1.5 graph skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CenterSpanCluster:
    cluster_id: str
    document_id: str
    center_span_id: str
    cluster_name: str
    cluster_type: str
    micro_summary: str
    member_span_ids: list[str]
    confidence: float
    status: str
    created_at: datetime


@dataclass(frozen=True)
class SpanEdge:
    edge_id: str
    document_id: str
    from_id: str
    to_id: str
    from_type: str
    to_type: str
    edge_type: str
    weight: float
    source: str
    created_at: datetime

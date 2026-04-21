"""Cluster data model placeholders for M1.5."""

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
    summary: str
    member_span_ids: list[str]
    confidence: float
    status: str
    created_at: datetime


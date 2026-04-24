"""Qdrant vector store adapter for M1.4."""

from __future__ import annotations

import os
import re
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.models.span import Span


@dataclass(frozen=True)
class VectorRecord:
    span: Span
    vector: list[float]
    embedding_text: str


@dataclass(frozen=True)
class VectorSearchResult:
    span_id: str
    score: float
    payload: dict[str, Any]


class QdrantVectorStore:
    """Small Qdrant wrapper that supports remote and local-path operation."""

    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        local_path: Path | None,
        distance: str,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.local_path = local_path
        self.distance = distance

    @classmethod
    def from_config(cls, storage_config: AppConfig, *, document_id: str) -> "QdrantVectorStore":
        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise RuntimeError(
                "The qdrant-client package is required for M1.4 vector indexing. "
                "Install optional M1 dependencies first."
            ) from error

        qdrant_config = storage_config.values.get("qdrant", {})
        url = _env_value(qdrant_config.get("url_env"))
        api_key = _env_value(qdrant_config.get("api_key_env"))
        collection_prefix = str(qdrant_config.get("collection_prefix", "loreweaver"))
        collection_name = build_collection_name(collection_prefix, document_id)
        distance = str(qdrant_config.get("distance", "cosine"))

        if url:
            client = QdrantClient(url=url, api_key=api_key)
            return cls(
                client=client,
                collection_name=collection_name,
                local_path=None,
                distance=distance,
            )

        local_path = Path(qdrant_config.get("local_path", "data/indexes/qdrant"))
        local_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(local_path))
        return cls(
            client=client,
            collection_name=collection_name,
            local_path=local_path,
            distance=distance,
        )

    def recreate_collection(self, *, vector_size: int) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as error:
            raise RuntimeError("qdrant-client is required for vector indexing.") from error

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=_qdrant_distance(self.distance, Distance),
            ),
        )

    def ensure_collection(self, *, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.recreate_collection(vector_size=vector_size)

    def upsert_records(self, records: list[VectorRecord], *, batch_size: int = 64) -> int:
        try:
            from qdrant_client.models import PointStruct
        except ImportError as error:
            raise RuntimeError("qdrant-client is required for vector indexing.") from error

        count = 0
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start : batch_start + batch_size]
            points = [
                PointStruct(
                    id=point_id_for_span(record.span.span_id),
                    vector=record.vector,
                    payload=_payload_for_record(record),
                )
                for record in batch
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            count += len(points)
        return count

    def search(self, query_vector: list[float], *, top_k: int) -> list[VectorSearchResult]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        points = getattr(response, "points", response)
        return [
            VectorSearchResult(
                span_id=str(point.payload.get("span_id", "")),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in points
        ]

    def retrieve_vectors(self, span_ids: list[str]) -> dict[str, list[float]]:
        if not span_ids or not self.client.collection_exists(self.collection_name):
            return {}
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id_for_span(span_id) for span_id in span_ids],
            with_payload=True,
            with_vectors=True,
        )
        vectors_by_span_id: dict[str, list[float]] = {}
        for point in points:
            payload = dict(getattr(point, "payload", {}) or {})
            span_id = str(payload.get("span_id", ""))
            vector = getattr(point, "vector", None)
            if isinstance(vector, dict):
                vector = next(iter(vector.values()), None)
            if span_id and isinstance(vector, list):
                vectors_by_span_id[span_id] = [float(value) for value in vector]
        return vectors_by_span_id

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def build_collection_name(prefix: str, document_id: str) -> str:
    raw = f"{prefix}_{document_id}_spans"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", raw)


def point_id_for_span(span_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"loreweaver:{span_id}"))


def _payload_for_record(record: VectorRecord) -> dict[str, Any]:
    span = record.span
    return {
        "span_id": span.span_id,
        "document_id": span.document_id,
        "chapter_id": span.chapter_id,
        "window_id": span.window_id,
        "micro_topic": span.micro_topic,
        "span_type": span.span_type,
        "micro_summary": span.micro_summary,
        "entities": span.entities,
        "topics": span.topics,
        "salience_score": span.salience_score,
        "span_start_idx": span.span_start_idx,
        "span_end_idx": span.span_end_idx,
        "locator_confidence": span.locator_confidence,
        "embedding_text_sha256": hashlib.sha256(record.embedding_text.encode("utf-8")).hexdigest(),
    }


def _qdrant_distance(distance: str, distance_enum: Any) -> Any:
    normalized = distance.lower()
    if normalized == "dot":
        return distance_enum.DOT
    if normalized == "euclid":
        return distance_enum.EUCLID
    if normalized == "manhattan":
        return distance_enum.MANHATTAN
    return distance_enum.COSINE


def _env_value(env_name: object) -> str | None:
    if not env_name:
        return None
    value = os.environ.get(str(env_name))
    if not value:
        return None
    return value

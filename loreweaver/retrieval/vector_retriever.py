"""Vector retrieval for M1.6."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.indexing.embeddings import (
    MockEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
    embedding_settings_from_configs,
)
from loreweaver.retrieval.models import RetrievalHit
from loreweaver.storage.qdrant_store import QdrantVectorStore
from loreweaver.storage.sqlite_store import SQLiteStore


def retrieve_vector(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    models_config: AppConfig,
    store: SQLiteStore,
    document_id: str,
    question: str,
    top_k: int,
    mock_embeddings: bool = False,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    settings = embedding_settings_from_configs(config=config, models_config=models_config)
    try:
        if mock_embeddings:
            client = MockEmbeddingClient(dimensions=settings.expected_dimensions or 8)
            effective_settings = replace(settings, provider="mock", model=f"mock::{settings.model}")
        else:
            client = OpenAICompatibleEmbeddingClient(settings)
            effective_settings = settings
        query_vector = client.embed_texts([question]).vectors[0]
        qdrant_store = QdrantVectorStore.from_config(storage_config, document_id=document_id)
    except Exception as error:
        return [], {
            "source": "vector",
            "status": "error",
            "count": 0,
            "model": settings.model,
            "mock_embeddings": mock_embeddings,
            "error": str(error),
        }

    try:
        try:
            results = qdrant_store.search(query_vector, top_k=top_k)
        except Exception as error:
            return [], {
                "source": "vector",
                "status": "error",
                "count": 0,
                "model": effective_settings.model,
                "error": str(error),
            }
    finally:
        qdrant_store.close()

    spans_by_id = {
        span.span_id: span for span in store.list_spans_by_ids(result.span_id for result in results)
    }
    hits = [
        RetrievalHit(
            span_id=result.span_id,
            source="vector",
            score=result.score,
            span=spans_by_id.get(result.span_id),
            metadata={
                "rank": index + 1,
                "payload": result.payload,
                "embedding_provider": effective_settings.provider,
                "embedding_model": effective_settings.model,
            },
        )
        for index, result in enumerate(results)
    ]
    return hits, {
        "source": "vector",
        "status": "ok",
        "count": len(hits),
        "model": effective_settings.model,
        "mock_embeddings": mock_embeddings,
    }

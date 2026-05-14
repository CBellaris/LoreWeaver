"""M1.4 indexing pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.indexing.embeddings import (
    build_embedding_input,
    embedding_cache_key,
)
from loreweaver.model_services import EmbeddingResult, ModelServiceFactory
from loreweaver.model_services.config import ModelServiceConfig, ProviderConfig
from loreweaver.models.span import Span
from loreweaver.progress import ProgressReporter
from loreweaver.storage.bm25_store import BM25Index, bm25_index_path
from loreweaver.storage.qdrant_store import QdrantVectorStore, VectorRecord
from loreweaver.storage.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class EmbeddedSpan:
    span: Span
    input_text: str
    vector: list[float]
    from_cache: bool


def build_m14_indexes(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    models_config: AppConfig,
    run_id: str,
    document_id: str | None = None,
    limit: int | None = None,
    mock_embeddings: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_index_tables()
    document = store.get_document(document_id)
    spans = store.list_spans(document.document_id, located_only=True)
    if limit is not None:
        spans = spans[:limit]
    if not spans:
        raise ValueError(
            f"No located spans found for document_id={document.document_id}. "
            "Run loreweaver extract first."
        )

    factory = ModelServiceFactory.from_configs(config=config, models_config=models_config)
    embedding_settings = factory.resolve("embedding")
    effective_settings = embedding_settings
    if mock_embeddings:
        client = factory.embedding("embedding", mock=True)
        effective_settings = replace(
            embedding_settings,
            provider=ProviderConfig(
                name="mock",
                adapter="mock",
                api_key_env=None,
                base_url=None,
            ),
            model=f"mock::{embedding_settings.model}",
        )
    else:
        client = factory.embedding("embedding")

    if progress is not None:
        progress.emit(
            "planned",
            stage="index.plan",
            label=f"Plan indexing for {len(spans)} spans",
            current=0,
            total=len(spans),
            unit="spans",
            detail={
                "document_id": document.document_id,
                "span_count": len(spans),
                "embedding_model": embedding_settings.model,
                "mock_embeddings": mock_embeddings,
            },
        )

    embedded_spans, embedding_usage = embed_spans(
        store=store,
        spans=spans,
        settings=effective_settings,
        client=client,
        config=config,
        progress=progress,
    )
    vector_size = len(embedded_spans[0].vector)
    if effective_settings.expected_dimensions and vector_size != effective_settings.expected_dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: expected {effective_settings.expected_dimensions}, "
            f"got {vector_size}"
        )

    qdrant_store = QdrantVectorStore.from_config(storage_config, document_id=document.document_id)
    try:
        qdrant_store.recreate_collection(vector_size=vector_size)
        vector_count = qdrant_store.upsert_records(
            [
                VectorRecord(
                    span=embedded.span,
                    vector=embedded.vector,
                    embedding_text=embedded.input_text,
                )
                for embedded in embedded_spans
            ],
            batch_size=effective_settings.batch_size,
        )
        qdrant_count = qdrant_store.count()
        qdrant_local_path = str(qdrant_store.local_path) if qdrant_store.local_path else None
        collection_name = qdrant_store.collection_name
    finally:
        qdrant_store.close()

    index_dir = Path(storage_config.values.get("bm25", {}).get("index_dir", "data/indexes"))
    bm25_index = BM25Index.from_spans(document_id=document.document_id, spans=spans)
    bm25_path = bm25_index.save(bm25_index_path(index_dir, document.document_id))

    report = {
        "run_id": run_id,
        "document": {
            "document_id": document.document_id,
            "title": document.title,
            "author": document.author,
        },
        "sqlite_path": str(storage_config.sqlite_path),
        "located_span_count": len(spans),
        "embedding": {
            "provider": effective_settings.provider.name,
            "model": effective_settings.model,
            "dimensions": vector_size,
            "batch_size": effective_settings.batch_size,
            "mock": mock_embeddings,
            "cache_hits": sum(1 for embedded in embedded_spans if embedded.from_cache),
            "cache_misses": sum(1 for embedded in embedded_spans if not embedded.from_cache),
            "input_tokens": embedding_usage["input_tokens"],
            "estimated_cost_yuan": round(embedding_usage["estimated_cost_yuan"], 8),
        },
        "qdrant": {
            "collection_name": collection_name,
            "local_path": qdrant_local_path,
            "upsert_count": vector_count,
            "collection_count": qdrant_count,
        },
        "bm25": {
            "index_path": str(bm25_path),
            "document_count": len(bm25_index.documents),
        },
    }

    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{run_id}_index_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.insert_index_report(run_id, document.document_id, report)

    if progress is not None:
        progress.emit(
            "completed",
            stage="index.completed",
            label="Indexing completed",
            current=len(spans),
            total=len(spans),
            unit="spans",
            status="completed",
            detail=report,
        )
    return report


def embed_spans(
    *,
    store: SQLiteStore,
    spans: list[Span],
    settings: ModelServiceConfig,
    client: Any,
    config: AppConfig,
    progress: ProgressReporter | None = None,
) -> tuple[list[EmbeddedSpan], dict[str, float]]:
    input_config = config.values.get("indexing", {}).get("embedding_input", {})
    include_key_quote = bool(input_config.get("include_key_quote", False))
    include_located_text = bool(input_config.get("include_located_text", False))

    inputs = [
        build_embedding_input(
            span,
            include_key_quote=include_key_quote,
            include_located_text=include_located_text,
        )
        for span in spans
    ]
    embedded_spans: list[EmbeddedSpan | None] = [None] * len(spans)
    missing: list[tuple[int, str, str, str]] = []

    for index, input_text in enumerate(inputs):
        cache_key, input_sha256 = embedding_cache_key(settings, input_text)
        cached_vector = store.get_embedding_cache(cache_key)
        if cached_vector is None:
            missing.append((index, input_text, cache_key, input_sha256))
            continue
        embedded_spans[index] = EmbeddedSpan(
            span=spans[index],
            input_text=input_text,
            vector=cached_vector,
            from_cache=True,
        )

    input_tokens = 0
    for batch_start in range(0, len(missing), settings.batch_size):
        batch = missing[batch_start : batch_start + settings.batch_size]
        if progress is not None:
            progress.emit(
                "embedding_batch_start",
                stage="index.embedding",
                label=f"Embed batch {batch_start // settings.batch_size + 1}",
                current=min(batch_start + len(batch), len(missing)),
                total=len(missing),
                unit="spans",
                detail={
                    "batch_index": batch_start // settings.batch_size + 1,
                    "batch_count": (len(missing) + settings.batch_size - 1) // settings.batch_size,
                    "batch_size": len(batch),
                },
            )
        response: EmbeddingResult = client.embed([item[1] for item in batch])
        if len(response.vectors) != len(batch):
            raise ValueError(
                f"Embedding response count mismatch: expected {len(batch)}, "
                f"got {len(response.vectors)}"
            )
        input_tokens += int(response.usage.get("input_tokens", 0) or 0)
        for (index, input_text, cache_key, input_sha256), vector in zip(batch, response.vectors):
            normalized_vector = [float(value) for value in vector]
            store.upsert_embedding_cache(
                cache_key=cache_key,
                provider=settings.provider.name,
                model=settings.model,
                input_sha256=input_sha256,
                input_text=input_text,
                vector=normalized_vector,
            )
            embedded_spans[index] = EmbeddedSpan(
                span=spans[index],
                input_text=input_text,
                vector=normalized_vector,
                from_cache=False,
            )

    finalized = [embedded for embedded in embedded_spans if embedded is not None]
    if len(finalized) != len(spans):
        raise ValueError("Internal embedding error: not all spans received vectors.")
    return finalized, {
        "input_tokens": float(input_tokens),
        "estimated_cost_yuan": (input_tokens / 1000.0) * settings.pricing.input_yuan_per_1k,
    }


def search_vector_index(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    models_config: AppConfig,
    query: str,
    document_id: str | None = None,
    top_k: int = 10,
    mock_embeddings: bool = False,
) -> dict[str, Any]:
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    document = store.get_document(document_id)
    factory = ModelServiceFactory.from_configs(config=config, models_config=models_config)
    settings = factory.resolve("embedding")
    client = factory.embedding("embedding", mock=mock_embeddings)
    query_vector = client.embed([query]).vectors[0]

    qdrant_store = QdrantVectorStore.from_config(storage_config, document_id=document.document_id)
    try:
        try:
            results = qdrant_store.search(query_vector, top_k=top_k)
        except ValueError as error:
            message = str(error)
            if "not found" not in message.lower():
                raise
            raise ValueError(
                "Vector index not found for this document. Run loreweaver index first."
            ) from error
    finally:
        qdrant_store.close()
    span_ids = [result.span_id for result in results]
    spans_by_id = {span.span_id: span for span in store.list_spans_by_ids(span_ids)}

    return {
        "document_id": document.document_id,
        "query": query,
        "top_k": top_k,
        "results": [
            {
                "rank": index + 1,
                "span_id": result.span_id,
                "score": result.score,
                "chapter_id": result.payload.get("chapter_id"),
                "span_start_idx": result.payload.get("span_start_idx"),
                "span_end_idx": result.payload.get("span_end_idx"),
                "summary": result.payload.get("summary"),
                "entities": result.payload.get("entities", []),
                "topics": result.payload.get("topics", []),
                "located_text": spans_by_id.get(result.span_id).located_text
                if result.span_id in spans_by_id
                else "",
            }
            for index, result in enumerate(results)
        ],
    }


def search_bm25_index(
    *,
    storage_config: AppConfig,
    query: str,
    document_id: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    document = store.get_document(document_id)
    index_dir = Path(storage_config.values.get("bm25", {}).get("index_dir", "data/indexes"))
    index_path = bm25_index_path(index_dir, document.document_id)
    if not index_path.exists():
        raise ValueError(f"BM25 index not found: {index_path}. Run loreweaver index first.")
    bm25_index = BM25Index.load(index_path)
    results = bm25_index.search(query, top_k=top_k)
    spans_by_id = {span.span_id: span for span in store.list_spans_by_ids(result.span_id for result in results)}

    return {
        "document_id": document.document_id,
        "query": query,
        "top_k": top_k,
        "index_path": str(index_path),
        "results": [
            {
                "rank": index + 1,
                "span_id": result.span_id,
                "score": result.score,
                "chapter_id": result.document.chapter_id,
                "span_start_idx": result.document.span_start_idx,
                "span_end_idx": result.document.span_end_idx,
                "summary": result.document.summary,
                "entities": result.document.entities,
                "topics": result.document.topics,
                "located_text": spans_by_id.get(result.span_id).located_text
                if result.span_id in spans_by_id
                else "",
            }
            for index, result in enumerate(results)
        ],
    }

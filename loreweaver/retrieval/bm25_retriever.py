"""BM25 retrieval for M1.6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.retrieval.models import RetrievalHit
from loreweaver.storage.bm25_store import BM25Index, bm25_index_path
from loreweaver.storage.sqlite_store import SQLiteStore


def retrieve_bm25(
    *,
    storage_config: AppConfig,
    store: SQLiteStore,
    document_id: str,
    question: str,
    top_k: int,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    index_dir = Path(storage_config.values.get("bm25", {}).get("index_dir", "data/indexes"))
    index_path = bm25_index_path(index_dir, document_id)
    if not index_path.exists():
        return [], {
            "source": "bm25",
            "status": "missing_index",
            "index_path": str(index_path),
            "count": 0,
        }

    bm25_index = BM25Index.load(index_path)
    results = bm25_index.search(question, top_k=top_k)
    spans_by_id = {
        span.span_id: span for span in store.list_spans_by_ids(result.span_id for result in results)
    }
    hits = [
        RetrievalHit(
            span_id=result.span_id,
            source="bm25",
            score=result.score,
            span=spans_by_id.get(result.span_id),
            metadata={"index_path": str(index_path), "rank": index + 1},
        )
        for index, result in enumerate(results)
    ]
    return hits, {
        "source": "bm25",
        "status": "ok",
        "index_path": str(index_path),
        "count": len(hits),
    }

"""M1.6 hybrid retrieval pipeline."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.logging import new_run_id
from loreweaver.retrieval.bm25_retriever import retrieve_bm25
from loreweaver.retrieval.graph_retriever import retrieve_graph
from loreweaver.retrieval.models import RerankResult, RetrievalHit, UnionCandidate
from loreweaver.retrieval.query_router import route_query
from loreweaver.retrieval.reranker import build_rerank_candidates, build_reranker
from loreweaver.retrieval.union import merge_retrieval_hits, union_report
from loreweaver.retrieval.vector_retriever import retrieve_vector
from loreweaver.storage.sqlite_store import SQLiteStore


def retrieve_m16(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    models_config: AppConfig,
    question: str,
    document_id: str | None = None,
    mock_embeddings: bool = False,
    mock_reranker: bool = False,
    no_reranker: bool = False,
) -> dict[str, Any]:
    run_id = new_run_id("retrieve")
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_index_tables()
    store.initialize_graph_tables()
    document = store.get_document(document_id)
    retrieval_config = config.values.get("retrieval", {})
    query_type = route_query(question)

    graph_hits, graph_report = retrieve_graph(
        store=store,
        document_id=document.document_id,
        question=question,
        query_type=query_type,
        cluster_top_k=int(retrieval_config.get("graph_cluster_top_k", 4)),
        span_per_cluster=int(retrieval_config.get("graph_span_per_cluster", 12)),
    )
    vector_hits, vector_report = retrieve_vector(
        config=config,
        storage_config=storage_config,
        models_config=models_config,
        store=store,
        document_id=document.document_id,
        question=question,
        top_k=int(retrieval_config.get("vector_top_k", 30)),
        mock_embeddings=mock_embeddings,
    )
    bm25_hits, bm25_report = retrieve_bm25(
        storage_config=storage_config,
        store=store,
        document_id=document.document_id,
        question=question,
        top_k=int(retrieval_config.get("bm25_top_k", 30)),
    )

    hits: list[RetrievalHit] = [*graph_hits, *vector_hits, *bm25_hits]
    source_counts = Counter(hit.source for hit in hits)
    candidates = merge_retrieval_hits(
        hits,
        question=question,
        max_candidates=int(retrieval_config.get("union_max_candidates", 80)),
    )
    chapters_by_id = {
        chapter.chapter_id: chapter for chapter in store.list_chapters(document.document_id)
    }
    rerank_candidates = build_rerank_candidates(candidates, chapters_by_id=chapters_by_id)
    reranker = build_reranker(
        models_config=models_config,
        mock=mock_reranker,
        disabled=no_reranker,
    )
    rerank_results = reranker.rerank(question, rerank_candidates)
    rerank_top_k = int(retrieval_config.get("rerank_top_k", 15))
    top_results = rerank_results[:rerank_top_k]
    candidates_by_id = {candidate.span_id: candidate for candidate in candidates}

    report = {
        "run_id": run_id,
        "query_id": run_id,
        "document_id": document.document_id,
        "question": question,
        "query_type": query_type,
        "sqlite_path": str(storage_config.sqlite_path),
        "retrieval": {
            "graph": graph_report,
            "vector": vector_report,
            "bm25": bm25_report,
            "union": union_report(candidates, source_counts=dict(source_counts)),
        },
        "reranker": {
            "provider": reranker.provider,
            "model": reranker.model,
            "input_count": len(rerank_candidates),
            "top_k": rerank_top_k,
        },
        "top_results": [
            _result_payload(result, candidates_by_id[result.span_id])
            for result in top_results
            if result.span_id in candidates_by_id
        ],
        "candidates": [_candidate_payload(candidate) for candidate in candidates],
    }
    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{run_id}_retrieval_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.insert_query_run(run_id, document.document_id, question, report)
    return report


def _result_payload(result: RerankResult, candidate: UnionCandidate) -> dict[str, Any]:
    span = candidate.span
    return {
        "rank": result.rank,
        "span_id": result.span_id,
        "rerank_score": result.score,
        "fused_score": candidate.fused_score,
        "sources": candidate.sources,
        "source_scores": candidate.source_scores,
        "normalized_scores": candidate.normalized_scores,
        "cluster_ids": candidate.metadata.get("cluster_ids", []),
        "chapter_id": span.chapter_id,
        "span_start_idx": span.span_start_idx,
        "span_end_idx": span.span_end_idx,
        "micro_topic": span.micro_topic,
        "span_type": span.span_type,
        "micro_summary": span.micro_summary,
        "entities": span.entities,
        "topics": span.topics,
        "key_quote": span.key_quote,
        "rerank_text_sha256": result.text_sha256,
    }


def _candidate_payload(candidate: UnionCandidate) -> dict[str, Any]:
    span = candidate.span
    return {
        "span_id": candidate.span_id,
        "fused_score": candidate.fused_score,
        "sources": candidate.sources,
        "source_scores": candidate.source_scores,
        "normalized_scores": candidate.normalized_scores,
        "cluster_ids": candidate.metadata.get("cluster_ids", []),
        "chapter_id": span.chapter_id,
        "span_start_idx": span.span_start_idx,
        "span_end_idx": span.span_end_idx,
        "micro_topic": span.micro_topic,
        "micro_summary": span.micro_summary,
    }

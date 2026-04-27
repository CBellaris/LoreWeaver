"""Evidence Pack assembly for M1.7."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.evidence.citation import make_citation_id, validate_citation_ids
from loreweaver.evidence.interval import (
    build_span_evidence_seeds,
    expand_seeds_to_intervals,
    interval_payload,
    merge_evidence_intervals,
    select_intervals_for_budget,
)
from loreweaver.logging import new_run_id
from loreweaver.models.evidence import EvidenceBlock, QueryEvidencePack
from loreweaver.storage.sqlite_store import SQLiteStore


def assemble_evidence_pack_from_retrieval_report(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    retrieval_report: dict[str, Any],
) -> dict[str, Any]:
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_extraction_tables()
    store.initialize_index_tables()
    store.initialize_graph_tables()
    store.initialize_evidence_tables()

    document_id = str(retrieval_report["document_id"])
    document = store.get_document(document_id)
    chapters = store.list_chapters(document.document_id)
    chapters_by_id = {chapter.chapter_id: chapter for chapter in chapters}
    normalized_text = Path(document.normalized_path).read_text(encoding="utf-8")
    evidence_config = config.values.get("evidence", {})

    top_results = [
        {"document_id": document.document_id, **dict(item)}
        for item in retrieval_report.get("top_results", [])
    ]
    seeds, seed_warnings = build_span_evidence_seeds(top_results)
    expanded, expand_warnings = expand_seeds_to_intervals(
        seeds,
        chapters_by_id=chapters_by_id,
        pre_context_chars=int(evidence_config.get("pre_context_chars", 300)),
        post_context_chars=int(evidence_config.get("post_context_chars", 500)),
    )
    merged = merge_evidence_intervals(
        expanded,
        merge_gap_chars=int(evidence_config.get("merge_gap_chars", 500)),
    )
    selected = select_intervals_for_budget(
        merged,
        max_evidence_chars=int(evidence_config.get("max_evidence_chars", 40000)),
        max_blocks=int(evidence_config.get("max_blocks", 12)),
    )
    evidence_blocks = _build_evidence_blocks(selected, normalized_text=normalized_text)
    validate_citation_ids([block.citation_id for block in evidence_blocks])

    query_id = str(retrieval_report.get("query_id") or new_run_id("evidence"))
    created_at = datetime.now(timezone.utc)
    pack = QueryEvidencePack(
        query_id=query_id,
        document_id=document.document_id,
        user_question=str(retrieval_report.get("question", "")),
        query_type=str(retrieval_report.get("query_type", "unknown")),
        retrieved_span_ids=[str(item.get("span_id")) for item in top_results if item.get("span_id")],
        cluster_ids=_unique_cluster_ids(top_results),
        merged_intervals=[interval_payload(interval) for interval in selected],
        evidence_blocks=[asdict(block) for block in evidence_blocks],
        retrieval_sources=_retrieval_sources_payload(retrieval_report, top_results),
        rerank_scores={
            str(item.get("span_id")): float(item.get("rerank_score", 0.0))
            for item in top_results
            if item.get("span_id")
        },
        token_estimate=_estimate_tokens(evidence_blocks),
        answer=None,
        created_at=created_at,
    )
    pack_payload = _pack_payload(pack)
    report = {
        "run_id": new_run_id("evidence"),
        "query_id": query_id,
        "document_id": document.document_id,
        "question": pack.user_question,
        "query_type": pack.query_type,
        "sqlite_path": str(storage_config.sqlite_path),
        "source_retrieval_report_path": retrieval_report.get("report_path"),
        "evidence_pack": pack_payload,
        "assembly": {
            "top_result_count": len(top_results),
            "valid_seed_count": len(seeds),
            "expanded_interval_count": len(expanded),
            "merged_interval_count": len(merged),
            "selected_interval_count": len(selected),
            "evidence_block_count": len(evidence_blocks),
            "evidence_chars": sum(len(block.text) for block in evidence_blocks),
            "token_estimate": pack.token_estimate,
            "warnings": [
                {"span_id": warning.span_id, "reason": warning.reason}
                for warning in [*seed_warnings, *expand_warnings]
            ],
            "config": {
                "pre_context_chars": int(evidence_config.get("pre_context_chars", 300)),
                "post_context_chars": int(evidence_config.get("post_context_chars", 500)),
                "merge_gap_chars": int(evidence_config.get("merge_gap_chars", 500)),
                "max_evidence_chars": int(evidence_config.get("max_evidence_chars", 40000)),
                "max_blocks": int(evidence_config.get("max_blocks", 12)),
            },
        },
    }

    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{query_id}_evidence_pack.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.insert_evidence_pack(pack, report=report)
    return report


def _build_evidence_blocks(
    intervals: list[Any],
    *,
    normalized_text: str,
) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    for index, interval in enumerate(intervals, start=1):
        text = normalized_text[interval.start_idx : interval.end_idx]
        blocks.append(
            EvidenceBlock(
                citation_id=make_citation_id(index),
                document_id=interval.document_id,
                chapter_id=interval.chapter_id,
                chapter_title=interval.chapter_title,
                start_idx=interval.start_idx,
                end_idx=interval.end_idx,
                text=text,
                source_span_ids=interval.source_span_ids,
                retrieval_sources=interval.retrieval_sources,
                rerank_score=interval.rerank_score,
            )
        )
    return blocks


def _pack_payload(pack: QueryEvidencePack) -> dict[str, Any]:
    payload = asdict(pack)
    payload["created_at"] = pack.created_at.isoformat()
    return payload


def _retrieval_sources_payload(
    retrieval_report: dict[str, Any],
    top_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "summary": retrieval_report.get("retrieval", {}),
        "by_span_id": {
            str(item.get("span_id")): list(item.get("sources", []))
            for item in top_results
            if item.get("span_id")
        },
    }


def _unique_cluster_ids(top_results: list[dict[str, Any]]) -> list[str]:
    cluster_ids: list[str] = []
    for item in top_results:
        for cluster_id in item.get("cluster_ids", []) or []:
            value = str(cluster_id)
            if value and value not in cluster_ids:
                cluster_ids.append(value)
    return cluster_ids


def _estimate_tokens(blocks: list[EvidenceBlock]) -> int:
    text_chars = sum(len(block.text) for block in blocks)
    citation_chars = sum(len(block.citation_id) + len(block.chapter_title) for block in blocks)
    return max(0, text_chars + citation_chars)

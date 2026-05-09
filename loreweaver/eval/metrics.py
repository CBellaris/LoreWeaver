"""Chapter-level retrieval metrics for M1.9."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from loreweaver.eval.question_set import EvalQuestion


DEFAULT_CUTOFFS = (1, 3, 5, 10, 20)


def chapter_ranking_from_retrieval_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse span-level retrieval results into a chapter-level ranking."""
    by_chapter: dict[str, dict[str, Any]] = {}
    for result in report.get("top_results", []):
        chapter_id = str(result.get("chapter_id") or "")
        if not chapter_id:
            continue
        score = float(result.get("rerank_score", result.get("fused_score", 0.0)) or 0.0)
        current = by_chapter.get(chapter_id)
        if current is None:
            by_chapter[chapter_id] = {
                "chapter_id": chapter_id,
                "score": score,
                "best_span_id": result.get("span_id"),
                "best_span_rank": result.get("rank"),
                "hit_count": 1,
                "sources": sorted(str(source) for source in result.get("sources", [])),
            }
            continue
        current["hit_count"] += 1
        current["sources"] = sorted(
            set(current["sources"]).union(str(source) for source in result.get("sources", []))
        )
        if score > current["score"]:
            current["score"] = score
            current["best_span_id"] = result.get("span_id")
            current["best_span_rank"] = result.get("rank")

    ranked = sorted(by_chapter.values(), key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def score_question(
    *,
    question: EvalQuestion,
    predicted_chapters: list[dict[str, Any]],
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    gold_by_id = {
        chapter.chapter_id: {
            "weight": chapter.weight,
            "relevance": chapter.relevance,
            "chapter_index": chapter.chapter_index,
            "facet": chapter.facet,
        }
        for chapter in question.expected_chapters
    }
    predicted_ids = [str(item["chapter_id"]) for item in predicted_chapters]

    metrics: dict[str, Any] = {
        "question_id": question.question_id,
        "profile": question.profile,
        "query_type": question.query_type,
        "gold_chapter_count": len(question.expected_chapters),
        "predicted_chapter_count": len(predicted_chapters),
    }
    for cutoff in cutoffs:
        predicted_ids_at_k = predicted_ids[:cutoff]
        predicted_set_at_k = set(predicted_ids_at_k)
        metrics[f"weighted_recall_at_{cutoff}"] = sum(
            gold["weight"]
            for chapter_id, gold in gold_by_id.items()
            if chapter_id in predicted_set_at_k
        )
        metrics[f"hit_at_{cutoff}"] = 1.0 if predicted_set_at_k.intersection(gold_by_id) else 0.0
        metrics[f"ndcg_at_{cutoff}"] = ndcg_at_k(
            predicted_ids,
            {
                chapter_id: float(gold["relevance"])
                for chapter_id, gold in gold_by_id.items()
            },
            cutoff,
        )
        metrics[f"core_recall_at_{cutoff}"] = core_recall_at_k(
            predicted_set_at_k,
            gold_by_id,
        )
        metrics[f"facet_coverage_at_{cutoff}"] = facet_coverage_at_k(
            predicted_set_at_k,
            gold_by_id,
            required_facets=question.required_facets,
        )
        metrics[f"noise_at_{cutoff}"] = noise_at_k(
            predicted_chapters[:cutoff],
            question=question,
        )

    metrics["mrr"] = reciprocal_rank(predicted_ids, set(gold_by_id))
    metrics["missed_weight_at_20"] = 1.0 - metrics.get("weighted_recall_at_20", 0.0)
    return metrics


def aggregate_scores(
    question_scores: list[dict[str, Any]],
    *,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    if not question_scores:
        raise ValueError("No question scores to aggregate.")

    return {
        "question_count": len(question_scores),
        "overall": _mean_metrics(question_scores, cutoffs=cutoffs),
        "by_profile": _grouped_means(question_scores, "profile", cutoffs=cutoffs),
        "by_query_type": _grouped_means(question_scores, "query_type", cutoffs=cutoffs),
    }


def reciprocal_rank(predicted_ids: list[str], gold_ids: set[str]) -> float:
    for index, chapter_id in enumerate(predicted_ids, start=1):
        if chapter_id in gold_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(predicted_ids: list[str], relevance_by_id: dict[str, float], k: int) -> float:
    dcg = 0.0
    for index, chapter_id in enumerate(predicted_ids[:k], start=1):
        relevance = relevance_by_id.get(chapter_id, 0.0)
        dcg += (2.0**relevance - 1.0) / math.log2(index + 1)

    ideal_relevances = sorted(relevance_by_id.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2.0**relevance - 1.0) / math.log2(index + 1)
        for index, relevance in enumerate(ideal_relevances, start=1)
    )
    if ideal_dcg <= 0:
        return 0.0
    return dcg / ideal_dcg


def core_recall_at_k(
    predicted_ids_at_k: set[str],
    gold_by_id: dict[str, dict[str, Any]],
) -> float:
    core_ids = {
        chapter_id
        for chapter_id, gold in gold_by_id.items()
        if float(gold.get("relevance", 0.0)) >= 3.0
    }
    if not core_ids:
        return 0.0
    return len(predicted_ids_at_k.intersection(core_ids)) / len(core_ids)


def facet_coverage_at_k(
    predicted_ids_at_k: set[str],
    gold_by_id: dict[str, dict[str, Any]],
    *,
    required_facets: list[str],
) -> float:
    required = {facet for facet in required_facets if facet}
    if not required:
        required = {
            str(gold.get("facet"))
            for gold in gold_by_id.values()
            if str(gold.get("facet") or "").strip()
        }
    if not required:
        return 0.0
    covered = {
        str(gold_by_id[chapter_id].get("facet"))
        for chapter_id in predicted_ids_at_k.intersection(gold_by_id)
        if str(gold_by_id[chapter_id].get("facet") or "").strip()
    }
    return len(covered.intersection(required)) / len(required)


def noise_at_k(predicted_chapters_at_k: list[dict[str, Any]], *, question: EvalQuestion) -> float:
    if not predicted_chapters_at_k:
        return 0.0
    negative_ids = {
        chapter.chapter_id
        for chapter in question.negative_chapters
        if chapter.chapter_id
    }
    negative_indexes = {
        chapter.chapter_index
        for chapter in question.negative_chapters
        if chapter.chapter_index > 0
    }
    if not negative_ids and not negative_indexes:
        return 0.0
    noisy = 0
    for chapter in predicted_chapters_at_k:
        chapter_id = str(chapter.get("chapter_id") or "")
        chapter_index = int(chapter.get("chapter_index") or 0)
        if chapter_id in negative_ids or chapter_index in negative_indexes:
            noisy += 1
    return noisy / len(predicted_chapters_at_k)


def _grouped_means(
    scores: list[dict[str, Any]],
    group_key: str,
    *,
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        grouped[str(score.get(group_key) or "unknown")].append(score)
    return {
        key: _mean_metrics(items, cutoffs=cutoffs)
        for key, items in sorted(grouped.items(), key=lambda item: item[0])
    }


def _mean_metrics(scores: list[dict[str, Any]], *, cutoffs: tuple[int, ...]) -> dict[str, float]:
    keys = ["mrr", "missed_weight_at_20"]
    for cutoff in cutoffs:
        keys.extend(
            [
                f"weighted_recall_at_{cutoff}",
                f"hit_at_{cutoff}",
                f"ndcg_at_{cutoff}",
                f"core_recall_at_{cutoff}",
                f"facet_coverage_at_{cutoff}",
                f"noise_at_{cutoff}",
            ]
        )
    return {
        key: round(sum(float(score.get(key, 0.0)) for score in scores) / len(scores), 6)
        for key in keys
    }

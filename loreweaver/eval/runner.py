"""Evaluation runner for M1.9 chapter-level recall."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from loreweaver.config import AppConfig
from loreweaver.eval.metrics import (
    DEFAULT_CUTOFFS,
    aggregate_scores,
    chapter_ranking_from_retrieval_report,
    score_question,
)
from loreweaver.eval.question_set import EvalQuestion, load_question_set
from loreweaver.logging import new_run_id
from loreweaver.progress import ProgressReporter
from loreweaver.retrieval.pipeline import retrieve_m16
from loreweaver.storage.sqlite_store import SQLiteStore

EVAL_RETRIEVAL_PROFILES: dict[str, dict[str, int]] = {
    "broad": {
        "graph_cluster_top_k": 8,
        "graph_span_per_cluster": 20,
        "vector_top_k": 80,
        "bm25_top_k": 80,
        "union_max_candidates": 200,
        "rerank_top_k": 50,
    },
    "mixed": {
        "graph_cluster_top_k": 8,
        "graph_span_per_cluster": 20,
        "vector_top_k": 80,
        "bm25_top_k": 80,
        "union_max_candidates": 200,
        "rerank_top_k": 50,
    },
}


def run_eval(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    models_config: AppConfig,
    question_set_path: str | Path,
    document_id: str | None = None,
    output_path: str | Path | None = None,
    limit: int | None = None,
    mock_embeddings: bool = False,
    mock_reranker: bool = False,
    no_reranker: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Run LoreWeaver retrieval against a question set and score chapter recall."""
    questions = load_question_set(question_set_path)
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise ValueError("No eval questions selected.")

    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    document = store.get_document(document_id)
    chapters_by_id = {
        chapter.chapter_id: chapter
        for chapter in store.list_chapters(document.document_id)
    }
    run_id = new_run_id("eval_run")
    if output_path is None:
        output_path = config.data_dir / "eval" / "runs" / f"{run_id}_predictions.jsonl"
    predictions_path = Path(output_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    if progress is not None:
        progress.emit(
            "planned",
            stage="eval.plan",
            label=f"Plan evaluation for {len(questions)} questions",
            current=0,
            total=len(questions),
            unit="questions",
            detail={
                "run_id": run_id,
                "document_id": document.document_id,
                "question_count": len(questions),
            },
        )

    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        if progress is not None:
            progress.emit(
                "question_start",
                stage="eval.question",
                label=f"Run {question.question_id}",
                current=index - 1,
                total=len(questions),
                unit="questions",
                detail={
                    "index": index,
                    "total": len(questions),
                    "question_id": question.question_id,
                    "question": question.question,
                },
            )
        retrieval_report = retrieve_m16(
            config=_config_for_question(config, question),
            storage_config=storage_config,
            models_config=models_config,
            question=question.question,
            document_id=document.document_id,
            mock_embeddings=mock_embeddings,
            mock_reranker=mock_reranker,
            no_reranker=no_reranker,
            progress=progress.child(command="retrieve") if progress is not None else None,
        )
        predicted_chapters = _attach_chapter_titles(
            chapter_ranking_from_retrieval_report(retrieval_report),
            chapters_by_id=chapters_by_id,
        )
        question_score = score_question(question=question, predicted_chapters=predicted_chapters)
        prediction = _prediction_payload(
            question=question,
            retrieval_report=retrieval_report,
            predicted_chapters=predicted_chapters,
            score=question_score,
        )
        predictions.append(prediction)
        scores.append(question_score)
        if progress is not None:
            progress.emit(
                "question_done",
                stage="eval.question",
                label=f"Scored {question.question_id}",
                current=index,
                total=len(questions),
                unit="questions",
                detail={
                    "index": index,
                    "total": len(questions),
                    "question_id": question.question_id,
                    "weighted_recall_at_20": question_score["weighted_recall_at_20"],
                    "facet_coverage_at_20": question_score["facet_coverage_at_20"],
                    "mrr": question_score["mrr"],
                },
            )

    predictions_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions) + "\n",
        encoding="utf-8",
    )
    summary = aggregate_scores(scores, cutoffs=DEFAULT_CUTOFFS)
    report = {
        "run_id": run_id,
        "document_id": document.document_id,
        "question_set_path": str(question_set_path),
        "predictions_path": str(predictions_path),
        "question_count": len(questions),
        "mock_embeddings": mock_embeddings,
        "mock_reranker": mock_reranker,
        "no_reranker": no_reranker,
        "metrics": summary,
    }
    report_path = predictions_path.with_suffix(".summary.json")
    failures_path = predictions_path.with_suffix(".failures.md")
    report["report_path"] = str(report_path)
    report["failures_path"] = str(failures_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failures_path.write_text(render_failure_report(predictions), encoding="utf-8")
    if progress is not None:
        progress.emit(
            "completed",
            stage="eval.completed",
            label="Evaluation completed",
            current=len(questions),
            total=len(questions),
            unit="questions",
            status="completed",
            detail={
                "predictions_path": str(predictions_path),
                "report_path": str(report_path),
                "failures_path": str(failures_path),
            },
        )
    return report


def summarize_eval_run(predictions_path: str | Path) -> dict[str, Any]:
    predictions = load_predictions(predictions_path)
    scores = [prediction["score"] for prediction in predictions]
    summary = aggregate_scores(scores, cutoffs=DEFAULT_CUTOFFS)
    report = {
        "predictions_path": str(predictions_path),
        "question_count": len(predictions),
        "metrics": summary,
    }
    report_path = Path(predictions_path).with_suffix(".summary.json")
    failures_path = Path(predictions_path).with_suffix(".failures.md")
    report["report_path"] = str(report_path)
    report["failures_path"] = str(failures_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failures_path.write_text(render_failure_report(predictions), encoding="utf-8")
    return report


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    prediction_path = Path(path)
    predictions = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not predictions:
        raise ValueError(f"No predictions found: {prediction_path}")
    return predictions


def render_failure_report(predictions: list[dict[str, Any]], *, limit: int = 50) -> str:
    ordered = sorted(
        predictions,
        key=lambda item: (
            float(item["score"].get("weighted_recall_at_20", 0.0)),
            float(item["score"].get("facet_coverage_at_20", 0.0)),
            float(item["score"].get("mrr", 0.0)),
        ),
    )
    lines = ["# LoreWeaver M1.9 Eval Failures", ""]
    for item in ordered[:limit]:
        score = item["score"]
        if score.get("weighted_recall_at_20", 0.0) >= 1.0:
            continue
        lines.extend(
            [
                f"## {item['question_id']}",
                "",
                f"question: {item['question']}",
                "",
                f"weighted_recall_at_20: {score.get('weighted_recall_at_20', 0.0):.3f}",
                f"facet_coverage_at_20: {score.get('facet_coverage_at_20', 0.0):.3f}",
                f"noise_at_20: {score.get('noise_at_20', 0.0):.3f}",
                f"mrr: {score.get('mrr', 0.0):.3f}",
                "",
                "gold:",
            ]
        )
        for gold in item["expected_chapters"]:
            lines.append(
                "- "
                f"ch{gold['chapter_index']:03d} "
                f"{gold['chapter_id']} "
                f"weight={gold['weight']:.3f} "
                f"relevance={gold['relevance']} "
                f"facet={gold.get('facet', '')} "
                f"{gold.get('reason', '')}"
            )
        lines.extend(["", "predicted:"])
        for predicted in item["predicted_chapters"][:10]:
            lines.append(
                "- "
                f"rank={predicted['rank']} "
                f"ch{predicted.get('chapter_index', 0):03d} "
                f"{predicted['chapter_id']} "
                f"score={predicted['score']:.6f} "
                f"span={predicted.get('best_span_id')}"
            )
        lines.extend(["", "---", ""])
    return "\n".join(lines)


def _prediction_payload(
    *,
    question: EvalQuestion,
    retrieval_report: dict[str, Any],
    predicted_chapters: list[dict[str, Any]],
    score: dict[str, Any],
) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "question": question.question,
        "answer": question.answer,
        "profile": question.profile,
        "query_type": question.query_type,
        "required_facets": question.required_facets,
        "expected_chapters": [
            {
                "chapter_id": chapter.chapter_id,
                "chapter_index": chapter.chapter_index,
                "weight": chapter.weight,
                "relevance": chapter.relevance,
                "facet": chapter.facet,
                "reason": chapter.reason,
            }
            for chapter in question.expected_chapters
        ],
        "negative_chapters": [
            {
                "chapter_id": chapter.chapter_id,
                "chapter_index": chapter.chapter_index,
                "reason": chapter.reason,
            }
            for chapter in question.negative_chapters
        ],
        "predicted_chapters": predicted_chapters,
        "score": score,
        "retrieval_report_path": retrieval_report.get("report_path"),
        "retrieval_query_id": retrieval_report.get("query_id"),
    }


def _attach_chapter_titles(
    predicted_chapters: list[dict[str, Any]],
    *,
    chapters_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    for item in predicted_chapters:
        chapter = chapters_by_id.get(item["chapter_id"])
        if chapter is None:
            continue
        item["chapter_index"] = chapter.chapter_index
        item["chapter_title"] = chapter.chapter_title
    return predicted_chapters


def _config_for_question(config: AppConfig, question: EvalQuestion) -> AppConfig:
    overrides = _retrieval_overrides(config, question.profile)
    if not overrides:
        return config
    values = deepcopy(config.values)
    retrieval = dict(values.get("retrieval", {}))
    retrieval.update(overrides)
    values["retrieval"] = retrieval
    return AppConfig(path=config.path, values=values)


def _retrieval_overrides(config: AppConfig, profile: str) -> dict[str, int]:
    configured = (
        config.values.get("eval", {})
        .get("retrieval_profiles", {})
        .get(profile, {})
    )
    if configured:
        return {key: int(value) for key, value in configured.items()}
    return EVAL_RETRIEVAL_PROFILES.get(profile, {})

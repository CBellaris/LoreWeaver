"""Evaluation question set loading for M1.9."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldChapter:
    chapter_id: str
    chapter_index: int
    weight: float
    relevance: float
    facet: str
    reason: str = ""


@dataclass(frozen=True)
class NegativeChapter:
    chapter_id: str
    chapter_index: int
    reason: str = ""


@dataclass(frozen=True)
class EvalQuestion:
    question_id: str
    question: str
    answer: str
    profile: str
    query_type: str
    expected_chapters: list[GoldChapter]
    required_facets: list[str] = field(default_factory=list)
    negative_chapters: list[NegativeChapter] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_question_set(path: str | Path) -> list[EvalQuestion]:
    """Load a JSONL M1.9 question set."""
    question_path = Path(path)
    questions: list[EvalQuestion] = []
    for line_number, raw_line in enumerate(
        question_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {question_path}:{line_number}") from error
        questions.append(eval_question_from_payload(payload, line_number=line_number))
    if not questions:
        raise ValueError(f"Question set is empty: {question_path}")
    return questions


def write_question_set(path: str | Path, questions: list[EvalQuestion]) -> None:
    """Write questions as deterministic UTF-8 JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(eval_question_to_payload(question), ensure_ascii=False, sort_keys=True)
        for question in questions
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def eval_question_from_payload(payload: dict[str, Any], *, line_number: int = 0) -> EvalQuestion:
    expected_raw = payload.get("expected_chapters", [])
    if not isinstance(expected_raw, list) or not expected_raw:
        location = f" line {line_number}" if line_number else ""
        raise ValueError(f"Question{location} must include non-empty expected_chapters.")

    expected = normalize_gold_weights([_gold_chapter_from_payload(item) for item in expected_raw])
    question_id = str(payload.get("question_id") or "").strip()
    question = str(payload.get("question") or "").strip()
    profile = str(payload.get("profile") or "").strip()
    query_type = str(payload.get("query_type") or "").strip()
    if not all([question_id, question, profile, query_type]):
        location = f" line {line_number}" if line_number else ""
        raise ValueError(
            f"Question{location} must include question_id, question, profile, "
            "and query_type."
        )

    required_facets = _string_list(payload.get("required_facets", []))
    if not required_facets:
        required_facets = _unique_facets(expected)

    negative_raw = payload.get("negative_chapters", [])
    if negative_raw is None:
        negative_raw = []
    if not isinstance(negative_raw, list):
        location = f" line {line_number}" if line_number else ""
        raise ValueError(f"Question{location} negative_chapters must be a list.")

    known_keys = {
        "question_id",
        "question",
        "answer",
        "profile",
        "query_type",
        "expected_chapters",
        "required_facets",
        "negative_chapters",
    }
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in known_keys and value is not None
    }
    return EvalQuestion(
        question_id=question_id,
        question=question,
        answer=str(payload.get("answer") or ""),
        profile=profile,
        query_type=query_type,
        expected_chapters=expected,
        required_facets=required_facets,
        negative_chapters=[
            _negative_chapter_from_payload(item)
            for item in negative_raw
            if isinstance(item, dict)
        ],
        metadata=metadata,
    )


def eval_question_to_payload(question: EvalQuestion) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    }
    payload.update(question.metadata)
    return payload


def normalize_gold_weights(chapters: list[GoldChapter]) -> list[GoldChapter]:
    """Normalize gold weights while preserving relevance and facet labels."""
    total_weight = sum(max(0.0, chapter.weight) for chapter in chapters)
    if total_weight <= 0:
        relevance_total = sum(max(0.0, chapter.relevance) for chapter in chapters)
        if relevance_total <= 0:
            equal = 1.0 / len(chapters)
            return [_with_weight(chapter, equal) for chapter in chapters]
        return [
            _with_weight(chapter, max(0.0, chapter.relevance) / relevance_total)
            for chapter in chapters
        ]
    return [
        _with_weight(chapter, max(0.0, chapter.weight) / total_weight)
        for chapter in chapters
    ]


def _gold_chapter_from_payload(payload: dict[str, Any]) -> GoldChapter:
    chapter_id = str(payload.get("chapter_id") or "").strip()
    if not chapter_id:
        raise ValueError("Gold chapter is missing chapter_id.")
    chapter_index = int(payload.get("chapter_index") or 0)
    weight = float(payload.get("weight", 0.0) or 0.0)
    relevance = float(payload.get("relevance", weight) or 0.0)
    facet = str(payload.get("facet") or "general").strip()
    return GoldChapter(
        chapter_id=chapter_id,
        chapter_index=chapter_index,
        weight=weight,
        relevance=relevance,
        facet=facet,
        reason=str(payload.get("reason") or ""),
    )


def _negative_chapter_from_payload(payload: dict[str, Any]) -> NegativeChapter:
    return NegativeChapter(
        chapter_id=str(payload.get("chapter_id") or "").strip(),
        chapter_index=int(payload.get("chapter_index") or 0),
        reason=str(payload.get("reason") or ""),
    )


def _with_weight(chapter: GoldChapter, weight: float) -> GoldChapter:
    return GoldChapter(
        chapter_id=chapter.chapter_id,
        chapter_index=chapter.chapter_index,
        weight=weight,
        relevance=chapter.relevance,
        facet=chapter.facet,
        reason=chapter.reason,
    )


def _unique_facets(chapters: list[GoldChapter]) -> list[str]:
    facets: list[str] = []
    for chapter in chapters:
        if chapter.facet and chapter.facet not in facets:
            facets.append(chapter.facet)
    return facets


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]

"""Reranker interface and providers for M1.6."""

from __future__ import annotations

import hashlib
from typing import Protocol

from loreweaver.config import AppConfig
from loreweaver.model_services import ModelServiceFactory
from loreweaver.models.chapter import Chapter
from loreweaver.retrieval.models import RerankCandidate, RerankResult, UnionCandidate
from loreweaver.storage.bm25_store import tokenize_for_bm25


class Reranker(Protocol):
    provider: str
    model: str

    def rerank(self, question: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        """Return one score per candidate, sorted by descending relevance."""


class MockReranker:
    provider = "mock"

    def __init__(self, model: str = "mock-reranker") -> None:
        self.model = model

    def rerank(self, question: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        query_tokens = set(tokenize_for_bm25(question))
        scored: list[tuple[RerankCandidate, float]] = []
        for candidate in candidates:
            doc_tokens = set(tokenize_for_bm25(candidate.text))
            overlap = len(query_tokens.intersection(doc_tokens)) / max(1, len(query_tokens))
            score = min(1.0, 0.65 * overlap + 0.35 * candidate.candidate.fused_score)
            scored.append((candidate, score))
        return _results_from_scores(scored, provider=self.provider, model=self.model)


class NoopReranker:
    provider = "noop"
    model = "union-fused-score"

    def rerank(self, question: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        del question
        scored = [(candidate, candidate.candidate.fused_score) for candidate in candidates]
        return _results_from_scores(scored, provider=self.provider, model=self.model)


class ServiceReranker:
    def __init__(self, client: object) -> None:
        self._client = client
        self.provider = str(getattr(client, "provider"))
        self.model = str(getattr(client, "model"))

    def rerank(self, question: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        if not candidates:
            return []
        service_result = self._client.rerank(  # type: ignore[attr-defined]
            query=question,
            documents=[candidate.text for candidate in candidates],
            top_n=len(candidates),
        )
        indexed_scores = [(score.index, score.score) for score in service_result.scores]
        scored = [
            (candidates[index], score)
            for index, score in indexed_scores
            if 0 <= index < len(candidates)
        ]
        seen = {index for index, _score in indexed_scores}
        for index, candidate in enumerate(candidates):
            if index not in seen:
                scored.append((candidate, 0.0))
        return _results_from_scores(scored, provider=self.provider, model=self.model)


def build_reranker(
    *,
    models_config: AppConfig,
    mock: bool = False,
    disabled: bool = False,
) -> Reranker:
    factory = ModelServiceFactory.from_configs(models_config=models_config)
    service_config = factory.resolve("reranker")
    if mock:
        return MockReranker(model=f"mock::{service_config.model or 'reranker'}")
    if disabled or not service_config.enabled:
        return NoopReranker()
    return ServiceReranker(factory.reranker("reranker"))


def build_rerank_candidates(
    candidates: list[UnionCandidate],
    *,
    chapters_by_id: dict[str, Chapter],
) -> list[RerankCandidate]:
    return [
        RerankCandidate(
            candidate=candidate,
            text=build_rerank_text(candidate, chapters_by_id=chapters_by_id),
        )
        for candidate in candidates
    ]


def build_rerank_text(
    candidate: UnionCandidate,
    *,
    chapters_by_id: dict[str, Chapter],
) -> str:
    span = candidate.span
    chapter = chapters_by_id.get(span.chapter_id)
    return "\n".join(
        part
        for part in [
            f"章节：{chapter.chapter_title if chapter else span.chapter_id}",
            f"摘要：{span.summary}",
            f"实体：{', '.join(span.entities)}",
            f"主题：{', '.join(span.topics)}",
            f"原文短引：{span.key_quote}",
        ]
        if part.strip()
    )


def _results_from_scores(
    scored: list[tuple[RerankCandidate, float]],
    *,
    provider: str,
    model: str,
) -> list[RerankResult]:
    ordered = sorted(scored, key=lambda item: item[1], reverse=True)
    return [
        RerankResult(
            span_id=candidate.candidate.span_id,
            score=round(float(score), 8),
            rank=index + 1,
            provider=provider,
            model=model,
            text_sha256=hashlib.sha256(candidate.text.encode("utf-8")).hexdigest(),
        )
        for index, (candidate, score) in enumerate(ordered)
    ]

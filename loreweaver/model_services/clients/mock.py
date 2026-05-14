"""Mock model service clients for local tests and no-API plumbing."""

from __future__ import annotations

import hashlib
import json
import math
import re

from loreweaver.model_services.types import (
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    JsonChatResult,
    RerankScore,
    RerankServiceResult,
)


class MockChatModel:
    provider = "mock"

    def __init__(self, model: str = "mock-chat") -> None:
        self.model = model

    def complete(self, request: ChatRequest) -> ChatResult:
        content = request.messages[-1]["content"] if request.messages else ""
        answer = content
        usage = {
            "input_tokens": estimate_tokens(content),
            "output_tokens": estimate_tokens(answer),
            "total_tokens": estimate_tokens(content) + estimate_tokens(answer),
        }
        return ChatResult(
            content=answer,
            usage=usage,
            provider=self.provider,
            model=self.model,
        )

    def complete_json(self, request: ChatRequest) -> JsonChatResult:
        result = self.complete(request)
        payload = json.loads(result.content)
        return JsonChatResult(
            payload=payload,
            content=result.content,
            usage=result.usage,
            provider=result.provider,
            model=result.model,
        )


class MockEmbeddingModel:
    provider = "mock"

    def __init__(self, *, model: str = "mock-embedding", dimensions: int = 8) -> None:
        if dimensions <= 0:
            raise ValueError("Mock embedding dimensions must be positive.")
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[hash_embedding(text, self.dimensions) for text in texts],
            usage={
                "input_tokens": sum(estimate_tokens(text) for text in texts),
                "total_tokens": sum(estimate_tokens(text) for text in texts),
            },
            provider=self.provider,
            model=self.model,
        )


class NoopRerankModel:
    provider = "noop"
    model = "input-order"

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> RerankServiceResult:
        del query
        limit = top_n or len(documents)
        return RerankServiceResult(
            scores=[RerankScore(index=index, score=0.0) for index in range(min(limit, len(documents)))],
            usage={},
            provider=self.provider,
            model=self.model,
        )


class MockRerankModel:
    provider = "mock"

    def __init__(self, model: str = "mock-reranker") -> None:
        self.model = model

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> RerankServiceResult:
        query_terms = set(_simple_terms(query))
        scored: list[RerankScore] = []
        for index, document in enumerate(documents):
            document_terms = set(_simple_terms(document))
            score = len(query_terms.intersection(document_terms)) / max(1, len(query_terms))
            scored.append(RerankScore(index=index, score=round(score, 8)))
        ordered = sorted(scored, key=lambda item: item.score, reverse=True)
        return RerankServiceResult(
            scores=ordered[: top_n or len(ordered)],
            usage={},
            provider=self.provider,
            model=self.model,
        )


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def hash_embedding(text: str, dimensions: int) -> list[float]:
    raw = bytearray()
    seed = text.encode("utf-8")
    counter = 0
    while len(raw) < dimensions:
        raw.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    vector = [(byte / 255.0) - 0.5 for byte in raw[:dimensions]]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _simple_terms(text: str) -> list[str]:
    compact = "".join(text.lower().split())
    terms = re.findall(r"[a-z0-9_]+", compact)
    terms.extend(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
    return [term for term in terms if term]


def extraction_payload_from_messages(messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
    user_text = messages[-1]["content"]
    match = re.search(r"<<<WINDOW_TEXT\n(?P<text>.*)\nWINDOW_TEXT>>>", user_text, re.S)
    window_text = match.group("text") if match else user_text
    compact = " ".join(window_text.split())
    first_start = compact[: min(40, len(compact))]
    first_end = compact[min(len(compact), 80) : min(len(compact), 120)] or compact[-40:]
    second_start = compact[min(len(compact), 40) : min(len(compact), 80)] or first_start
    second_end = compact[min(len(compact), 120) : min(len(compact), 160)] or first_end
    payload = {
        "spans": [
            {
                "span_type": "event",
                "summary": compact[:100] or "空窗口",
                "entities": [],
                "topics": ["mock_extraction"],
                "salience_score": 0.5,
                "start_anchor_quote": first_start,
                "end_anchor_quote": first_end,
                "key_quote": first_start,
            },
            {
                "span_type": "mystery_clue",
                "summary": compact[40:140] or compact[:100] or "空窗口",
                "entities": [],
                "topics": ["mock_extraction"],
                "salience_score": 0.45,
                "start_anchor_quote": second_start,
                "end_anchor_quote": second_end,
                "key_quote": second_start,
            },
        ]
    }
    raw = json.dumps(payload, ensure_ascii=False)
    usage = {
        "input_tokens": estimate_tokens(user_text),
        "output_tokens": estimate_tokens(raw),
        "total_tokens": estimate_tokens(user_text) + estimate_tokens(raw),
    }
    return raw, usage

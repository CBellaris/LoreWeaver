"""BM25 index adapter for M1.4."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from loreweaver.models.span import Span


@dataclass(frozen=True)
class BM25Document:
    span_id: str
    document_id: str
    chapter_id: str
    span_start_idx: int | None
    span_end_idx: int | None
    micro_summary: str
    entities: list[str]
    topics: list[str]
    key_quote: str
    text: str
    tokens: list[str]


@dataclass(frozen=True)
class BM25SearchResult:
    span_id: str
    score: float
    document: BM25Document


class BM25Index:
    def __init__(self, *, document_id: str, documents: list[BM25Document]) -> None:
        self.document_id = document_id
        self.documents = documents
        self._model = None

    @classmethod
    def from_spans(cls, *, document_id: str, spans: list[Span]) -> "BM25Index":
        return cls(
            document_id=document_id,
            documents=[_document_from_span(span) for span in spans],
        )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        documents = [
            BM25Document(
                span_id=item["span_id"],
                document_id=item["document_id"],
                chapter_id=item["chapter_id"],
                span_start_idx=item["span_start_idx"],
                span_end_idx=item["span_end_idx"],
                micro_summary=item["micro_summary"],
                entities=list(item["entities"]),
                topics=list(item["topics"]),
                key_quote=item["key_quote"],
                text=item["text"],
                tokens=list(item["tokens"]),
            )
            for item in payload["documents"]
        ]
        return cls(document_id=payload["document_id"], documents=documents)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "document_id": self.document_id,
            "documents": [asdict(document) for document in self.documents],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def search(self, query: str, *, top_k: int = 10) -> list[BM25SearchResult]:
        if top_k <= 0 or not self.documents:
            return []
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        model = self._bm25_model()
        scores = model.get_scores(tokens)
        ranked_indexes = sorted(
            range(len(self.documents)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        results: list[BM25SearchResult] = []
        for index in ranked_indexes:
            score = float(scores[index])
            overlap = len(set(tokens).intersection(self.documents[index].tokens))
            if score == 0 and overlap == 0:
                continue
            results.append(
                BM25SearchResult(
                    span_id=self.documents[index].span_id,
                    score=score,
                    document=self.documents[index],
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _bm25_model(self) -> object:
        if self._model is None:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as error:
                raise RuntimeError(
                    "The rank-bm25 package is required for M1.4 BM25 indexing. "
                    "Install optional M1 dependencies first."
                ) from error
            self._model = BM25Okapi([document.tokens for document in self.documents])
        return self._model


def bm25_index_path(index_dir: str | Path, document_id: str) -> Path:
    return Path(index_dir) / f"{document_id}_bm25.json"


def tokenize_for_bm25(text: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text):
        value = match.group(0)
        if _is_cjk(value):
            tokens.extend(value)
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
            tokens.extend(value[index : index + 3] for index in range(len(value) - 2))
        else:
            tokens.append(value.lower())
    return [token for token in tokens if token]


def _document_from_span(span: Span) -> BM25Document:
    text = "\n".join(
        part
        for part in [
            span.micro_summary,
            " ".join(span.entities),
            " ".join(span.topics),
            span.key_quote,
        ]
        if part
    )
    return BM25Document(
        span_id=span.span_id,
        document_id=span.document_id,
        chapter_id=span.chapter_id,
        span_start_idx=span.span_start_idx,
        span_end_idx=span.span_end_idx,
        micro_summary=span.micro_summary,
        entities=span.entities,
        topics=span.topics,
        key_quote=span.key_quote,
        text=text,
        tokens=tokenize_for_bm25(text),
    )


def _is_cjk(text: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in text)

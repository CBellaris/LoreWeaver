"""Structured extraction schemas for M1.3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except ImportError:  # pragma: no cover - exercised in minimal bootstrap envs.
    BaseModel = None  # type: ignore[assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]
    model_validator = None  # type: ignore[assignment]


SPAN_TYPES = {
    "dialogue_exchange",
    "relationship_signal",
    "location_lore",
    "faction_lore",
    "power_rule",
    "event",
    "mystery_clue",
    "object_lore",
    "scene_action",
    "other",
}


if BaseModel is not None:

    class SpanCandidatePayload(BaseModel):
        """LLM-facing payload for one Span candidate."""

        model_config = ConfigDict(extra="forbid")

        span_type: str = Field(default="other", max_length=40)
        summary: str = Field(min_length=1, max_length=500)
        entities: list[str] = Field(default_factory=list, max_length=30)
        topics: list[str] = Field(default_factory=list, max_length=20)
        salience_score: float = Field(ge=0.0, le=1.0)
        start_anchor_quote: str = Field(min_length=1, max_length=160)
        end_anchor_quote: str = Field(min_length=1, max_length=160)
        key_quote: str = Field(default="", max_length=260)

        @field_validator(
            "span_type",
            "summary",
            "start_anchor_quote",
            "end_anchor_quote",
            "key_quote",
        )
        @classmethod
        def _strip_text(cls, value: str) -> str:
            return value.strip()

        @field_validator("summary", "start_anchor_quote", "end_anchor_quote")
        @classmethod
        def _require_text(cls, value: str) -> str:
            if not value:
                raise ValueError("field must not be blank")
            return value

        @field_validator("span_type")
        @classmethod
        def _normalize_span_type(cls, value: str) -> str:
            normalized = value.strip() or "other"
            return normalized if normalized in SPAN_TYPES else "other"

        @field_validator("entities", "topics")
        @classmethod
        def _clean_list(cls, value: list[str]) -> list[str]:
            return _clean_string_list(value)

        def as_json_dict(self) -> dict[str, Any]:
            return self.model_dump(mode="json")


    class WindowExtractionPayload(BaseModel):
        """LLM-facing structured output for one candidate window."""

        model_config = ConfigDict(extra="forbid")

        spans: list[SpanCandidatePayload] = Field(default_factory=list, min_length=1)

        @model_validator(mode="after")
        def _require_spans(self) -> "WindowExtractionPayload":
            if not self.spans:
                raise ValueError("spans must contain at least one item")
            return self

        def as_json_dict(self) -> dict[str, Any]:
            return self.model_dump(mode="json")


    ExtractedSpanPayload = SpanCandidatePayload


    class ExtractionFailure(BaseModel):
        """Persistable failure record for retry/debug queues."""

        window_id: str
        stage: str
        reason: str
        attempt: int
        raw_output: str | None = None


else:

    @dataclass(frozen=True)
    class SpanCandidatePayload:
        """Small validation fallback used before optional dependencies are installed."""

        span_type: str = "other"
        summary: str = ""
        entities: list[str] = field(default_factory=list)
        topics: list[str] = field(default_factory=list)
        salience_score: float = 0.0
        start_anchor_quote: str = ""
        end_anchor_quote: str = ""
        key_quote: str = ""

        def __post_init__(self) -> None:
            summary = self.summary.strip()
            start_anchor = self.start_anchor_quote.strip()
            end_anchor = self.end_anchor_quote.strip()
            if not summary:
                raise ValueError("summary must not be blank")
            if not start_anchor:
                raise ValueError("start_anchor_quote must not be blank")
            if not end_anchor:
                raise ValueError("end_anchor_quote must not be blank")
            score = float(self.salience_score)
            if score < 0.0 or score > 1.0:
                raise ValueError("salience_score must be between 0 and 1")
            span_type = self.span_type.strip() or "other"
            if span_type not in SPAN_TYPES:
                span_type = "other"
            object.__setattr__(self, "span_type", span_type)
            object.__setattr__(self, "summary", summary)
            object.__setattr__(self, "salience_score", score)
            object.__setattr__(self, "start_anchor_quote", start_anchor)
            object.__setattr__(self, "end_anchor_quote", end_anchor)
            object.__setattr__(self, "key_quote", self.key_quote.strip())
            object.__setattr__(self, "entities", _clean_string_list(self.entities))
            object.__setattr__(self, "topics", _clean_string_list(self.topics))

        @classmethod
        def model_validate(cls, data: Any) -> "SpanCandidatePayload":
            if not isinstance(data, dict):
                raise ValueError("span candidate must be a JSON object")
            return cls(
                span_type=str(data.get("span_type", "other")),
                summary=str(data.get("summary", "")),
                entities=list(data.get("entities", [])),
                topics=list(data.get("topics", [])),
                salience_score=float(data.get("salience_score", 0.0)),
                start_anchor_quote=str(data.get("start_anchor_quote", "")),
                end_anchor_quote=str(data.get("end_anchor_quote", "")),
                key_quote=str(data.get("key_quote", "")),
            )

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            del mode
            return {
                "span_type": self.span_type,
                "summary": self.summary,
                "entities": self.entities,
                "topics": self.topics,
                "salience_score": self.salience_score,
                "start_anchor_quote": self.start_anchor_quote,
                "end_anchor_quote": self.end_anchor_quote,
                "key_quote": self.key_quote,
            }

        def as_json_dict(self) -> dict[str, Any]:
            return self.model_dump()


    @dataclass(frozen=True)
    class WindowExtractionPayload:
        spans: list[SpanCandidatePayload] = field(default_factory=list)

        @classmethod
        def model_validate(cls, data: Any) -> "WindowExtractionPayload":
            if not isinstance(data, dict):
                raise ValueError("payload must be a JSON object")
            spans = [
                SpanCandidatePayload.model_validate(item)
                for item in list(data.get("spans", []))
            ]
            if not spans:
                raise ValueError("spans must contain at least one item")
            return cls(spans=spans)

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            del mode
            return {"spans": [span.model_dump() for span in self.spans]}

        def as_json_dict(self) -> dict[str, Any]:
            return self.model_dump()


    ExtractedSpanPayload = SpanCandidatePayload


    @dataclass(frozen=True)
    class ExtractionFailure:
        window_id: str
        stage: str
        reason: str
        attempt: int
        raw_output: str | None = None


def _clean_string_list(value: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned

"""Deterministic local extraction client used by tests and dry runs."""

from __future__ import annotations

import json
import re

from loreweaver.extraction.types import ChatClient
from loreweaver.extraction.usage import estimate_tokens
from loreweaver.model_services import ChatRequest, ChatResult


class MockChatClient:
    """Deterministic local extractor for tests and dry M1.3 plumbing checks."""

    provider = "mock"
    model = "mock-extractor"

    def complete(self, request: ChatRequest) -> ChatResult:
        user_text = request.messages[-1]["content"]
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
                    "span_type": "progression",
                    "summary": compact[:100] or "空窗口",
                    "entities": [],
                    "salience_score": 0.5,
                    "start_anchor_quote": first_start,
                    "end_anchor_quote": first_end,
                    "key_quote": first_start,
                },
                {
                    "span_type": "exposition",
                    "summary": compact[40:140] or compact[:100] or "空窗口",
                    "entities": [],
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
        return ChatResult(
            content=raw,
            usage=usage,
            provider=self.provider,
            model=self.model,
        )

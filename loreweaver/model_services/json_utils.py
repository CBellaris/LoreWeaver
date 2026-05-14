"""Provider response normalization helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from loreweaver.model_services.errors import EmptyModelResponse, InvalidJsonResponse


def chat_content_from_response(response: Any, *, context: str = "chat completion") -> str:
    if response is None:
        raise EmptyModelResponse(f"{context} returned a null response from the provider.")
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise EmptyModelResponse(
            f"{context} returned no choices from the provider. Raw response: "
            f"{response_excerpt(response)}"
        )

    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(choice, dict):
        message_payload = choice.get("message") or {}
        if isinstance(message_payload, dict):
            content = message_payload.get("content")
    if not content:
        raise EmptyModelResponse(
            f"{context} returned an empty message content. Raw response: "
            f"{response_excerpt(response)}"
        )
    return str(content)


def usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
            "output_tokens": int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            ),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    return {
        "input_tokens": int(
            getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0
        ),
        "output_tokens": int(
            getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0
        ),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise InvalidJsonResponse("Model response does not contain a JSON object.")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise InvalidJsonResponse(str(error)) from error
    if not isinstance(payload, dict):
        raise InvalidJsonResponse("Model response must be a JSON object.")
    return payload


def response_excerpt(response: Any, *, limit: int = 4000) -> str:
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif hasattr(response, "dict"):
        payload = response.dict()
    else:
        payload = repr(response)
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"

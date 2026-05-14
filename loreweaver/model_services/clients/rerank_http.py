"""HTTP reranker client implementations."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from loreweaver.model_services.config import ModelServiceConfig
from loreweaver.model_services.errors import MissingApiKeyError
from loreweaver.model_services.types import RerankScore, RerankServiceResult


class HttpRerankClient:
    """Reranker for providers exposing a SiliconFlow-style /rerank endpoint."""

    def __init__(self, service_config: ModelServiceConfig) -> None:
        if not service_config.api_key_env:
            raise MissingApiKeyError(
                f"Provider {service_config.provider.name} does not define api_key_env"
            )
        api_key = os.environ.get(service_config.api_key_env)
        if not api_key:
            raise MissingApiKeyError(
                f"Missing API key environment variable: {service_config.api_key_env}"
            )
        self.service_config = service_config
        self.provider = service_config.provider.name
        self.model = service_config.model
        self._api_key = api_key
        self._base_url = (service_config.base_url or "https://api.siliconflow.cn/v1").rstrip("/")

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> RerankServiceResult:
        if not documents:
            return RerankServiceResult(
                scores=[],
                usage={},
                provider=self.provider,
                model=self.model,
            )
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
            "return_documents": False,
        }
        request = urllib.request.Request(
            f"{self._base_url}/rerank",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.service_config.timeout_seconds,
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        return RerankServiceResult(
            scores=_parse_rerank_response(response_payload),
            usage={},
            provider=self.provider,
            model=self.model,
        )


def _parse_rerank_response(payload: dict[str, Any]) -> list[RerankScore]:
    raw_results = payload.get("results", [])
    parsed: list[RerankScore] = []
    for item in raw_results:
        index = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if index is None or score is None:
            continue
        parsed.append(RerankScore(index=int(index), score=float(score)))
    return parsed

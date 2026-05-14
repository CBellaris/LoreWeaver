"""Evidence-grounded answer generation for M1.8."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from loreweaver.config import AppConfig
from loreweaver.evidence.assembler import assemble_evidence_pack_from_retrieval_report
from loreweaver.logging import new_run_id
from loreweaver.model_services import ChatRequest, ModelServiceFactory, resolve_model_service
from loreweaver.model_services.clients.openai_compatible import OpenAICompatibleClient
from loreweaver.model_services.config import ModelServiceConfig, ProviderConfig
from loreweaver.progress import ProgressReporter
from loreweaver.qa.prompts import build_answer_messages, build_repair_messages
from loreweaver.retrieval.pipeline import retrieve_m16
from loreweaver.storage.sqlite_store import SQLiteStore

_CITATION_REF_PATTERN = re.compile(r"\[E\d{3}\]")


class AnswerClient(Protocol):
    provider: str
    model: str

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        """Return answer text and token usage."""


@dataclass(frozen=True)
class AnswerValidation:
    ok: bool
    citations: list[str]
    errors: list[str]


class OpenAICompatibleAnswerClient:
    """OpenAI-compatible chat client for final evidence-grounded answers."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key_env: str,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        service_config = ModelServiceConfig(
            service="qa",
            capability="chat",
            provider=ProviderConfig(
                name=provider,
                adapter="openai_compatible",
                api_key_env=api_key_env,
                base_url=base_url,
            ),
            model=model,
        )
        self._client = OpenAICompatibleClient(service_config)

    @classmethod
    def from_config(cls, service_config: ModelServiceConfig) -> "OpenAICompatibleAnswerClient":
        api_key_env = service_config.api_key_env
        if not api_key_env:
            raise ValueError(f"Provider {service_config.provider.name} does not define api_key_env")
        return cls(
            provider=service_config.provider.name,
            model=service_config.model,
            api_key_env=api_key_env,
            base_url=service_config.base_url,
        )

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        result = self._client.complete(
            ChatRequest(messages=messages, temperature=temperature),
        )
        return result.content, result.usage


class MockAnswerClient:
    """Deterministic local answerer for tests and no-API M1.8 plumbing checks."""

    provider = "mock"

    def __init__(self, model: str = "mock-answerer") -> None:
        self.model = model

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        del temperature
        content = messages[-1]["content"]
        citation_ids = list(dict.fromkeys(_CITATION_REF_PATTERN.findall(content)))
        first = citation_ids[0] if citation_ids else ""
        if not first:
            answer = (
                "结论：证据不足以确认。\n\n"
                "证据：当前 Evidence Pack 没有可引用证据块。\n\n"
                "分析：无法在没有证据的情况下回答。\n\n"
                "不确定性：缺少可用证据。"
            )
        else:
            answer = (
                f"结论：基于当前证据，只能给出谨慎判断 {first}。\n\n"
                f"证据：{first} 提供了与问题直接相关的原文区间。\n\n"
                f"分析：该证据可支持一个低风险概括；更细的因果链需要更多证据 "
                f"{first}。\n\n"
                "不确定性：若问题要求跨章节演变，当前证据可能仍不完整。"
            )
        usage = {
            "input_tokens": _estimate_tokens(content),
            "output_tokens": _estimate_tokens(answer),
            "total_tokens": _estimate_tokens(content) + _estimate_tokens(answer),
        }
        return answer, usage


def ask_m18(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    models_config: AppConfig,
    question: str,
    document_id: str | None = None,
    mock_embeddings: bool = False,
    mock_reranker: bool = False,
    no_reranker: bool = False,
    mock_answer: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Run M1.8 end-to-end online QA: retrieve, assemble evidence, answer, persist."""
    if progress is not None:
        progress = progress.child(command="ask")
        progress.emit(
            "planned",
            stage="ask.plan",
            label="Plan answer workflow",
            current=0,
            total=3,
            unit="phases",
            detail={"question": question, "document_id": document_id},
        )
    retrieval_report = retrieve_m16(
        config=config,
        storage_config=storage_config,
        models_config=models_config,
        question=question,
        document_id=document_id,
        mock_embeddings=mock_embeddings,
        mock_reranker=mock_reranker,
        no_reranker=no_reranker,
        progress=progress.child(command="retrieve") if progress is not None else None,
    )
    evidence_report = assemble_evidence_pack_from_retrieval_report(
        config=config,
        storage_config=storage_config,
        retrieval_report=retrieval_report,
        progress=progress.child(command="evidence") if progress is not None else None,
    )
    return answer_evidence_pack(
        config=config,
        storage_config=storage_config,
        models_config=models_config,
        evidence_report=evidence_report,
        mock_answer=mock_answer,
        retrieval_report=retrieval_report,
        progress=progress,
    )


def answer_evidence_pack(
    *,
    config: AppConfig,
    storage_config: AppConfig,
    models_config: AppConfig,
    evidence_report: dict[str, Any],
    mock_answer: bool = False,
    retrieval_report: dict[str, Any] | None = None,
    answer_client: AnswerClient | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Generate and persist an answer from an already assembled Evidence Pack."""
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_index_tables()
    store.initialize_graph_tables()
    store.initialize_evidence_tables()

    pack = evidence_report["evidence_pack"]
    query_id = str(pack["query_id"])
    document_id = str(pack["document_id"])
    question = str(pack["user_question"])
    evidence_blocks = list(pack.get("evidence_blocks", []))
    valid_citations = [str(block["citation_id"]) for block in evidence_blocks]
    cluster_summaries = _cluster_summaries(store, document_id, pack.get("cluster_ids", []))

    qa_settings = _qa_settings(config=config, models_config=models_config)
    if answer_client is not None:
        client = answer_client
    elif mock_answer:
        client = MockAnswerClient(model=f"mock::{qa_settings['model']}")
    else:
        client = OpenAICompatibleAnswerClient.from_config(
            resolve_model_service(
                models_config=models_config,
                app_config=config,
                service="qa",
            )
        )

    messages = build_answer_messages(
        question=question,
        query_type=str(pack.get("query_type", "unknown")),
        cluster_summaries=cluster_summaries,
        evidence_blocks=evidence_blocks,
    )
    if progress is not None:
        progress.emit(
            "stage_start",
            stage="ask.answer",
            label="Generate cited answer",
            current=2,
            total=3,
            unit="phases",
            detail={
                "provider": client.provider,
                "model": client.model,
                "evidence_block_count": len(evidence_blocks),
                "mock": mock_answer,
            },
        )
    answer, usage = client.complete(messages=messages, temperature=qa_settings["temperature"])
    validation = validate_answer_citations(
        answer,
        valid_citation_ids=valid_citations,
        require_citations=qa_settings["require_citations"],
    )
    repaired = False
    repair_usage: dict[str, int] = {}
    if not validation.ok and evidence_blocks:
        if progress is not None:
            progress.emit(
                "stage_start",
                stage="ask.repair",
                label="Repair answer citations",
                current=2,
                total=3,
                unit="phases",
                detail={"errors": validation.errors},
            )
        repair_messages = build_repair_messages(
            question=question,
            answer=answer,
            validation_errors=validation.errors,
            evidence_blocks=evidence_blocks,
        )
        repaired_answer, repair_usage = client.complete(
            messages=repair_messages,
            temperature=qa_settings["temperature"],
        )
        repaired_validation = validate_answer_citations(
            repaired_answer,
            valid_citation_ids=valid_citations,
            require_citations=qa_settings["require_citations"],
        )
        if repaired_validation.ok:
            answer = repaired_answer
            validation = repaired_validation
            repaired = True

    pack["answer"] = answer
    report = {
        "run_id": new_run_id("answer"),
        "query_id": query_id,
        "document_id": document_id,
        "question": question,
        "query_type": pack.get("query_type", "unknown"),
        "sqlite_path": str(storage_config.sqlite_path),
        "source_retrieval_report_path": (
            retrieval_report or {}
        ).get("report_path")
        or evidence_report.get("source_retrieval_report_path"),
        "source_evidence_report_path": evidence_report.get("report_path"),
        "answer": answer,
        "answer_validation": {
            "ok": validation.ok,
            "citations": validation.citations,
            "errors": validation.errors,
            "valid_citation_ids": valid_citations,
            "repaired": repaired,
        },
        "qa": {
            "provider": client.provider,
            "model": client.model,
            "mock": mock_answer,
            "usage": usage,
            "repair_usage": repair_usage,
            "cluster_summary_count": len(cluster_summaries),
            "evidence_block_count": len(evidence_blocks),
            "temperature": qa_settings["temperature"],
        },
        "evidence_pack": pack,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    runs_dir = config.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{query_id}_answer_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    store.update_evidence_pack_answer(query_id, answer=answer, report=report, pack_payload=pack)
    store.insert_query_run(query_id, document_id, question, report)
    if progress is not None:
        progress.emit(
            "completed",
            stage="ask.completed",
            label="Answer workflow completed",
            current=3,
            total=3,
            unit="phases",
            status="completed",
            detail={
                "report_path": str(report_path),
                "citations_ok": validation.ok,
                "repaired": repaired,
            },
        )
    return report


def validate_answer_citations(
    answer: str,
    *,
    valid_citation_ids: list[str],
    require_citations: bool = True,
) -> AnswerValidation:
    citations = list(dict.fromkeys(_CITATION_REF_PATTERN.findall(answer)))
    valid_set = set(valid_citation_ids)
    errors: list[str] = []
    missing = [citation for citation in citations if citation not in valid_set]
    if missing:
        errors.append(f"unknown citations: {', '.join(missing)}")
    if require_citations and valid_citation_ids and not citations:
        errors.append("answer contains no citations")
    if require_citations and not valid_citation_ids:
        errors.append("evidence pack contains no citation ids")
    return AnswerValidation(ok=not errors, citations=citations, errors=errors)


def _qa_settings(*, config: AppConfig, models_config: AppConfig) -> dict[str, Any]:
    service_config = ModelServiceFactory.from_configs(
        config=config,
        models_config=models_config,
    ).resolve("qa")
    extra = service_config.extra or {}
    return {
        "provider": service_config.provider.name,
        "model": service_config.model or "gpt-4o",
        "api_key_env": str(service_config.api_key_env or "OPENAI_API_KEY"),
        "base_url": service_config.base_url,
        "temperature": float(service_config.temperature or 0),
        "require_citations": bool(extra.get("require_citations", True)),
    }


def _cluster_summaries(
    store: SQLiteStore,
    document_id: str,
    cluster_ids: list[str],
) -> list[dict[str, str]]:
    if not cluster_ids:
        return []
    requested = {str(cluster_id) for cluster_id in cluster_ids}
    summaries: list[dict[str, str]] = []
    for cluster in store.list_center_span_clusters(document_id):
        if cluster.cluster_id not in requested:
            continue
        summaries.append(
            {
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.cluster_name,
                "cluster_type": cluster.cluster_type,
                "summary": cluster.summary,
            }
        )
    return summaries


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

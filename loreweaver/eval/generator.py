"""Long-context LLM question generation for M1.9."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from loreweaver.config import AppConfig
from loreweaver.eval.corpus import load_corpus
from loreweaver.eval.question_set import (
    EvalQuestion,
    eval_question_from_payload,
    write_question_set,
)
from loreweaver.logging import new_run_id
from loreweaver.model_services import ChatRequest, resolve_model_service
from loreweaver.model_services.clients.openai_compatible import OpenAICompatibleClient
from loreweaver.model_services.config import ModelServiceConfig, ProviderConfig
from loreweaver.model_services.errors import EmptyModelResponse


class QuestionGeneratorClient(Protocol):
    provider: str
    model: str

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_output_tokens: int | None,
        json_response_format: bool,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Return a JSON object and usage metadata."""


class EmptyQuestionGenerationResponse(RuntimeError):
    """Raised when a provider returns HTTP 200 without a usable completion."""


@dataclass(frozen=True)
class EvalGeneratorSettings:
    provider: str
    model: str
    api_key_env: str
    base_url: str | None
    temperature: float
    max_output_tokens: int | None
    json_response_format: bool


class OpenAICompatibleQuestionGenerator:
    """OpenAI-compatible chat client for long-context eval question generation."""

    def __init__(self, settings: EvalGeneratorSettings) -> None:
        self.provider = settings.provider
        self.model = settings.model
        service_config = ModelServiceConfig(
            service="eval_question_generator",
            capability="chat",
            provider=ProviderConfig(
                name=settings.provider,
                adapter="openai_compatible",
                api_key_env=settings.api_key_env,
                base_url=settings.base_url,
            ),
            model=settings.model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            json_response_format=settings.json_response_format,
        )
        self._client = OpenAICompatibleClient(service_config)

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_output_tokens: int | None,
        json_response_format: bool,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        try:
            result = self._client.complete(
                ChatRequest(
                    messages=messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    response_format="json_object" if json_response_format else "none",
                )
            )
        except EmptyModelResponse as error:
            raise EmptyQuestionGenerationResponse(str(error)) from error
        content = result.content
        usage = result.usage
        return _parse_json_object(content), usage


def generate_question_set(
    *,
    config: AppConfig,
    models_config: AppConfig,
    corpus_path: str | Path,
    output_path: str | Path | None = None,
    question_count: int = 200,
    profile: str = "broad",
    max_output_tokens: int | None = None,
    client: QuestionGeneratorClient | None = None,
) -> dict[str, Any]:
    """Generate and persist a JSONL chapter-level recall question set."""
    corpus = load_corpus(corpus_path)
    settings = eval_generator_settings_from_config(models_config)
    generator = client or OpenAICompatibleQuestionGenerator(settings)
    profile = normalize_profile(profile)
    messages = build_generation_messages(
        corpus=corpus,
        question_count=question_count,
        profile=profile,
    )
    payload, usage = generator.complete_json(
        messages=messages,
        temperature=settings.temperature,
        max_output_tokens=max_output_tokens or settings.max_output_tokens,
        json_response_format=settings.json_response_format,
    )
    questions = _questions_from_generation_payload(payload)
    if output_path is None:
        document_id = corpus["document"]["document_id"]
        output_path = (
            config.data_dir
            / "eval"
            / "question_sets"
            / (
                f"{document_id}_ch{corpus['chapter_start']:03d}_"
                f"{corpus['chapter_end']:03d}_{profile}_v001.jsonl"
            )
        )
    write_question_set(output_path, questions)

    report = {
        "run_id": new_run_id("eval_generate"),
        "corpus_path": str(corpus_path),
        "question_set_path": str(output_path),
        "document_id": corpus["document"]["document_id"],
        "chapter_start": corpus["chapter_start"],
        "chapter_end": corpus["chapter_end"],
        "requested_question_count": question_count,
        "generated_question_count": len(questions),
        "profile": profile,
        "generator": {
            "provider": generator.provider,
            "model": generator.model,
            "temperature": settings.temperature,
            "max_output_tokens": max_output_tokens or settings.max_output_tokens,
            "usage": usage,
        },
    }
    report_path = Path(output_path).with_suffix(".generation_report.json")
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_generation_messages(
    *,
    corpus: dict[str, Any],
    question_count: int,
    profile: str = "broad",
) -> list[dict[str, str]]:
    chapters = corpus["chapters"]
    chapter_catalog = [
        {
            "chapter_id": chapter["chapter_id"],
            "chapter_index": chapter["chapter_index"],
            "chapter_title": chapter["chapter_title"],
            "char_count": chapter["char_count"],
        }
        for chapter in chapters
    ]
    source = "\n\n".join(
        "\n".join(
            [
                f"[CHAPTER {chapter['chapter_index']}]",
                f"chapter_id: {chapter['chapter_id']}",
                f"title: {chapter['chapter_title']}",
                "text:",
                chapter["text"],
            ]
        )
        for chapter in chapters
    )
    system = (
        "你是 LoreWeaver 的评测集构建器。你的任务是基于完整小说章节原文，"
        "生成用于测试章节级召回的高质量问题集。只输出 JSON 对象，不要输出解释文字。"
    )
    user = _profile_prompt(
        profile=normalize_profile(profile),
        question_count=question_count,
        chapter_catalog=chapter_catalog,
        source=source,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def normalize_profile(profile: str) -> str:
    value = profile.strip().lower()
    if value not in {"broad", "pinpoint", "mixed"}:
        raise ValueError("Eval generation profile must be one of: broad, pinpoint, mixed.")
    return value


def _profile_prompt(
    *,
    profile: str,
    question_count: int,
    chapter_catalog: list[dict[str, Any]],
    source: str,
) -> str:
    profile_rules = {
        "broad": """
本次只生成 broad 召回压力测试题。你不是在生成阅读理解题，而是在生成长篇小说知识库的广域召回评测题。

强制要求：
- 禁止生成“某章发生了什么”“某人为什么做某个单一动作”“某个怪物/物品叫什么”这类单点题。
- 每个问题都必须要求综合多个章节才能回答。
- 每题 expected_chapters 不得少于 6 章，推荐 8-20 章。
- 问题要广且泛，允许答案需要归纳、比较、总结、梳理。
- 每题 required_facets 不得少于 4 个。
- expected_chapters 中每个章节必须标注 facet，同一 facet 可以对应多章。
- 可以给出 0-5 个 negative_chapters，用于标注看似相关但不应作为核心证据的章节。

题型配额尽量接近：
- character_profile 25%
- worldbuilding_survey 25%
- geography_politics 15%
- system_mechanics 15%
- cross_work_comparison 10%
- timeline_arc 10%

优秀问题示例：
- 总结瑞贝卡这个角色目前展现出的性格、能力、处境和叙事作用。
- 本书设定中哪些地方借鉴了 DND 或经典西幻世界观？异同是什么？
- 总结目前洛伦大陆的地理、国家、边境和危险区域设定。
- 梳理高文对这个世界认知发生了哪些关键变化。
- 目前关于魔潮、畸变体、刚铎和神明体系的设定线索能拼出什么图景？
""",
        "pinpoint": """
本次只生成 pinpoint 局部召回题，用于测试明确事实和局部因果定位。

要求：
- 每题 expected_chapters 为 1-3 章。
- 问题应有较明确的答案边界。
- 不生成需要整合十几章的大型总结题。

题型配额尽量接近：
- plot_fact 30%
- character_relation 20%
- causality 20%
- timeline 15%
- foreshadowing 10%
- worldbuilding 5%
""",
        "mixed": """
本次生成 mixed 题集，其中 70% 为 broad 广域召回压力测试题，30% 为 pinpoint 局部召回题。
broad 题 expected_chapters 不得少于 6 章；pinpoint 题 expected_chapters 为 1-3 章。
""",
    }[profile].strip()

    return f"""
请基于下面的章节原文生成 {question_count} 个中文评测问题。

总体目标：
- 评估检索系统是否能召回相关章节，不评估最终回答文采。
- gold 标注粒度只到章节，不要标注 span。
- expected_chapters 必须使用给定的 chapter_id 和 chapter_index。
- weight 表示该章节对回答该问题的重要性，单题 expected_chapters 的 weight 总和必须为 1。
- relevance 使用 1-3：3=核心答案章节，2=重要补充章节，1=弱相关背景章节。
- facet 表示该章节支持问题答案的哪个方面，例如“身份处境”“地理边界”“魔法机制”“DND 相似点”。

{profile_rules}

输出 JSON 格式：
{{
  "questions": [
    {{
      "question_id": "q_0001",
      "question": "问题文本",
      "answer": "简短标准答案，用于人工抽样复核",
      "profile": "{profile}",
      "query_type": "character_profile | worldbuilding_survey | geography_politics | system_mechanics | cross_work_comparison | timeline_arc | plot_fact | character_relation | causality | foreshadowing",
      "required_facets": ["答案必须覆盖的方面"],
      "expected_chapters": [
        {{
          "chapter_id": "必须来自章节目录",
          "chapter_index": 1,
          "relevance": 3,
          "weight": 1.0,
          "facet": "该章节支持的答案方面",
          "reason": "为什么该章节相关"
        }}
      ],
      "negative_chapters": [
        {{
          "chapter_id": "可选，必须来自章节目录",
          "chapter_index": 1,
          "reason": "为什么看似相关但不应作为核心证据"
        }}
      ],
      "gold_confidence": 0.0
    }}
  ]
}}

章节目录：
{json.dumps(chapter_catalog, ensure_ascii=False)}

章节原文：
{source}
""".strip()


def eval_generator_settings_from_config(models_config: AppConfig) -> EvalGeneratorSettings:
    service_config = resolve_model_service(
        models_config=models_config,
        service="eval_question_generator",
    )
    return EvalGeneratorSettings(
        provider=service_config.provider.name,
        model=service_config.model or "deepseek-v4-pro",
        api_key_env=str(service_config.api_key_env or "DEEPSEEK_API_KEY"),
        base_url=service_config.base_url,
        temperature=(
            float(service_config.temperature)
            if service_config.temperature is not None
            else 0.2
        ),
        max_output_tokens=service_config.max_output_tokens,
        json_response_format=service_config.json_response_format,
    )


def _questions_from_generation_payload(payload: dict[str, Any]) -> list[EvalQuestion]:
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("Generator response must include a non-empty questions array.")
    questions = [
        eval_question_from_payload(question, line_number=index)
        for index, question in enumerate(raw_questions, start=1)
    ]
    seen: set[str] = set()
    unique_questions: list[EvalQuestion] = []
    for question in questions:
        if question.question_id in seen:
            raise ValueError(f"Duplicate question_id: {question.question_id}")
        seen.add(question.question_id)
        unique_questions.append(question)
    return unique_questions


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Generator response must be a JSON object.")
    return payload


def _chat_content_from_response(response: Any) -> str:
    if response is None:
        raise EmptyQuestionGenerationResponse(
            "Eval question generation returned a null response from the provider. "
            "This usually means the long-context request was accepted at HTTP level but "
            "the provider returned an empty/null body, often because the input/output "
            "budget was exceeded or the provider timed out internally. "
            "Try increasing --max-output-tokens, reducing --question-count, or splitting "
            "the corpus into smaller chapter ranges."
        )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise EmptyQuestionGenerationResponse(
            "Eval question generation returned no choices from the provider. "
            "This often means the long-context request was rejected, the provider hit an "
            "internal output/context limit, or it returned an error payload with HTTP 200. "
            f"Raw response: {_response_excerpt(response)}"
        )

    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(choice, dict):
        message_payload = choice.get("message") or {}
        if isinstance(message_payload, dict):
            content = message_payload.get("content")
    if not content:
        raise EmptyQuestionGenerationResponse(
            "Eval question generation returned an empty message content. "
            f"Raw response: {_response_excerpt(response)}"
        )
    return str(content)


def _usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _response_excerpt(response: Any, *, limit: int = 4000) -> str:
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


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)

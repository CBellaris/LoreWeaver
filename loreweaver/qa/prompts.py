"""Question answering prompts for M1.8."""

from __future__ import annotations

from typing import Any


ANSWER_SYSTEM_PROMPT = """你是 LoreWeaver 的证据解释器，不是自由创作模型。

你必须遵守：
1. 只能基于用户提供的 evidence_blocks 回答。
2. 每个关键结论都必须引用一个或多个形如 [E001] 的证据编号。
3. 如果证据不足，必须明确写“证据不足以确认”，不要补编剧情、章节或引用。
4. 推测必须标记为“推测”，并说明推测依据的证据编号。
5. 不得使用不存在的引用编号，不得编造章节名或原文。
6. 输出使用中文，结构包含：结论、证据、分析、不确定性。
"""


REPAIR_SYSTEM_PROMPT = """你是 LoreWeaver 的回答修复器。

只修复引用问题，不改变原回答的实质判断：
1. 删除不存在的引用编号。
2. 给没有引用的关键判断补上已有 evidence_blocks 中最相关的引用。
3. 如果无法用证据支持，改写为“证据不足以确认”。
4. 最终回答仍必须包含：结论、证据、分析、不确定性。
"""


def build_answer_messages(
    *,
    question: str,
    query_type: str,
    cluster_summaries: list[dict[str, str]],
    evidence_blocks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the QA model input from the already assembled Evidence Pack."""
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    f"用户问题：{question}",
                    f"问题类型：{query_type}",
                    _format_cluster_summaries(cluster_summaries),
                    _format_evidence_blocks(evidence_blocks),
                    "输出约束：必须使用 evidence_blocks 中已有的 [E###] 引用编号。",
                ]
            ),
        },
    ]


def build_repair_messages(
    *,
    question: str,
    answer: str,
    validation_errors: list[str],
    evidence_blocks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build one retry prompt for citation repair."""
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    f"用户问题：{question}",
                    f"引用校验错误：{'; '.join(validation_errors)}",
                    _format_evidence_blocks(evidence_blocks),
                    f"待修复回答：\n{answer}",
                ]
            ),
        },
    ]


def _format_cluster_summaries(cluster_summaries: list[dict[str, str]]) -> str:
    if not cluster_summaries:
        return "Cluster 摘要：无"
    lines = ["Cluster 摘要："]
    for cluster in cluster_summaries:
        lines.append(
            "- "
            f"{cluster.get('cluster_id', '')} "
            f"({cluster.get('cluster_type', 'unknown')}): "
            f"{cluster.get('cluster_name', '')} - {cluster.get('summary', '')}"
        )
    return "\n".join(lines)


def _format_evidence_blocks(evidence_blocks: list[dict[str, Any]]) -> str:
    if not evidence_blocks:
        return "evidence_blocks：无"
    lines = ["evidence_blocks："]
    for block in evidence_blocks:
        lines.append(
            "\n".join(
                [
                    f"{block['citation_id']}",
                    f"chapter: {block.get('chapter_title', '')}",
                    f"range: {block.get('start_idx')}-{block.get('end_idx')}",
                    f"text:\n{block.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(lines)

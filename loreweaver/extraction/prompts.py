"""Extraction prompts for M1.3."""

from __future__ import annotations

import json

from loreweaver.models.window import CandidateWindow


SPAN_TYPE_DEFINITIONS = {
    "progression": "时间顺序中的外部进展，包括行动、事件、冲突变化、历史沿革或阶段性结果。",
    "interaction": "人物、群体或势力之间的对话、回应、协商、冲突、关系动态。",
    "exposition": "对世界、制度、地点、规则、背景、物品、势力等的直接说明。",
    "reflection": "角色或叙述者的记忆、推理、判断、内心分析、认知变化。",
    "transition": "时间、地点、场景、叙述焦点或行动阶段之间的承接转换。",
    "mixed": "同一连续进展中多个功能强耦合，无法稳定判定主导功能。",
    "other": "结构性文本、残片、无法归类但需要覆盖的内容。",
}


SYSTEM_PROMPT = """你是 LoreWeaver 的结构化文本抽取器。
你的任务是把小说候选窗口切分成一组“基础 Span”。
所有抽取内容均来自给定窗口。
基础 Span 是读者复述章节脉络时会自然说出的一步连续进展，用来保证原文被完整建模。
每个基础 Span 可包含若干相邻段落、若干轮同目的对话，以及围绕同一进展发生的动作、反应、解释和补充信息。
基础 Span 按原文顺序首尾相接，边界优先服从场景目标变化、叙事阶段推进、说明对象变化、时间地点转移或主要冲突变化。
定位字段必须使用 start_anchor_quote 和 end_anchor_quote，它们都必须逐字摘自窗口原文。
输出必须是严格 JSON 对象。"""


def build_extraction_messages(
    window: CandidateWindow,
    *,
    anchor_min_chars: int = 4,
    anchor_max_chars: int = 80,
) -> list[dict[str, str]]:
    """Build chat-completion messages for one candidate window."""
    schema_hint = {
        "spans": [
            {
                "span_type": (
                    "progression | interaction | exposition | reflection | "
                    "transition | mixed | other"
                ),
                "summary": "1-2句中文摘要，概括这个基础 Span 对章节脉络的连续推进；需自包含主体、动作或设定对象",
                "entities": ["人物、地点、势力、物品、特殊术语；没有则为空数组"],
                "salience_score": "0到1之间的小数，表示切分完成后该基础 Span 的剧情/设定重要度",
                "start_anchor_quote": (
                    f"Span 开头附近 {anchor_min_chars}-{anchor_max_chars} 个原文字符，"
                    "必须逐字来自窗口"
                ),
                "end_anchor_quote": (
                    f"Span 结尾附近 {anchor_min_chars}-{anchor_max_chars} 个原文字符，"
                    "必须逐字来自窗口"
                ),
            }
        ]
    }
    span_type_hint = "\n".join(
        f"- {span_type}: {definition}"
        for span_type, definition in SPAN_TYPE_DEFINITIONS.items()
    )
    user_content = (
        "请将下面候选窗口完整切分成一组基础 Span。\n\n"
        f"window_id: {window.window_id}\n"
        f"document_id: {window.document_id}\n"
        f"chapter_id: {window.chapter_id}\n"
        f"window_global_range: [{window.window_start}, {window.window_end})\n\n"
        "拆分规则：\n"
        "- 输出 JSON 对象，顶层必须是 spans 数组。\n"
        "- 除 span_type 枚举值外，所有文本字段使用中文。\n"
        "- spans 按原文顺序排列，合起来完整覆盖整个窗口。\n"
        "- 相邻 Span 原则上首尾相接，形成一条连续的覆盖链。\n"
        "- 先按章节脉络划分连续进展，再为每个 Span 填写字段。\n"
        "- 同一场景目标、同一行动阶段、同一讨论目标、同一说明对象归入同一个基础 Span。\n"
        "- 一个基础 Span 可以同时包含行动、对话、人物反应、设定说明和关系信号；span_type 只标注该 Span 的主导文本功能。\n"
        "- 边界信号包括新的场景目标、叙事阶段、说明对象、时间地点或主要冲突。\n"
        f"- start_anchor_quote 与 end_anchor_quote 是定位锚点，取 Span 起止附近 {anchor_min_chars}-{anchor_max_chars} 个原文字符。\n"
        "- start_anchor_quote 应靠近 Span 起点，end_anchor_quote 应靠近 Span 终点，二者顺序必须正确。\n"
        "- summary 必须能独立表达 Span 核心内容，包含必要的主语或对象。\n\n"
        "span_type 枚举含义：\n"
        f"{span_type_hint}\n\n"
        "JSON字段要求：\n"
        f"{json.dumps(schema_hint, ensure_ascii=False, indent=2)}\n\n"
        "候选窗口原文：\n"
        "<<<WINDOW_TEXT\n"
        f"{window.text}\n"
        "WINDOW_TEXT>>>"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

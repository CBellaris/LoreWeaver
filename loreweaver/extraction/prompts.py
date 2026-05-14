"""Extraction prompts for M1.3."""

from __future__ import annotations

import json

from loreweaver.models.window import CandidateWindow


SYSTEM_PROMPT = """你是 LoreWeaver 的结构化文本抽取器。
你的任务不是做章节总结，而是把小说候选窗口切分成一组“基础 Span”。
所有抽取内容均来自给定窗口。
基础 Span 是按阅读理解自然形成的连续叙事单元，用来保证原文被完整建模。
每个基础 Span 应围绕同一场景目标、对话主题、连续动作链、说明性论述或设定解释。
基础 Span 按原文顺序首尾相接，边界优先服从语义转折、场景转移、话题切换、说明对象变化或行动目标变化。
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
                    "dialogue_exchange | relationship_signal | location_lore | "
                    "faction_lore | power_rule | event | mystery_clue | "
                    "object_lore | scene_action | other"
                ),
                "summary": "1-2句中文摘要，说明这个基础 Span 在叙事、场景或设定推进中的作用；需自包含主体、动作或设定点",
                "entities": ["人物、地点、势力、物品、特殊术语；没有则为空数组"],
                "salience_score": "0到1之间的小数，表示该基础 Span 的世界观/剧情信息密度；过渡、动作、气氛铺垫和闲聊通常为低分",
                "start_anchor_quote": (
                    f"Span 开头附近 {anchor_min_chars}-{anchor_max_chars} 个原文字符，"
                    "必须逐字来自窗口"
                ),
                "end_anchor_quote": (
                    f"Span 结尾附近 {anchor_min_chars}-{anchor_max_chars} 个原文字符，"
                    "必须逐字来自窗口"
                ),
                "key_quote": "可选：最能代表该 Span 核心信息的一小段原文",
            }
        ]
    }
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
        "- 基础 Span 的粒度是连续叙事 beat：同一场景目标、同一对话主题、连续动作链、连续设定解释归入同一个 Span。\n"
        "- 边界信号包括场景、时间或地点转移，对话目标或话题切换，说明对象变化，剧情因果推进进入下一步。\n"
        f"- start_anchor_quote 与 end_anchor_quote 是定位锚点，取 Span 起止附近 {anchor_min_chars}-{anchor_max_chars} 个原文字符。\n"
        "- start_anchor_quote 应靠近 Span 起点，end_anchor_quote 应靠近 Span 终点，二者顺序必须正确。\n"
        "- key_quote 只表达核心证据，保持短小。\n\n"
        "- summary 必须能独立表达 Span 核心内容，包含必要的主语或对象。\n\n"
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

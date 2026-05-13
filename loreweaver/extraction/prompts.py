"""Extraction prompts for M1.3."""

from __future__ import annotations

import json

from loreweaver.models.window import CandidateWindow


SYSTEM_PROMPT = """你是 LoreWeaver 的结构化文本抽取器。
你的任务不是做章节总结，而是在小说候选窗口中发现多个“微观 Span”。
必须只根据给定窗口回答，不要补充窗口之外的信息。
每个 Span 应尽量小，只覆盖一个具体信息点、互动、设定、地点描述、规则、伏笔或关系信号。
允许 Span 重叠、嵌套：同一段对话如果同时体现人物关系和地点设定，应拆成多个 Span。
定位字段必须使用 start_anchor_quote 和 end_anchor_quote，它们都必须逐字摘自窗口原文。
输出必须是严格 JSON 对象，不要使用 Markdown。"""


def build_extraction_messages(
    window: CandidateWindow,
    *,
    min_spans_per_window: int = 2,
    max_spans_per_window: int = 12,
    target_span_chars_min: int = 30,
    target_span_chars_max: int = 800,
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
                "micro_summary": "1-2句中文摘要，只说明这个小 Span 的信息点；需自包含主体、动作或设定点",
                "entities": ["人物、地点、势力、物品、特殊术语；没有则为空数组"],
                "topics": ["抽象主题，例如人物关系、地理线索、力量体系；没有则为空数组"],
                "salience_score": "0到1之间的小数，表示世界观/剧情信息密度",
                "start_anchor_quote": (
                    f"Span 开头附近 {anchor_min_chars}-{anchor_max_chars} 个原文字符，"
                    "必须逐字来自窗口"
                ),
                "end_anchor_quote": (
                    f"Span 结尾附近 {anchor_min_chars}-{anchor_max_chars} 个原文字符，"
                    "必须逐字来自窗口"
                ),
                "key_quote": "可选：最能代表该 Span 核心信息的一小段原文",
                "overlap_reason": "可选：如果它与其他 Span 重叠，说明为什么需要保留",
            }
        ]
    }
    user_content = (
        "请从下面候选窗口中拆分多个微观 Span。\n\n"
        f"window_id: {window.window_id}\n"
        f"document_id: {window.document_id}\n"
        f"chapter_id: {window.chapter_id}\n"
        f"window_global_range: [{window.window_start}, {window.window_end})\n\n"
        "拆分规则：\n"
        "- 输出 JSON 对象，顶层必须是 spans 数组。\n"
        "- 除 span_type 枚举值外，所有文本字段必须使用中文，不要中英混写。\n"
        f"- 尽量输出 {min_spans_per_window}-{max_spans_per_window} 个 Span；如果窗口很短，至少输出 1 个。\n"
        f"- 每个 Span 目标长度约 {target_span_chars_min}-{target_span_chars_max} 字符；短设定点可以接近下限。\n"
        "- 不要把整个窗口压缩成 1 个大总结，除非窗口本身只有一个很短的话题。\n"
        "- 一个 Span 只覆盖一个小话题，宁可拆小，不要把地点设定、人物关系、行动事件混在一起。\n"
        "- 如果一个候选 Span 会超过约 500 字符，通常应继续拆成更小的信息点；只有单一连续说明无法拆分时才保留长 Span。\n"
        "- 优先覆盖高价值信息：人物初次登场、身份变化、能力/规则、地点/势力历史、重要物品、异常现象、伏笔、关系变化。\n"
        "- 普通动作过渡可以跳过；但如果动作揭示身份、关系、规则或剧情转折，应抽成 Span。\n"
        "- 允许重叠：例如一段对话可同时产生“对话互动 Span”和“地点设定 Span”。\n"
        f"- start_anchor_quote 与 end_anchor_quote 是定位锚点，不需要覆盖完整 Span；建议 {anchor_min_chars}-50 字符，最长不要超过 {anchor_max_chars} 字符，且必须逐字来自窗口。\n"
        "- start_anchor_quote 应靠近 Span 起点，end_anchor_quote 应靠近 Span 终点，二者顺序必须正确。\n"
        "- key_quote 只表达核心证据，不承担完整定位；不要把整段 Span 塞进 key_quote。\n\n"
        "- micro_summary 必须能独立表达 Span 核心信息，不要依赖标题字段补充主语或对象。\n\n"
        "拆分示例：如果 A 对 B 说“闲聊内容。XX森林你知道吗？那里......”，应拆为：\n"
        "1. A 与 B 的对话互动 Span，覆盖 A 开口到该轮对话结束；\n"
        "2. XX森林设定 Span，只覆盖关于 XX森林的描述；\n"
        "3. 若闲聊体现 A/B 相处方式，可另拆人物关系 Span。\n\n"
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

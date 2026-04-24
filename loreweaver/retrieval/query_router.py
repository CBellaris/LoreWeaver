"""Lightweight query router for M1.6."""

from __future__ import annotations


QUERY_TYPES = (
    "character_relation",
    "faction_history",
    "location",
    "power_system",
    "timeline",
    "unknown",
)


def route_query(question: str) -> str:
    """Classify a user question into the small M1.6 retrieval buckets."""
    text = question.lower()
    keyword_map = {
        "character_relation": (
            "关系",
            "人物",
            "角色",
            "主角",
            "同伴",
            "敌人",
            "relationship",
            "character",
        ),
        "faction_history": (
            "势力",
            "国家",
            "家族",
            "组织",
            "派系",
            "历史",
            "faction",
            "kingdom",
            "family",
        ),
        "location": (
            "地点",
            "地理",
            "城市",
            "大陆",
            "地图",
            "遗迹",
            "location",
            "place",
            "map",
        ),
        "power_system": (
            "力量",
            "魔法",
            "法师",
            "能力",
            "规则",
            "体系",
            "power",
            "magic",
        ),
        "timeline": (
            "时间线",
            "演变",
            "变化",
            "变迁",
            "先后",
            "timeline",
            "chronology",
        ),
    }
    for query_type, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return query_type
    return "unknown"

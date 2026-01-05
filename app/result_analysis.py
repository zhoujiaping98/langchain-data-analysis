from __future__ import annotations

import json
from typing import Sequence

from .llm import get_llm


def _rows_preview(rows: Sequence[dict], limit: int = 50) -> list[dict]:
    if not rows:
        return []
    return list(rows[:limit])


def analyze_query_result(query: str, sql: str, rows: Sequence[dict]) -> str:
    if not rows:
        return "本次查询没有返回数据。"

    preview = _rows_preview(rows, limit=50)
    prompt = (
        "你是数据分析助手，请根据用户问题、SQL 和查询结果给出简明分析。\n"
        "要求：中文输出；3-6条要点；不编造数据；若样本有限请说明。\n\n"
        f"用户问题：{query}\n\n"
        f"SQL：\n{sql}\n\n"
        "结果样本（JSON数组）：\n"
        f"{json.dumps(preview, ensure_ascii=False)}\n"
    )

    try:
        llm = get_llm()
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return text.strip()
    except Exception:
        return "结果分析失败，请稍后重试。"

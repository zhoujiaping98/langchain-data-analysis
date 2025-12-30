from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field
import sqlglot
from sqlglot import exp

from .assets import get_metric
from .config import settings
from .db import explain, fetch_all
from .layer2_risk import assess_post_risk, validate_select_only
from .llm import get_llm
from .sql_compiler import compile_metric_sql
from .time_parser import infer_time_range, TimeRange


class QueryPlan(BaseModel):
    primary_metric: str | None = Field(default=None, description="metric_key if known")
    dimensions: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    limit: int = 1000


@dataclass
class ExecutionPlan:
    sql: str
    plan: QueryPlan
    explain: list[dict]
    post_risk: dict
    rows: list[dict]


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    return json.loads(m.group(0))


def _filter_columns(expr: str) -> set[str]:
    if not expr:
        return set()
    try:
        ast = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {expr}", dialect="mysql")
    except Exception:
        return set()
    return {c.name.lower() for c in ast.find_all(exp.Column) if getattr(c, "name", None)}


def _normalize_filters(filters: list[str], metric, time_range: TimeRange | None) -> list[str]:
    if not filters:
        return []
    default_filters = (metric.default_filters or "").lower()
    time_cols = {metric.time_column.lower(), "dt"}
    out: list[str] = []
    for f in filters:
        f_lower = f.lower()
        cols = _filter_columns(f)
        if cols & time_cols:
            continue
        if "status" in default_filters and "status" in f_lower and "paid" in f_lower and "paid" in default_filters:
            continue
        out.append(f)
    return out


def plan_with_llm(query: str, hinted_metric: str | None, allowed_dims: list[str]) -> QueryPlan:
    llm = get_llm()
    tr = infer_time_range(query)

    sys = (
        "你是一个业务查询解析助手。\n"
        "任务：把用户问题解析为 JSON 的查询意图，不要输出 SQL。\n"
        "只输出 JSON，不要解释。\n"
        "字段：primary_metric（可选）, dimensions, filters, limit。\n"
        "dimensions 必须从允许集合中选择；如果不确定，输出空数组。\n"
        "filters 数组中的每一项是 SQL WHERE 的布尔表达式片段（例如 status='paid'）。\n"
    )
    user = {
        "query": query,
        "hinted_metric": hinted_metric,
        "allowed_dimensions": allowed_dims,
        "time_range_hint": tr.label if tr else None,
    }
    resp = llm.invoke(sys + "\n输入:\n" + json.dumps(user, ensure_ascii=False))
    txt = resp.content if hasattr(resp, "content") else str(resp)
    raw = _extract_json(txt) or {}
    if raw.get("limit") is None:
        raw.pop("limit", None)
    qp = QueryPlan(**raw)

    if qp.primary_metric is None:
        qp.primary_metric = hinted_metric
    qp.dimensions = [d for d in qp.dimensions if d in set(allowed_dims)]
    qp.limit = min(int(qp.limit or 1000), settings.max_rows)
    return qp


def repair_sql_with_llm(sql: str, error: str) -> str:
    llm = get_llm()
    prompt = (
        "你是 SQL 修复助手。目标：修复 SQL 使其可在 MySQL 执行。\n"
        "要求：只允许 SELECT；不得包含任何写操作；必须保留原意；尽量最小改动。\n"
        f"错误信息: {error}\n"
        f"原 SQL:\n{sql}\n"
        "输出：只输出修复后的 SQL，不要解释。"
    )
    resp = llm.invoke(prompt)
    txt = resp.content if hasattr(resp, "content") else str(resp)
    txt = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", txt.strip())
    return txt.strip()


def controlled_text_to_sql(query: str, hinted_metric: str | None = None) -> ExecutionPlan:
    # 必须有时间范围：否则追问（可按需要改成默认最近 N 天）
    tr = infer_time_range(query)
    if not tr:
        raise ValueError("缺少时间范围（例如：上周/上个月/最近30天/2025-01-01~2025-02-01）。")

    metric_key = hinted_metric
    metric = get_metric(metric_key) if metric_key else None
    if not metric:
        raise ValueError("无法识别指标：请先在语义层定义指标口径，或在问题中明确指标名称。")

    qp = plan_with_llm(query, hinted_metric=metric.metric_key, allowed_dims=metric.allowed_dims)
    qp.filters = _normalize_filters(qp.filters, metric, tr)
    if hinted_metric and any(k in query.lower() for k in ["金额", "总额", "总金额", "gmv", "收入", "营收"]):
        qp.primary_metric = hinted_metric

    sql = compile_metric_sql(
        metric=metric,
        time_range=tr,
        dimensions=qp.dimensions,
        filters=qp.filters,
        limit=qp.limit,
    )

    # validate/repair loop
    for _ in range(settings.max_repair_rounds + 1):
        ok, reasons = validate_select_only(sql)
        if ok:
            break
        sql = repair_sql_with_llm(sql, "; ".join(reasons))

    post = assess_post_risk(sql)
    if post.action == "block":
        raise ValueError("SQL 风险过高，已阻止执行：" + "; ".join(post.reasons))

    exp_rows = explain(sql)
    rows = fetch_all(sql)
    return ExecutionPlan(sql=sql, plan=qp, explain=exp_rows, post_risk=post.__dict__, rows=rows)

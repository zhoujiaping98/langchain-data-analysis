from __future__ import annotations

from dataclasses import dataclass

from .assets import get_metric
from .db import explain, fetch_all
from .layer2_risk import assess_post_risk
from .layer3_controlled_t2sql import plan_with_llm, _normalize_filters
from .sql_compiler import compile_metric_sql
from .time_parser import infer_time_range


@dataclass
class TemplateResult:
    sql: str
    explain: list[dict]
    rows: list[dict]
    post_risk: dict


def run_template(metric_key: str, query: str, limit: int = 1000) -> TemplateResult:
    metric = get_metric(metric_key)
    if not metric:
        raise ValueError(f"未知指标: {metric_key}")

    tr = infer_time_range(query)
    if not tr:
        raise ValueError("模板查询需要明确时间范围（例如：上周/上个月/2025-01-01~2025-01-31）。")

    # 尝试解析查询中的过滤条件和维度
    allowed_dims = metric.allowed_dims if metric else []

    try:
        # 使用LLM解析查询计划（包括过滤条件和维度）
        plan = plan_with_llm(query, metric_key, allowed_dims)
        normalized_filters = _normalize_filters(plan.filters, metric, tr)

        # 构建维度列表
        dimensions = plan.dimensions if plan.dimensions else []

    except Exception:
        # 如果解析失败，回退到基础版本（只有时间过滤）
        dimensions = []
        normalized_filters = []

    sql = compile_metric_sql(metric=metric, time_range=tr, dimensions=dimensions, filters=normalized_filters,
                             limit=limit)
    post = assess_post_risk(sql)
    if post.action == "block":
        raise ValueError("SQL 风险过高，已阻止执行：" + "; ".join(post.reasons))

    exp_rows = explain(sql)
    rows = fetch_all(sql)
    return TemplateResult(sql=sql, explain=exp_rows, rows=rows, post_risk=post.__dict__)
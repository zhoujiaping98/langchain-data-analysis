from __future__ import annotations

from dataclasses import dataclass

from .assets import get_metric
from .db import explain, fetch_all
from .layer2_risk import assess_post_risk
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

    sql = compile_metric_sql(metric=metric, time_range=tr, dimensions=[], filters=[], limit=limit)
    post = assess_post_risk(sql)
    if post.action == "block":
        raise ValueError("SQL 风险过高，已阻止执行：" + "; ".join(post.reasons))

    exp_rows = explain(sql)
    rows = fetch_all(sql)
    return TemplateResult(sql=sql, explain=exp_rows, rows=rows, post_risk=post.__dict__)

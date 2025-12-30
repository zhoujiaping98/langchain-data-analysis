from __future__ import annotations

from .config import settings
from .models import MetricDef
from .time_parser import TimeRange


def compile_metric_sql(
    metric: MetricDef,
    time_range: TimeRange,
    dimensions: list[str],
    filters: list[str],
    limit: int,
) -> str:
    select_parts = [f"{metric.measure_expr} AS {metric.metric_key}"]
    group_by: list[str] = []

    for d in dimensions:
        select_parts.insert(0, d)
        group_by.append(d)

    where_parts: list[str] = []
    if metric.default_filters:
        where_parts.append(f"({metric.default_filters})")

    where_parts.append(
        f"({metric.time_column} >= '{time_range.start}' AND {metric.time_column} < '{time_range.end_exclusive}')"
    )

    seen = set()
    for f in filters:
        key = f.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        where_parts.append(f"({f})")

    sql = f"SELECT {', '.join(select_parts)}\nFROM {metric.fact_table}"
    if where_parts:
        sql += "\nWHERE " + "\n  AND ".join(where_parts)
    if group_by:
        sql += "\nGROUP BY " + ", ".join(group_by)

    sql += f"\nLIMIT {min(int(limit), settings.max_rows)}"
    return sql

from __future__ import annotations

from typing import Dict, Any

import sqlglot
from sqlglot import exp

from .config import settings
from .time_parser import TimeRange


def choose_time_column(table_info: Dict[str, Any]) -> str | None:
    names = [str(c.get("name") or "") for c in table_info.get("columns", [])]
    lower = [n.lower() for n in names]
    preferred = ["create_time", "created_at", "update_time", "updated_at", "dt", "date", "time"]
    for p in preferred:
        if p in lower:
            return names[lower.index(p)]
    for n in names:
        n_l = n.lower()
        if "time" in n_l or "date" in n_l:
            return n
    return None


def list_dimensions(table_info: Dict[str, Any]) -> list[str]:
    out: list[str] = []
    for c in table_info.get("columns", []):
        name = str(c.get("name") or "")
        n_l = name.lower()
        if not name:
            continue
        if "time" in n_l or "date" in n_l:
            continue
        if n_l.endswith("_id") or n_l == "id":
            continue
        out.append(name)
    return out


def compile_interactive_sql(
    table: str,
    time_column: str,
    time_range: TimeRange,
    dimensions: list[str],
    limit: int | None = None,
) -> str:
    cols = [exp.column(d) for d in dimensions] if dimensions else []
    measure = exp.alias_(exp.Count(this=exp.Star()), "row_count")
    select_expr = exp.select(*cols, measure)
    select_expr = select_expr.from_(exp.Table(this=exp.to_identifier(table)))

    time_filter = exp.and_(
        exp.GTE(this=exp.column(time_column), expression=exp.Literal.string(time_range.start)),
        exp.LT(this=exp.column(time_column), expression=exp.Literal.string(time_range.end_exclusive)),
    )
    select_expr = select_expr.where(time_filter)

    if dimensions:
        select_expr = select_expr.group_by(*[exp.column(d) for d in dimensions])

    final_limit = min(int(limit or 1000), settings.max_rows)
    select_expr = select_expr.limit(final_limit)
    return select_expr.sql(dialect="mysql")

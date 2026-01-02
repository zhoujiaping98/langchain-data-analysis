from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricDef:
    metric_key: str
    metric_name_zh: str
    description: str
    fact_table: str
    time_column: str
    measure_expr: str
    default_filters: str
    allowed_dims: list[str]
    trigger_keywords: list[str] = None


@dataclass
class TermDef:
    term: str
    canonical: str
    definition: str
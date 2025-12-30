from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson
from sqlalchemy import text

from .db import get_engine
from .models import MetricDef, TermDef

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"


def _read_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def load_seed_metrics() -> list[MetricDef]:
    data = _read_json(ASSETS_DIR / "metrics.seed.json")
    out: list[MetricDef] = []
    for m in data:
        out.append(
            MetricDef(
                metric_key=m["metric_key"],
                metric_name_zh=m["metric_name_zh"],
                description=m.get("description", ""),
                fact_table=m["fact_table"],
                time_column=m["time_column"],
                measure_expr=m["measure_expr"],
                default_filters=m.get("default_filters", ""),
                allowed_dims=m.get("allowed_dims", []),
            )
        )
    return out


def load_seed_terms() -> list[TermDef]:
    data = _read_json(ASSETS_DIR / "terms.seed.json")
    return [TermDef(**t) for t in data]


def list_active_metrics() -> list[MetricDef]:
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT metric_key, metric_name_zh, metric_desc, fact_table, time_column,
                       measure_expr, default_filters, allowed_dims
                FROM metric_definitions
                WHERE is_active = 1
                """
            )
        ).mappings().all()

    out: list[MetricDef] = []
    for r in rows:
        allowed_dims = r["allowed_dims"]
        if isinstance(allowed_dims, str):
            try:
                allowed_dims = json.loads(allowed_dims)
            except Exception:
                allowed_dims = []
        out.append(
            MetricDef(
                metric_key=r["metric_key"],
                metric_name_zh=r["metric_name_zh"],
                description=r.get("metric_desc") or "",
                fact_table=r["fact_table"],
                time_column=r["time_column"],
                measure_expr=r["measure_expr"],
                default_filters=r.get("default_filters") or "",
                allowed_dims=list(allowed_dims) if allowed_dims else [],
            )
        )
    return out


def get_metric(metric_key: str) -> MetricDef | None:
    eng = get_engine()
    with eng.connect() as conn:
        r = conn.execute(
            text(
                """
                SELECT metric_key, metric_name_zh, metric_desc, fact_table, time_column,
                       measure_expr, default_filters, allowed_dims
                FROM metric_definitions
                WHERE metric_key = :k AND is_active = 1
                LIMIT 1
                """
            ),
            {"k": metric_key},
        ).mappings().first()

    if not r:
        return None

    allowed_dims = r["allowed_dims"]
    if isinstance(allowed_dims, str):
        try:
            allowed_dims = json.loads(allowed_dims)
        except Exception:
            allowed_dims = []
    return MetricDef(
        metric_key=r["metric_key"],
        metric_name_zh=r["metric_name_zh"],
        description=r.get("metric_desc") or "",
        fact_table=r["fact_table"],
        time_column=r["time_column"],
        measure_expr=r["measure_expr"],
        default_filters=r.get("default_filters") or "",
        allowed_dims=list(allowed_dims) if allowed_dims else [],
    )


def term_registry() -> dict[str, str]:
    # term -> canonical
    terms = load_seed_terms()
    return {t.term: t.canonical for t in terms}

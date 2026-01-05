from __future__ import annotations

import json

from app.assets import load_seed_metrics
from app.db import execute


def upsert_metric(m):
    execute(
        """
        INSERT INTO metric_definitions
          (metric_key, metric_name_zh, metric_desc, fact_table, time_column, measure_expr, default_filters, allowed_dims, trigger_keywords, version, is_active)
        VALUES
          (:k, :n, :d, :ft, :tc, :me, :df, :ad, :tk, 1, 1)
        ON DUPLICATE KEY UPDATE
          metric_name_zh=VALUES(metric_name_zh),
          metric_desc=VALUES(metric_desc),
          fact_table=VALUES(fact_table),
          time_column=VALUES(time_column),
          measure_expr=VALUES(measure_expr),
          default_filters=VALUES(default_filters),
          allowed_dims=VALUES(allowed_dims),
          trigger_keywords=VALUES(trigger_keywords),
          is_active=1
        """,
        {
            "k": m.metric_key,
            "n": m.metric_name_zh,
            "d": m.description,
            "ft": m.fact_table,
            "tc": m.time_column,
            "me": m.measure_expr,
            "df": m.default_filters,
            "ad": json.dumps(m.allowed_dims, ensure_ascii=False),
            "tk": json.dumps(m.trigger_keywords, ensure_ascii=False),
        },
    )


def main():
    metrics = load_seed_metrics()
    for m in metrics:
        upsert_metric(m)
    print(f"OK: seeded {len(metrics)} metrics.")


if __name__ == "__main__":
    main()

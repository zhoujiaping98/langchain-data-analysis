from __future__ import annotations

import datetime as dt

from .db import execute


def capture_success(user_id: str, role: str, original_query: str, sql: str, metric_key: str | None, satisfied: bool) -> None:
    execute(
        """
        INSERT INTO query_patterns
          (user_id, user_role, user_query, metric_key, sql_text, satisfied, created_at)
        VALUES
          (:user_id, :role, :q, :metric, :sql, :sat, :ts)
        """,
        {
            "user_id": user_id,
            "role": role,
            "q": original_query,
            "metric": metric_key,
            "sql": sql,
            "sat": 1 if satisfied else 0,
            "ts": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
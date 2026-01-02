from __future__ import annotations

import datetime as dt
import logging

from .db import execute

_logger = logging.getLogger(__name__)

_MAX_QUERY_LEN = 2000
_MAX_SQL_LEN = 8000


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    v = value.strip()
    if len(v) <= limit:
        return v
    return v[:limit]


def capture_success(user_id: str, role: str, original_query: str, sql: str, metric_key: str | None, satisfied: bool) -> None:
    try:
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
                "q": _truncate(original_query, _MAX_QUERY_LEN),
                "metric": metric_key,
                "sql": _truncate(sql, _MAX_SQL_LEN),
                "sat": 1 if satisfied else 0,
                "ts": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
    except Exception as exc:
        _logger.warning("capture_success failed: %s", exc)

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import settings


def mysql_url() -> str:
    user = settings.mysql_user
    pwd = settings.mysql_password
    host = settings.mysql_host
    port = settings.mysql_port
    db = settings.mysql_database
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            mysql_url(),
            pool_pre_ping=True,
            pool_recycle=3600,
            future=True,
        )
    return _engine


def fetch_all(sql: str, params: dict | None = None) -> list[dict]:
    eng = get_engine()
    with eng.connect() as conn:
        res = conn.execute(text(sql), params or {})
        cols = res.keys()
        return [dict(zip(cols, row)) for row in res.fetchall()]


def execute(sql: str, params: dict | None = None) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(sql), params or {})


def explain(sql: str) -> list[dict]:
    eng = get_engine()
    with eng.connect() as conn:
        try:
            res = conn.execute(text("EXPLAIN FORMAT=JSON " + sql))
            cols = res.keys()
            return [dict(zip(cols, row)) for row in res.fetchall()]
        except Exception:
            res = conn.execute(text("EXPLAIN " + sql))
            cols = res.keys()
            return [dict(zip(cols, row)) for row in res.fetchall()]
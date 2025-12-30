from __future__ import annotations

import argparse
import datetime as dt
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp
from sqlalchemy import text

from app.db import get_engine


@dataclass
class ColumnSpec:
    name: str
    col_type: str


@dataclass
class TableSpec:
    name: str
    columns: dict[str, ColumnSpec]
    time_column: str


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            import json

            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
    return []


def columns_from_expr(expr: str) -> set[str]:
    if not expr:
        return set()
    try:
        ast = sqlglot.parse_one(f"SELECT {expr}", dialect="mysql")
    except Exception:
        return set()
    return {c.name for c in ast.find_all(exp.Column) if getattr(c, "name", None)}


def columns_from_filter(filter_sql: str) -> set[str]:
    if not filter_sql:
        return set()
    try:
        ast = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {filter_sql}", dialect="mysql")
    except Exception:
        return set()
    return {c.name for c in ast.find_all(exp.Column) if getattr(c, "name", None)}


def infer_type(col: str) -> str:
    c = col.lower()
    if c.endswith("_at"):
        return "DATETIME"
    if c.endswith("_date") or c in {"dt", "date"}:
        return "DATE"
    if c.endswith("_id") or c in {"id", "user_id", "order_id"}:
        return "BIGINT"
    if c in {"amount", "gmv", "revenue", "price"}:
        return "DECIMAL(18,2)"
    if c in {"status", "event_name", "channel", "seller_region"}:
        return "VARCHAR(64)"
    return "VARCHAR(128)"


def build_table_specs(rows: list[dict]) -> dict[str, TableSpec]:
    tables: dict[str, TableSpec] = {}
    for r in rows:
        table = r["fact_table"]
        time_col = r["time_column"]
        allowed_dims = parse_json_list(r.get("allowed_dims"))
        cols = set(allowed_dims)
        cols.add(time_col)
        cols |= columns_from_expr(r.get("measure_expr") or "")
        cols |= columns_from_filter(r.get("default_filters") or "")

        if table not in tables:
            tables[table] = TableSpec(name=table, columns={}, time_column=time_col)

        spec = tables[table]
        spec.time_column = time_col
        for c in cols:
            if not c:
                continue
            if c not in spec.columns:
                spec.columns[c] = ColumnSpec(name=c, col_type=infer_type(c))
    return tables


def create_table(spec: TableSpec) -> str:
    cols = ["id BIGINT PRIMARY KEY AUTO_INCREMENT"]
    for c in sorted(spec.columns.values(), key=lambda x: x.name):
        cols.append(f"{c.name} {c.col_type}")
    cols_sql = ",\n  ".join(cols)
    return f"CREATE TABLE IF NOT EXISTS {spec.name} (\n  {cols_sql}\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"


def existing_columns(table: str) -> list[dict[str, Any]]:
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(text(f"SHOW COLUMNS FROM {table}")).mappings().all()
    return [dict(r) for r in rows]


def required_columns(columns_meta: list[dict[str, Any]]) -> set[str]:
    required = set()
    for r in columns_meta:
        nullable = str(r.get("Null", "")).upper() == "YES"
        has_default = r.get("Default") is not None
        extra = str(r.get("Extra", "")).lower()
        if not nullable and not has_default and "auto_increment" not in extra:
            required.add(r["Field"])
    return required


def type_category(type_str: str | None) -> str:
    t = (type_str or "").lower()
    if "int" in t:
        return "int"
    if "decimal" in t or "numeric" in t or "float" in t or "double" in t:
        return "float"
    if "date" in t and "time" in t:
        return "datetime"
    if t.startswith("date"):
        return "date"
    return "str"


def sample_value(col: str, date_value: dt.date, col_type: str | None = None) -> Any:
    c = col.lower()
    cat = type_category(col_type)
    if cat == "datetime":
        return dt.datetime.combine(date_value, dt.time(hour=random.randint(0, 23), minute=random.randint(0, 59)))
    if cat == "date":
        return date_value
    if cat == "int":
        return random.randint(100000, 999999)
    if cat == "float":
        return round(random.uniform(10, 2000), 2)
    if c.endswith("_at"):
        return dt.datetime.combine(date_value, dt.time(hour=random.randint(0, 23), minute=random.randint(0, 59)))
    if c.endswith("_date") or c in {"dt", "date"}:
        return date_value
    if c in {"user_id", "order_id", "id"} or c.endswith("_id"):
        return random.randint(100000, 999999)
    if c in {"amount", "gmv", "revenue", "price"}:
        return round(random.uniform(10, 2000), 2)
    if c == "status":
        return random.choice(["paid", "pending", "cancelled"])
    if c == "event_name":
        return random.choice(["active", "login", "purchase"])
    if c == "channel":
        return random.choice(["web", "app", "mini"])
    if c == "seller_region":
        return random.choice(["east", "north", "south", "west"])
    return "n/a"


def insert_rows(
    spec: TableSpec,
    rows: int,
    start_date: dt.date,
    days: int,
    allowed_columns: set[str] | None = None,
    required_extra: set[str] | None = None,
    column_types: dict[str, str] | None = None,
) -> int:
    if rows <= 0:
        return 0

    columns = [c.name for c in sorted(spec.columns.values(), key=lambda x: x.name)]
    if allowed_columns is not None:
        columns = [c for c in columns if c in allowed_columns]
    if required_extra:
        for c in sorted(required_extra):
            if c not in columns:
                columns.append(c)
    if not columns:
        return 0
    placeholders = ", ".join([f":{c}" for c in columns])
    sql = f"INSERT INTO {spec.name} ({', '.join(columns)}) VALUES ({placeholders})"

    data = []
    for _ in range(rows):
        date_value = start_date + dt.timedelta(days=random.randint(0, max(days - 1, 0)))
        row = {c: sample_value(c, date_value, (column_types or {}).get(c)) for c in columns}
        data.append(row)

    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(sql), data)
    return len(data)


def table_row_count(name: str) -> int:
    eng = get_engine()
    with eng.connect() as conn:
        r = conn.execute(text(f"SELECT COUNT(*) AS c FROM {name}")).mappings().first()
    return int(r["c"]) if r else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="insert even if table has data")
    args = parser.parse_args()

    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT metric_key, fact_table, time_column, measure_expr, default_filters, allowed_dims
                FROM metric_definitions
                WHERE is_active = 1
                """
            )
        ).mappings().all()

    if not rows:
        print("No active metrics found.")
        return 1

    specs = build_table_specs(list(rows))
    created = 0
    inserted = 0
    for spec in specs.values():
        ddl = create_table(spec)
        with eng.begin() as conn:
            conn.execute(text(ddl))
        created += 1

        count = table_row_count(spec.name)
        if count > 0 and not args.force:
            print(f"Skip insert for {spec.name} (rows={count})")
            continue
        meta = existing_columns(spec.name)
        existing = {r["Field"] for r in meta}
        types = {r["Field"]: r.get("Type", "") for r in meta}
        missing = sorted(set(c.name for c in spec.columns.values()) - existing)
        if missing:
            print(f"Warn: {spec.name} missing columns {missing}, inserting only existing fields.")
        required = required_columns(meta) - existing.intersection({"id"})

        start_date = dt.date.today() - dt.timedelta(days=args.days)
        inserted += insert_rows(
            spec,
            args.rows,
            start_date,
            args.days,
            allowed_columns=existing,
            required_extra=required,
            column_types=types,
        )
        print(f"Inserted {args.rows} rows into {spec.name}")

    print(f"Tables prepared: {created}, rows inserted: {inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

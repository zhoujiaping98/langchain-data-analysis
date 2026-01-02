from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from typing import Any

from sqlalchemy import text

from app.db import get_engine


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
    if "json" in t:
        return "json"
    return "str"


def sample_value(col: str, col_type: str | None, date_value: dt.date) -> Any:
    name = col.lower()
    cat = type_category(col_type)
    if name.endswith("_time") or name.endswith("_at") or name in {"create_time", "update_time"}:
        return dt.datetime.combine(
            date_value,
            dt.time(hour=random.randint(0, 23), minute=random.randint(0, 59), second=random.randint(0, 59)),
        )
    if name.endswith("_date") or name in {"dt", "date"}:
        return date_value
    if cat == "datetime":
        return dt.datetime.combine(
            date_value,
            dt.time(hour=random.randint(0, 23), minute=random.randint(0, 59), second=random.randint(0, 59)),
        )
    if cat == "date":
        return date_value
    if cat == "int":
        if name.endswith("_flag"):
            return random.choice([0, 1])
        if name.endswith("_days") or name.endswith("_count") or name.endswith("_quantity"):
            return random.randint(1, 1000)
        return random.randint(1, 1000)
    if cat == "float":
        return round(random.uniform(10, 5000), 2)
    if cat == "json":
        return json.dumps({"seed": True, "col": col})

    if name.endswith("_id"):
        return random.randint(100000, 999999)
    if name in {"status", "document_status"}:
        return random.choice(["draft", "submitted", "approved", "rejected"])
    if name in {"material_type"}:
        return random.choice(["electrical", "mechanical", "software"])
    if name in {"application_type"}:
        return random.choice(["internal", "external", "partner"])
    if name in {"project_name"}:
        return f"project_{random.randint(1, 50)}"
    if name in {"creator", "author_name"}:
        return f"user_{random.randint(1, 100)}"
    if name in {"material_code"}:
        return f"MAT-{random.randint(1000, 9999)}"
    if name in {"bom_no"}:
        return f"BOM-{random.randint(10000, 99999)}"
    return f"v_{col}_{random.randint(1, 9999)}"


def table_columns(table: str) -> list[dict[str, Any]]:
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(text(f"SHOW COLUMNS FROM {table}")).mappings().all()
    return [dict(r) for r in rows]


def insert_rows(table: str, rows: int, days: int) -> int:
    cols = table_columns(table)
    if not cols:
        return 0

    insert_cols = []
    col_types = {}
    for c in cols:
        extra = str(c.get("Extra", "")).lower()
        if "auto_increment" in extra:
            continue
        insert_cols.append(c["Field"])
        col_types[c["Field"]] = c.get("Type")

    if not insert_cols:
        return 0

    placeholders = ", ".join([f":{c}" for c in insert_cols])
    sql = f"INSERT INTO {table} ({', '.join(insert_cols)}) VALUES ({placeholders})"

    start_date = dt.date.today() - dt.timedelta(days=days)
    data = []
    for _ in range(rows):
        date_value = start_date + dt.timedelta(days=random.randint(0, max(days - 1, 0)))
        row = {c: sample_value(c, col_types.get(c), date_value) for c in insert_cols}
        data.append(row)

    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(sql), data)
    return len(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--table", default="material_bom_list_info")
    args = parser.parse_args()

    count = insert_rows(args.table, args.rows, args.days)
    print(f"Inserted {count} rows into {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

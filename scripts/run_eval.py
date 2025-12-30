from __future__ import annotations

import argparse
import json
from pathlib import Path

import sqlglot

from app.assets import load_seed_metrics
from app.layer2_risk import assess_post_risk
from app.layer3_controlled_t2sql import plan_with_llm
from app.sql_compiler import compile_metric_sql
from app.time_parser import infer_time_range


RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def load_metrics_by_key() -> dict[str, object]:
    metrics = load_seed_metrics()
    return {m.metric_key: m for m in metrics}


def normalize_sql(sql: str) -> str:
    return sqlglot.parse_one(sql, dialect="mysql").sql(dialect="mysql", normalize=True)


def compare_sql(actual: str, expected: str) -> tuple[bool, str]:
    try:
        a = normalize_sql(actual)
        b = normalize_sql(expected)
    except Exception as e:
        return False, f"parse_error: {e}"
    return a == b, f"actual={a} expected={b}"


def build_sql_from_case(case: dict, metric: object, use_llm: bool) -> tuple[str, dict]:
    query = case["query"]
    tr = infer_time_range(query)
    if not tr:
        raise ValueError("time range not found")

    if use_llm:
        qp = plan_with_llm(query, hinted_metric=metric.metric_key, allowed_dims=metric.allowed_dims)
        plan = {"dimensions": qp.dimensions, "filters": qp.filters, "limit": qp.limit}
    else:
        plan = case.get("plan", {})

    sql = compile_metric_sql(
        metric=metric,
        time_range=tr,
        dimensions=plan.get("dimensions", []),
        filters=plan.get("filters", []),
        limit=plan.get("limit", 1000),
    )
    return sql, plan


def run_case(case: dict, metrics: dict[str, object], use_llm: bool) -> dict:
    query = case["query"]
    metric_key = case["metric_key"]
    expect = case.get("expect", {})

    metric = metrics.get(metric_key)
    if not metric:
        return {"name": case.get("name", metric_key), "ok": False, "error": "metric not found"}

    try:
        sql, plan = build_sql_from_case(case, metric, use_llm=use_llm)
    except Exception as e:
        return {"name": case.get("name", metric_key), "ok": False, "error": str(e)}

    risk = assess_post_risk(sql)
    max_level = expect.get("risk_max_level")
    if max_level is not None:
        ok = RISK_ORDER[risk.level] <= RISK_ORDER[max_level]
    else:
        ok = True

    expected_sql = case.get("expected_sql")
    sql_ok = True
    sql_diff = None
    if expected_sql:
        sql_ok, sql_diff = compare_sql(sql, expected_sql)
        ok = ok and sql_ok

    return {
        "name": case.get("name", metric_key),
        "ok": ok,
        "sql_ok": sql_ok,
        "sql_diff": sql_diff,
        "risk": risk.__dict__,
        "sql": sql,
        "plan": plan,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="scripts/eval_cases.json")
    parser.add_argument("--llm", action="store_true", help="use LLM to generate plan")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    metrics = load_metrics_by_key()

    results = [run_case(c, metrics, use_llm=args.llm) for c in cases]
    failed = [r for r in results if not r["ok"]]

    print(f"Total: {len(results)}, Failed: {len(failed)}")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        sql_note = ""
        if r.get("sql_ok") is False:
            sql_note = " sql_mismatch"
        print(f"[{status}] {r['name']}: level={r['risk']['level']}{sql_note}")
        if r.get("sql_ok") is False:
            print(f"  {r.get('sql_diff')}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

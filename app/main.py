from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, List

import sqlglot
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .layer1_router import HybridRouter
from .layer2_risk import assess_pre_risk
from .interactive_store import InteractionStore
from .assets import get_metric
from .layer3_controlled_t2sql import controlled_text_to_sql, get_database_schema, route_tables, run_sql
from .layer4_feedback import capture_success
from .template_engine import run_template
from .result_analysis import analyze_query_result


logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(title="LangChain 4-Layer Analytics Agent", version="0.1.0")

router = HybridRouter()
interaction_store = InteractionStore()
ROOT_DIR = Path(__file__).resolve().parent.parent


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="user identifier")
    role: str = Field(default="analyst", description="user role: analyst/admin/intern/... ")
    query: str = Field(..., description="natural language query")

    # optional feedback about previous response
    satisfied: bool | None = None
    last_sql: str | None = None
    last_metric: str | None = None


class ChatResponse(BaseModel):
    route: str
    confidence: float
    normalized_query: str
    best_metric: str | None
    candidates: list[dict]
    missing_slots: list[str]
    message: str
    sql: str | None = None
    explain: list[dict] | None = None
    rows: list[dict] | None = None
    post_risk: dict | None = None
    is_schema_guided: bool = False  # 是否使用了Schema Prompting
    sql_confidence: float = 0.0    # SQL生成置信度


class AgentStreamRequest(BaseModel):
    user_id: str = Field(..., description="user identifier")
    role: str = Field(default="analyst", description="user role: analyst/admin/intern/... ")
    query: str = Field(..., description="natural language query")
    action: str | None = Field(default=None, description="optional user action")
    selection: Any | None = Field(default=None, description="action payload")
    session_id: str | None = Field(default=None, description="interaction session id")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Layer4: accept feedback for previous step
    if req.satisfied is not None and req.last_sql:
        capture_success(
            user_id=req.user_id,
            role=req.role,
            original_query=req.query,
            sql=req.last_sql,
            metric_key=req.last_metric,
            satisfied=bool(req.satisfied),
        )

    # Layer2 pre-risk
    pre = assess_pre_risk(req.query, role=req.role)
    if pre.action == "block":
        return ChatResponse(
            route="BLOCK",
            confidence=1.0,
            normalized_query=req.query,
            best_metric=None,
            candidates=[],
            missing_slots=[],
            message="请求被阻止：" + "; ".join(pre.reasons),
        )

    # Layer1 router
    m = router.route(req.query, user_id=req.user_id, role=req.role)
    money_kw = ["金额", "总额", "总金额", "GMV", "收入", "营收"]
    if any(k.lower() in m.normalized_query.lower() for k in money_kw):
        if any(c.metric_key == "gmv" for c in m.candidates):
            m.best_metric = "gmv"

    if m.route == "ASK_CLARIFY":
        msg = "需要补充信息确认："
        if m.missing_slots:
            msg += "缺少 " + ", ".join(m.missing_slots) + "。"
        if m.candidates:
            msg += "可能指标：" + ", ".join([c.metric_key for c in m.candidates[:3]])
        return ChatResponse(
            route="ASK_CLARIFY",
            confidence=m.confidence,
            normalized_query=m.normalized_query,
            best_metric=m.best_metric,
            candidates=[c.__dict__ for c in m.candidates],
            missing_slots=m.missing_slots,
            message=msg,
        )

    if m.route == "TEMPLATE" and m.best_metric:
        try:
            r = run_template(m.best_metric, req.query)
            return ChatResponse(
                route="TEMPLATE",
                confidence=m.confidence,
                normalized_query=m.normalized_query,
                best_metric=m.best_metric,
                candidates=[c.__dict__ for c in m.candidates],
                missing_slots=m.missing_slots,
                message="已按指标口径生成并执行查询。",
                sql=r.sql,
                explain=r.explain,
                rows=r.rows,
                post_risk=r.post_risk,
            )
        except Exception as e:
            return ChatResponse(
                route="ERROR",
                confidence=m.confidence,
                normalized_query=m.normalized_query,
                best_metric=m.best_metric,
                candidates=[c.__dict__ for c in m.candidates],
                missing_slots=m.missing_slots,
                message=f"模板执行失败：{e}",
            )

    # Layer3 controlled t2sql - 支持无指标查询
    try:
        ep = controlled_text_to_sql(req.query, hinted_metric=m.best_metric)
        return ChatResponse(
            route="CONTROLLED_T2SQL",
            confidence=m.confidence,
            normalized_query=m.normalized_query,
            best_metric=ep.plan.primary_metric,
            candidates=[c.__dict__ for c in m.candidates],
            missing_slots=m.missing_slots,
            message="已在受控模式下生成并执行查询（可查看 SQL）。",
            sql=ep.sql,
            explain=ep.explain,
            rows=ep.rows,
            post_risk=ep.post_risk,
            is_schema_guided=ep.is_schema_guided,
            sql_confidence=ep.plan.sql_confidence,
        )
    except Exception as e:
        return ChatResponse(
            route="ERROR",
            confidence=m.confidence,
            normalized_query=m.normalized_query,
            best_metric=m.best_metric,
            candidates=[c.__dict__ for c in m.candidates],
            missing_slots=m.missing_slots,
            message=f"无法完成查询：{e}",
        )


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _build_interaction_options(m) -> list[dict]:
    options: list[dict] = []
    if m.candidates:
        for c in m.candidates[:5]:
            options.append(
                {
                    "label": f"指标：{c.metric_key}",
                    "value": c.metric_key,
                    "action": "choose_metric",
                }
            )
    if "time" in m.missing_slots:
        for t in ["上周", "上个月", "最近30天", "本月", "今年"]:
            options.append({"label": f"时间：{t}", "value": t, "action": "append_query"})
    return options


def _table_options(
    query: str,
    preferred: list[str] | None = None,
    restrict: bool = False,
) -> list[dict]:
    schema = get_database_schema()
    tables = list(schema.get("tables", {}).keys())
    preferred = preferred or []
    ranked = route_tables(query, schema, top_k=5)
    seen = set()
    options: list[dict] = []
    source = preferred if restrict else preferred + ranked + tables
    for t in source:
        if t in seen:
            continue
        seen.add(t)
        options.append({"label": f"表：{t}", "value": t, "action": "choose_table"})
        if len(options) >= 10:
            break
    return options


def _field_options(table_name: str, allowed_fields: list[str] | None = None) -> list[dict]:
    if allowed_fields:
        return [
            {"label": name, "value": name, "action": "choose_fields"} for name in allowed_fields[:20]
        ]
    schema = get_database_schema()
    table = schema.get("tables", {}).get(table_name, {})
    options: list[dict] = []
    for col in table.get("columns", []):
        name = col.get("name")
        if not name:
            continue
        options.append({"label": name, "value": str(name), "action": "choose_fields"})
        if len(options) >= 20:
            break
    return options


def _template_options(table_name: str, allowed_fields: list[str] | None = None) -> dict:
    schema = get_database_schema()
    table = schema.get("tables", {}).get(table_name, {})
    cols = [c.get("name") for c in table.get("columns", []) if c.get("name")]
    metric_fields = allowed_fields or cols
    time_fields = [c for c in cols if "time" in c.lower() or "date" in c.lower()]
    if not time_fields:
        time_fields = cols
    return {
        "aggregations": ["count", "sum", "avg", "max", "min"],
        "metric_fields": metric_fields[:20],
        "time_fields": time_fields[:10],
        "grains": ["day", "week", "month"],
        "time_ranges": ["上周", "上个月", "最近30天", "本月", "今年"],
    }


def _sql_tags(sql: str) -> dict:
    try:
        ast = sqlglot.parse_one(sql, dialect="mysql")
    except Exception:
        return {}
    tables = [t.name for t in ast.find_all(sqlglot.exp.Table) if getattr(t, "name", None)]
    selects = [s.sql(dialect="mysql") for s in ast.find_all(sqlglot.exp.Select)]
    where = ast.args.get("where")
    group = ast.args.get("group")
    order = ast.args.get("order")
    limit = ast.args.get("limit")
    return {
        "tables": tables,
        "select": selects[0] if selects else "",
        "where": where.sql(dialect="mysql") if where else "",
        "group_by": group.sql(dialect="mysql") if group else "",
        "order_by": order.sql(dialect="mysql") if order else "",
        "limit": limit.sql(dialect="mysql") if limit else "",
    }


def _allowed_cols_for_table(table_name: str) -> set[str]:
    schema = get_database_schema()
    table = schema.get("tables", {}).get(table_name, {})
    cols = set()
    for col in table.get("columns", []):
        name = col.get("name")
        if name:
            cols.add(str(name).lower())
    return cols


def _sql_literal(value: str) -> str:
    if value is None:
        return "NULL"
    v = str(value).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return v
    v = v.replace("'", "''")
    return f"'{v}'"


def _filters_from_structured(filters: list[dict], allowed_cols: set[str]) -> list[str]:
    out: list[str] = []
    op_map = {
        "=": "=",
        "!=": "!=",
        ">": ">",
        "<": "<",
        ">=": ">=",
        "<=": "<=",
        "contains": "LIKE",
        "in": "IN",
    }
    for f in filters:
        field = str(f.get("field") or "").strip()
        op = str(f.get("op") or "").strip().lower()
        value = f.get("value")
        if not field or field.lower() not in allowed_cols:
            continue
        if op not in op_map:
            continue
        if op == "contains":
            lit = _sql_literal(f"%{value}%")
            out.append(f"{field} LIKE {lit}")
        elif op == "in":
            items = [v.strip() for v in str(value or "").split(",") if v.strip()]
            if not items:
                continue
            lits = ", ".join(_sql_literal(v) for v in items)
            out.append(f"{field} IN ({lits})")
        else:
            lit = _sql_literal(value)
            out.append(f"{field} {op_map[op]} {lit}")
    return out


def _stream_events(req: AgentStreamRequest) -> Iterable[str]:
    query = req.query
    chosen_metric = None

    session_id, state = interaction_store.upsert(req.session_id, query)
    yield _sse("session", {"session_id": session_id})

    if req.action is None and req.selection is None and req.session_id:
        state.chosen_table = None
        state.chosen_fields = []
        state.chosen_metric = None
        state.intent_text = None
        state.pending_sql = None
        state.pending_post_risk = None
        state.template = None
        state.chosen_filters = None
        state.force_interactive = False

    if req.action == "append_query" and req.selection:
        query = f"{query} {req.selection}".strip()
        state.query = query
    elif req.action == "choose_metric" and req.selection:
        chosen_metric = str(req.selection)
        state.chosen_metric = chosen_metric
    elif req.action == "choose_table" and req.selection:
        state.chosen_table = str(req.selection)
    elif req.action == "choose_fields" and req.selection:
        if isinstance(req.selection, list):
            state.chosen_fields = [str(v) for v in req.selection]
        else:
            state.chosen_fields = [str(req.selection)]
    elif req.action == "skip_fields":
        state.chosen_fields = []
    elif req.action == "provide_intent" and req.selection:
        state.intent_text = str(req.selection)
    elif req.action == "skip_intent":
        state.intent_text = ""
    elif req.action == "choose_template" and isinstance(req.selection, dict):
        state.template = dict(req.selection)
    elif req.action == "choose_filters" and isinstance(req.selection, list):
        state.chosen_filters = list(req.selection)
    elif req.action == "skip_filters":
        state.chosen_filters = []
    elif req.action == "confirm_sql":
        if state.pending_sql:
            combined_query = state.query
            if state.intent_text:
                combined_query = f"{state.query} {state.intent_text}".strip()
            yield _sse("sql", {"sql": state.pending_sql, "post_risk": state.pending_post_risk})
            exp_rows, rows = run_sql(state.pending_sql)
            yield _sse("rows", {"rows": rows, "row_count": len(rows)})
            analysis = analyze_query_result(query=combined_query, sql=state.pending_sql, rows=rows)
            yield _sse("analysis", {"text": analysis})
            state.pending_sql = None
            state.pending_post_risk = None
            yield _sse("done", {"ok": True})
            return
    elif req.action == "cancel_sql":
        state.pending_sql = None
        state.pending_post_risk = None
        yield _sse("status", {"message": "已取消执行。"})
        yield _sse("done", {"ok": True})
        return
    elif req.action == "post_ok":
        yield _sse("status", {"message": "收到反馈：结果符合预期。"})
        yield _sse("done", {"ok": True})
        return
    elif req.action == "post_refine":
        state.intent_text = None
        state.pending_sql = None
        state.pending_post_risk = None
        state.chosen_filters = None
        state.force_interactive = True

    yield _sse("status", {"message": "已收到请求", "query": query})

    pre = assess_pre_risk(query, role=req.role)
    if pre.action == "block":
        yield _sse("block", {"message": "请求被阻止", "reasons": pre.reasons})
        yield _sse("done", {"ok": False})
        return

    m = router.route(query, user_id=req.user_id, role=req.role)
    if chosen_metric:
        m.best_metric = chosen_metric
        m.route = "TEMPLATE"
    elif state.chosen_metric:
        m.best_metric = state.chosen_metric
        m.route = "TEMPLATE"

    yield _sse(
        "route",
        {
            "route": m.route,
            "confidence": m.confidence,
            "normalized_query": m.normalized_query,
            "best_metric": m.best_metric,
            "missing_slots": m.missing_slots,
            "candidates": [c.__dict__ for c in m.candidates],
        },
    )

    if m.route == "ASK_CLARIFY":
        yield _sse(
            "ask",
            {
                "message": "需要补充信息后才能继续执行。",
                "missing_slots": m.missing_slots,
                "options": _build_interaction_options(m),
                "selection_mode": "single",
            },
        )
        yield _sse("done", {"ok": True, "needs_input": True})
        return

    need_interactive = (
        pre.action == "require_approval"
        or m.confidence < settings.agent_low_confidence_threshold
        or state.force_interactive
    )
    metric_def = get_metric(m.best_metric) if m.best_metric else None

    if need_interactive:
        if not state.chosen_table:
            preferred = [metric_def.fact_table] if metric_def else []
            yield _sse(
                "ask",
                {
                    "message": "为降低风险，请确认查询表。",
                    "options": _table_options(query, preferred=preferred, restrict=bool(metric_def)),
                    "selection_mode": "single",
                },
            )
            yield _sse("done", {"ok": True, "needs_input": True})
            return

        if metric_def and not metric_def.allowed_dims:
            state.chosen_fields = []
        if req.action != "skip_fields" and not state.chosen_fields and (
            not metric_def or metric_def.allowed_dims
        ):
            yield _sse(
                "ask",
                {
                    "message": f"请选择需要展示的字段（可多选），表：{state.chosen_table}",
                    "options": _field_options(
                        state.chosen_table,
                        allowed_fields=metric_def.allowed_dims if metric_def else None,
                    ),
                    "selection_mode": "multi",
                    "confirm_action": "choose_fields",
                    "confirm_label": "完成字段选择",
                    "skip_action": "skip_fields",
                    "skip_label": "跳过字段选择",
                },
            )
            yield _sse("done", {"ok": True, "needs_input": True})
            return
    elif not m.best_metric:
        if not state.chosen_table:
            yield _sse(
                "ask",
                {
                    "message": "未识别到指标，请选择查询表。",
                    "options": _table_options(query),
                    "selection_mode": "single",
                },
            )
            yield _sse("done", {"ok": True, "needs_input": True})
            return

        if req.action != "skip_fields" and not state.chosen_fields:
            yield _sse(
                "ask",
                {
                    "message": f"请选择需要展示的字段（可多选），表：{state.chosen_table}",
                    "options": _field_options(state.chosen_table),
                    "selection_mode": "multi",
                    "confirm_action": "choose_fields",
                    "confirm_label": "完成字段选择",
                    "skip_action": "skip_fields",
                    "skip_label": "跳过字段选择",
                },
            )
            yield _sse("done", {"ok": True, "needs_input": True})
            return

    sql = None
    rows = []
    post_risk = None
    effective_query = query
    if state.intent_text:
        effective_query = f"{state.query} {state.intent_text}".strip()
    if state.template:
        tpl = state.template
        tpl_hint = (
            f"统计方式:{tpl.get('aggregation')}; "
            f"指标字段:{tpl.get('metric_field')}; "
            f"时间字段:{tpl.get('time_field')}; "
            f"粒度:{tpl.get('grain')}; "
            f"时间范围:{tpl.get('time_range')}"
        )
        effective_query = f"{effective_query} {tpl_hint}".strip()

    if (not m.best_metric) or need_interactive:
        if state.chosen_table and not state.template:
            yield _sse(
                "ask",
                {
                    "message": "请选择统计口径模板。",
                    "input_mode": "template",
                    "template": _template_options(
                        state.chosen_table,
                        allowed_fields=metric_def.allowed_dims if metric_def else None,
                    ),
                },
            )
            yield _sse("done", {"ok": True, "needs_input": True})
            return

        if state.chosen_table and state.chosen_filters is None:
            allowed_fields = metric_def.allowed_dims if metric_def else None
            if not allowed_fields and state.chosen_table:
                allowed_fields = list(_allowed_cols_for_table(state.chosen_table))
            yield _sse(
                "ask",
                {
                    "message": "请选择过滤条件（可选）。",
                    "input_mode": "filters",
                    "filter_fields": allowed_fields or [],
                    "selection_mode": "filters",
                    "skip_action": "skip_filters",
                    "skip_label": "跳过过滤条件",
                },
            )
            yield _sse("done", {"ok": True, "needs_input": True})
            return

        if state.intent_text is None:
            yield _sse(
                "ask",
                {
                    "message": "请补充过滤条件/说明（可选）。",
                    "input_mode": "intent",
                    "selection_mode": "free_text",
                    "skip_action": "skip_intent",
                    "skip_label": "跳过补充",
                },
            )
            yield _sse("done", {"ok": True, "needs_input": True})
            return

        try:
            forced_filters = None
            if state.chosen_filters is not None and state.chosen_table:
                allowed_cols = _allowed_cols_for_table(state.chosen_table)
                forced_filters = _filters_from_structured(state.chosen_filters, allowed_cols)
            ep = controlled_text_to_sql(
                effective_query,
                hinted_metric=m.best_metric,
                forced_tables=[state.chosen_table] if state.chosen_table else None,
                allowed_dims_override=state.chosen_fields or None,
                execute_query=False,
                forced_filters=forced_filters,
            )
            state.pending_sql = ep.sql
            state.pending_post_risk = ep.post_risk
            yield _sse(
                "sql",
                {
                    "sql": ep.sql,
                    "post_risk": ep.post_risk,
                    "tags": _sql_tags(ep.sql),
                },
            )
            yield _sse(
                "ask",
                {
                    "message": "已生成 SQL，是否执行？",
                    "options": [
                        {"label": "确认执行", "value": "confirm", "action": "confirm_sql"},
                        {"label": "取消", "value": "cancel", "action": "cancel_sql"},
                    ],
                    "selection_mode": "single",
                },
            )
            yield _sse("done", {"ok": True, "needs_input": True})
            return
        except Exception as e:
            msg = str(e)
            if "时间范围" in msg:
                yield _sse(
                    "ask",
                    {
                        "message": "缺少时间范围，请补充（例如：上周/最近30天）。",
                        "options": _build_interaction_options(m),
                        "input_mode": "intent",
                        "selection_mode": "single",
                    },
                )
                yield _sse("done", {"ok": True, "needs_input": True})
                return
            yield _sse("error", {"message": f"无法完成查询：{e}"})
            yield _sse("done", {"ok": False})
            return

    if m.route == "TEMPLATE" and m.best_metric:
        try:
            r = run_template(m.best_metric, query)
            sql = r.sql
            rows = r.rows or []
            post_risk = r.post_risk
        except Exception as e:
            yield _sse("error", {"message": f"模板执行失败：{e}"})
            yield _sse("done", {"ok": False})
            return
    else:
        try:
            forced_filters = None
            if state.chosen_filters is not None and state.chosen_table:
                allowed_cols = _allowed_cols_for_table(state.chosen_table)
                forced_filters = _filters_from_structured(state.chosen_filters, allowed_cols)
            ep = controlled_text_to_sql(
                effective_query,
                hinted_metric=m.best_metric,
                forced_tables=[state.chosen_table] if state.chosen_table else None,
                allowed_dims_override=state.chosen_fields or None,
                forced_filters=forced_filters,
            )
            sql = ep.sql
            rows = ep.rows or []
            post_risk = ep.post_risk
        except Exception as e:
            yield _sse("error", {"message": f"无法完成查询：{e}"})
            yield _sse("done", {"ok": False})
            return

    yield _sse("sql", {"sql": sql, "post_risk": post_risk})
    yield _sse("rows", {"rows": rows, "row_count": len(rows)})

    analysis = analyze_query_result(query=query, sql=sql, rows=rows)
    yield _sse("analysis", {"text": analysis})
    yield _sse(
        "ask",
        {
            "message": "结果是否符合预期？如需优化，可补充过滤/维度。",
            "options": [
                {"label": "符合预期", "value": "ok", "action": "post_ok"},
                {"label": "继续优化", "value": "refine", "action": "post_refine"},
            ],
            "selection_mode": "single",
            "input_mode": "post_check",
        },
    )
    yield _sse("done", {"ok": True, "needs_input": True})


@app.post("/agent/stream")
def agent_stream(req: AgentStreamRequest):
    return StreamingResponse(_stream_events(req), media_type="text/event-stream")


app.mount("/ui", StaticFiles(directory=ROOT_DIR / "frontend", html=True), name="ui")

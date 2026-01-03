from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .layer1_router import HybridRouter
from .layer2_risk import assess_pre_risk
from .layer3_controlled_t2sql import (
    controlled_text_to_sql,
    controlled_text_to_sql_preview,
    get_database_schema,
    route_tables,
)
from .db import explain, fetch_all
from .interactive_query import choose_time_column, list_dimensions, compile_interactive_sql
from .interactive_state import InteractionStore
from .layer4_feedback import capture_success
from .template_engine import build_template_sql, run_template


logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(title="LangChain 4-Layer Analytics Agent", version="0.1.0")

router = HybridRouter()
interactive_store = InteractionStore()


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="user identifier")
    role: str = Field(default="analyst", description="user role: analyst/admin/intern/... ")
    query: str = Field(..., description="natural language query")

    # optional feedback about previous response
    satisfied: bool | None = None
    last_sql: str | None = None
    last_metric: str | None = None

    # interactive flow
    session_id: str | None = None
    action: str | None = None  # select_table | select_dimensions | select_time
    selection: dict | None = None


class SelectTableRequest(BaseModel):
    session_id: str
    table: str


class SelectDimensionsRequest(BaseModel):
    session_id: str
    dimensions: list[str] = Field(default_factory=list)


class SelectTimeRequest(BaseModel):
    session_id: str
    time_range: str


class ConfirmExecuteRequest(BaseModel):
    session_id: str
    confirm: bool = True


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
    session_id: str | None = None
    next_action: str | None = None
    options: list[dict] | None = None


@app.get("/health")
def health():
    return {"ok": True}


def _handle_interactive_action(
    session_id: str | None,
    action: str | None,
    selection: dict | None,
) -> ChatResponse | None:
    if not session_id and not action:
        return None
    state = interactive_store.get(session_id) if session_id else None
    if not state:
        return ChatResponse(
            route="ERROR",
            confidence=0.0,
            normalized_query="",
            best_metric=None,
            candidates=[],
            missing_slots=[],
            message="会话已过期，请重新发起查询。",
        )

    if action == "select_table":
        table = (selection or {}).get("table")
        if not table:
            return ChatResponse(
                route="ASK_TABLE",
                confidence=0.0,
                normalized_query=state.query,
                best_metric=None,
                candidates=[],
                missing_slots=["table"],
                message="请先选择要查询的表。",
                session_id=state.session_id,
                next_action="select_table",
            )
        state.table = table
        schema = get_database_schema()
        table_info = schema.get("tables", {}).get(table)
        if not table_info:
            return ChatResponse(
                route="ASK_TABLE",
                confidence=0.0,
                normalized_query=state.query,
                best_metric=None,
                candidates=[],
                missing_slots=["table"],
                message="该表不存在或不可访问，请重新选择。",
                session_id=state.session_id,
                next_action="select_table",
            )
        state.time_column = choose_time_column(table_info)
        dims = list_dimensions(table_info)[:8]
        state.stage = "NEED_DIMENSION"
        interactive_store.update(state)
        return ChatResponse(
            route="ASK_DIMENSION",
            confidence=0.0,
            normalized_query=state.query,
            best_metric=None,
            candidates=[],
            missing_slots=["dimensions"],
            message="请选择要分组的维度（可选，可为空）。",
            session_id=state.session_id,
            next_action="select_dimensions",
            options=[{"dimension": d} for d in dims],
        )

    if action == "select_dimensions":
        dims = (selection or {}).get("dimensions") or []
        if not isinstance(dims, list):
            dims = []
        state.dimensions = dims
        state.stage = "NEED_TIME"
        interactive_store.update(state)
        return ChatResponse(
            route="ASK_TIME",
            confidence=0.0,
            normalized_query=state.query,
            best_metric=None,
            candidates=[],
            missing_slots=["time_range"],
            message="请提供时间范围（如：上周/本月/2025-01-01~2025-01-31）。",
            session_id=state.session_id,
            next_action="select_time",
        )

    if action == "select_time":
        time_text = (selection or {}).get("time_range")
        if not time_text:
            return ChatResponse(
                route="ASK_TIME",
                confidence=0.0,
                normalized_query=state.query,
                best_metric=None,
                candidates=[],
                missing_slots=["time_range"],
                message="请提供时间范围。",
                session_id=state.session_id,
                next_action="select_time",
            )
        from .time_parser import infer_time_range

        tr = infer_time_range(str(time_text))
        if not tr:
            return ChatResponse(
                route="ASK_TIME",
                confidence=0.0,
                normalized_query=state.query,
                best_metric=None,
                candidates=[],
                missing_slots=["time_range"],
                message="无法识别时间范围，请重新输入。",
                session_id=state.session_id,
                next_action="select_time",
            )
        state.time_range = tr
        state.stage = "CONFIRM"

        if not state.table or not state.time_column:
            return ChatResponse(
                route="ERROR",
                confidence=0.0,
                normalized_query=state.query,
                best_metric=None,
                candidates=[],
                missing_slots=[],
                message="缺少表或时间字段信息，无法生成查询。",
                session_id=state.session_id,
            )
        sql = compile_interactive_sql(
            table=state.table,
            time_column=state.time_column,
            time_range=state.time_range,
            dimensions=state.dimensions,
            limit=settings.max_rows,
        )
        from .layer2_risk import assess_post_risk

        post = assess_post_risk(sql)
        if post.action == "block":
            return ChatResponse(
                route="ERROR",
                confidence=0.0,
                normalized_query=state.query,
                best_metric=None,
                candidates=[],
                missing_slots=[],
                message="SQL 风险过高，已阻止执行：" + "; ".join(post.reasons),
                session_id=state.session_id,
            )
        state.pending_sql = sql
        interactive_store.update(state)
        return ChatResponse(
            route="ASK_CONFIRM",
            confidence=1.0,
            normalized_query=state.query,
            best_metric=None,
            candidates=[],
            missing_slots=[],
            message="已生成查询预览，请确认是否执行。",
            sql=sql,
            post_risk=post.__dict__,
            session_id=state.session_id,
            next_action="confirm_execute",
        )

    if action == "confirm_execute":
        confirm = bool((selection or {}).get("confirm", True))
        if not confirm:
            return ChatResponse(
                route="CANCELLED",
                confidence=0.0,
                normalized_query=state.query,
                best_metric=None,
                candidates=[],
                missing_slots=[],
                message="已取消执行。",
                session_id=state.session_id,
            )
        if not state.pending_sql:
            return ChatResponse(
                route="ERROR",
                confidence=0.0,
                normalized_query=state.query,
                best_metric=None,
                candidates=[],
                missing_slots=[],
                message="缺少可执行SQL，请重新选择。",
                session_id=state.session_id,
            )
        from .layer2_risk import assess_post_risk

        post = assess_post_risk(state.pending_sql)
        if post.action == "block":
            return ChatResponse(
                route="ERROR",
                confidence=0.0,
                normalized_query=state.query,
                best_metric=None,
                candidates=[],
                missing_slots=[],
                message="SQL 风险过高，已阻止执行：" + "; ".join(post.reasons),
                session_id=state.session_id,
            )
        exp_rows = explain(state.pending_sql)
        rows = fetch_all(state.pending_sql)
        return ChatResponse(
            route="CONTROLLED_T2SQL",
            confidence=1.0,
            normalized_query=state.query,
            best_metric=None,
            candidates=[],
            missing_slots=[],
            message="已按交互式选择生成并执行查询。",
            sql=state.pending_sql,
            explain=exp_rows,
            rows=rows,
            post_risk=post.__dict__,
            session_id=state.session_id,
        )

    return ChatResponse(
        route="ERROR",
        confidence=0.0,
        normalized_query=state.query,
        best_metric=None,
        candidates=[],
        missing_slots=[],
        message="未知的交互操作。",
        session_id=state.session_id,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Interactive flow
    if req.session_id or req.action:
        resp = _handle_interactive_action(req.session_id, req.action, req.selection)
        if resp:
            return resp

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
            preview = build_template_sql(m.best_metric, req.query)
            if m.confidence < settings.router_conf_ask or preview.post_risk.get("action") != "allow_with_preview":
                state = interactive_store.create(req.query, stage="CONFIRM", suggested_tables=[])
                state.pending_sql = preview.sql
                interactive_store.update(state)
                return ChatResponse(
                    route="ASK_CONFIRM",
                    confidence=m.confidence,
                    normalized_query=m.normalized_query,
                    best_metric=m.best_metric,
                    candidates=[c.__dict__ for c in m.candidates],
                    missing_slots=[],
                    message="置信度较低或风险较高，请确认后执行。",
                    sql=preview.sql,
                    post_risk=preview.post_risk,
                    session_id=state.session_id,
                    next_action="confirm_execute",
                )
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

    # If no metric resolved, start interactive flow
    if not m.best_metric:
        schema_info = get_database_schema()
        candidates = route_tables(req.query, schema_info, top_k=1)
        if not candidates:
            candidates = list(schema_info.get("tables", {}).keys())[:3]
        state = interactive_store.create(req.query, stage="NEED_TABLE", suggested_tables=candidates)
        options = [{"table": t} for t in candidates]
        return ChatResponse(
            route="ASK_TABLE",
            confidence=m.confidence,
            normalized_query=m.normalized_query,
            best_metric=None,
            candidates=[c.__dict__ for c in m.candidates],
            missing_slots=["table"],
            message="请选择要查询的表。",
            session_id=state.session_id,
            next_action="select_table",
            options=options,
        )

    # Layer3 controlled t2sql - 支持无指标查询
    try:
        preview = controlled_text_to_sql_preview(req.query, hinted_metric=m.best_metric)
        if (
            m.confidence < settings.router_conf_ask
            or preview.post_risk.get("action") != "allow_with_preview"
            or preview.plan.sql_confidence < settings.l3_require_confirmation_sql_confidence
        ):
            state = interactive_store.create(req.query, stage="CONFIRM", suggested_tables=[])
            state.pending_sql = preview.sql
            interactive_store.update(state)
            return ChatResponse(
                route="ASK_CONFIRM",
                confidence=m.confidence,
                normalized_query=m.normalized_query,
                best_metric=m.best_metric,
                candidates=[c.__dict__ for c in m.candidates],
                missing_slots=[],
                message="置信度较低或风险较高，请确认后执行。",
                sql=preview.sql,
                post_risk=preview.post_risk,
                session_id=state.session_id,
                next_action="confirm_execute",
            )
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


@app.post("/interactive/select_table", response_model=ChatResponse)
def select_table(req: SelectTableRequest):
    return _handle_interactive_action(
        session_id=req.session_id,
        action="select_table",
        selection={"table": req.table},
    )


@app.post("/interactive/select_dimensions", response_model=ChatResponse)
def select_dimensions(req: SelectDimensionsRequest):
    return _handle_interactive_action(
        session_id=req.session_id,
        action="select_dimensions",
        selection={"dimensions": req.dimensions},
    )


@app.post("/interactive/select_time", response_model=ChatResponse)
def select_time(req: SelectTimeRequest):
    return _handle_interactive_action(
        session_id=req.session_id,
        action="select_time",
        selection={"time_range": req.time_range},
    )


@app.post("/interactive/confirm", response_model=ChatResponse)
def confirm_execute(req: ConfirmExecuteRequest):
    return _handle_interactive_action(
        session_id=req.session_id,
        action="confirm_execute",
        selection={"confirm": req.confirm},
    )

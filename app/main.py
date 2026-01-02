from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .layer1_router import HybridRouter
from .layer2_risk import assess_pre_risk
from .layer3_controlled_t2sql import controlled_text_to_sql
from .layer4_feedback import capture_success
from .template_engine import run_template


app = FastAPI(title="LangChain 4-Layer Analytics Agent", version="0.1.0")

router = HybridRouter()


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
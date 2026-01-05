from __future__ import annotations

import json
import re
import threading
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
import sqlglot
from sqlglot import exp
from sqlalchemy import text

from .assets import get_metric
from .config import settings
from .db import explain, fetch_all, get_engine
from .layer2_risk import assess_post_risk, validate_select_only, RiskAssessment
from .llm import get_llm
from .sql_compiler import compile_metric_sql
from .time_parser import infer_time_range, TimeRange


class QueryPlan(BaseModel):
    primary_metric: str | None = Field(default=None, description="metric_key if known")
    dimensions: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    limit: int = 1000
    suggested_tables: List[str] = Field(default_factory=list, description="建议使用的表名")
    sql_confidence: float = Field(default=0.0, description="SQL生成置信度")


@dataclass
class ExecutionPlan:
    sql: str
    plan: QueryPlan
    explain: list[dict]
    post_risk: dict
    rows: list[dict]
    is_schema_guided: bool = False  # 标识是否使用了schema prompting


class DatabaseSchemaCache:
    """数据库Schema缓存管理器"""

    def __init__(self, cache_ttl: int = 3600):  # 默认缓存1小时
        self.cache_ttl = cache_ttl
        self._cache = None
        self._last_update = 0
        self._lock = threading.Lock()

    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        import time
        return time.time() - self._last_update > self.cache_ttl

    def get_schema(self) -> Optional[Dict[str, Any]]:
        """获取缓存的schema信息"""
        with self._lock:
            if self._cache is None or self.is_expired():
                return None
            return self._cache

    def update_schema(self, schema_info: Dict[str, Any]):
        """更新schema缓存"""
        import time
        with self._lock:
            self._cache = schema_info
            self._last_update = time.time()

    def force_refresh(self):
        """强制刷新缓存"""
        with self._lock:
            self._cache = None
            self._last_update = 0


# 创建全局schema缓存实例
_schema_cache_manager = DatabaseSchemaCache(cache_ttl=1800)  # 缓存30分钟
_logger = logging.getLogger(__name__)


def get_database_schema() -> Dict[str, Any]:
    """获取数据库Schema信息，使用缓存优化性能"""
    # 尝试从缓存获取
    cached_schema = _schema_cache_manager.get_schema()
    if cached_schema is not None:
        return cached_schema

    # 缓存未命中或过期，重新获取
    schema_info = {
        "tables": {},
        "all_columns": set(),
        "table_relationships": {}
    }

    try:
        eng = get_engine()
        with eng.connect() as conn:
            # 获取所有表名
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result.fetchall()]

            # 串行拉取，避免连接池压力与线程超时泄漏
            for table_name in tables:
                try:
                    table_info = _get_table_schema(conn, table_name)
                    if table_info:
                        schema_info["tables"][table_name] = table_info
                        # 收集所有列名
                        for col in table_info["columns"]:
                            name = col["name"]
                            schema_info["all_columns"].add(str(name).lower())
                except Exception as e:
                    _logger.warning("获取表 %s 结构失败: %s", table_name, e)
                    continue

            # 转换set为list以便JSON序列化
            schema_info["all_columns"] = list(schema_info["all_columns"])

            # 更新缓存
            _schema_cache_manager.update_schema(schema_info)

    except Exception as e:
        _logger.exception("获取数据库Schema失败: %s", e)
        # 返回空的schema信息而不是崩溃
        return {
            "tables": {},
            "all_columns": [],
            "table_relationships": {}
        }

    return schema_info


def _quote_identifier(name: str) -> str:
    safe = name.replace("`", "``")
    return f"`{safe}`"


def _get_table_schema(conn, table_name: str) -> Optional[Dict[str, Any]]:
    """获取单个表的结构信息"""
    try:
        desc_result = conn.execute(text(f"DESCRIBE {_quote_identifier(table_name)}"))
        columns = desc_result.fetchall()

        table_info = {
            "columns": [],
            "primary_keys": [],
            "foreign_keys": []
        }

        for col in columns:
            col_name, col_type, is_null, key_type, default_val, extra = col
            col_info = {
                "name": col_name,
                "type": col_type,
                "nullable": is_null == "YES",
                "default": default_val,
                "is_primary": key_type == "PRI"
            }

            if key_type == "PRI":
                table_info["primary_keys"].append(col_name)

            table_info["columns"].append(col_info)

        return table_info

    except Exception as e:
        _logger.warning("获取表 %s 结构时出错: %s", table_name, e)
        return None


def build_schema_prompt(schema_info: Dict[str, Any]) -> str:
    """构建Schema Prompt"""
    prompt = "数据库Schema信息:\n"

    for table_name, table_info in schema_info["tables"].items():
        prompt += f"\n表名: {table_name}\n"
        prompt += "字段:\n"

        for col in table_info["columns"]:
            nullable_str = "可空" if col["nullable"] else "非空"
            pk_str = "主键" if col["is_primary"] else ""

            # 基于字段名智能推断语义
            semantic_hint = _infer_field_semantics(col['name'])
            hint_str = f"  # {semantic_hint}" if semantic_hint else ""

            prompt += f"  - {col['name']} ({col['type']}, {nullable_str}) {pk_str}{hint_str}\n"

        if table_info["primary_keys"]:
            prompt += f"主键: {', '.join(table_info['primary_keys'])}\n"

    prompt += "\n注意事项:\n"
    prompt += "1. 必须使用上述已存在的表名和字段名\n"
    prompt += "2. 不要虚构任何不存在的字段或表\n"
    prompt += "3. 时间字段格式: YYYY-MM-DD HH:MM:SS\n"
    prompt += "4. 字符串字段需要用单引号包围\n"
    prompt += "5. 只输出纯JSON格式，不要包含任何解释或注释\n"
    prompt += "6. # 注释标识了字段的业务语义推断，请参考但不要完全依赖\n"

    return prompt


def _filter_schema(schema_info: Dict[str, Any], tables: List[str]) -> Dict[str, Any]:
    if not tables:
        return schema_info
    out = {"tables": {}, "all_columns": [], "table_relationships": {}}
    for t in tables:
        info = schema_info.get("tables", {}).get(t)
        if not info:
            continue
        out["tables"][t] = info
        for col in info.get("columns", []):
            name = col.get("name")
            if name:
                out["all_columns"].append(name)
    return out


def _infer_field_semantics(field_name: str) -> str:
    """基于字段名智能推断业务语义"""
    field_lower = field_name.lower()

    # 项目相关
    if any(keyword in field_lower for keyword in ['project', '项目']):
        if any(keyword in field_lower for keyword in ['name', '名称', 'title', '标题']):
            return "项目名称"
        elif any(keyword in field_lower for keyword in ['type', '类型']):
            return "项目类型"
        elif any(keyword in field_lower for keyword in ['status', '状态']):
            return "项目状态"

    # 物料相关
    elif any(keyword in field_lower for keyword in ['material', '物料']):
        if any(keyword in field_lower for keyword in ['type', '类型']):
            return "物料类型"
        elif any(keyword in field_lower for keyword in ['code', '编码', '编号']):
            return "物料编码"
        elif any(keyword in field_lower for keyword in ['name', '名称']):
            return "物料名称"

    # 文档相关
    elif any(keyword in field_lower for keyword in ['document', '文档']):
        if any(keyword in field_lower for keyword in ['status', '状态']):
            return "文档状态"
        elif any(keyword in field_lower for keyword in ['type', '类型']):
            return "文档类型"

    # 人员相关
    elif any(keyword in field_lower for keyword in ['creator', '创建人', 'author', '作者', 'user', '用户']):
        return "创建人或作者"
    elif any(keyword in field_lower for keyword in ['name', '名称', 'username', '用户名']):
        return "人员姓名"

    # 时间相关
    elif any(keyword in field_lower for keyword in ['time', '时间', 'date', '日期', 'created', 'updated']):
        if 'create' in field_lower:
            return "创建时间"
        elif 'update' in field_lower:
            return "更新时间"
        elif 'time' in field_lower:
            return "时间字段"
        else:
            return "日期字段"

    # 状态相关
    elif any(keyword in field_lower for keyword in ['status', '状态']):
        return "状态字段"

    # 类型相关
    elif any(keyword in field_lower for keyword in ['type', '类型', 'category', '分类']):
        return "类型或分类"

    # ID相关
    elif any(keyword in field_lower for keyword in ['id', '_id']) and 'uuid' not in field_lower:
        return "主键或标识符"

    # 金额相关
    elif any(keyword in field_lower for keyword in ['amount', '金额', 'price', '价格', 'cost', '费用']):
        return "金额或价格"

    # 描述相关
    elif any(keyword in field_lower for keyword in ['desc', '描述', 'remark', '备注', 'note', '说明']):
        return "描述或备注"

    return ""  # 无法推断时返回空字符串


def get_similar_queries(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """获取相似的历史查询作为few-shot examples"""
    # 这里可以从query_patterns表或向量数据库中获取相似查询
    # 目前返回示例数据
    return [
        {
            "query": "最近30天的订单数量",
            "intent": {
                "primary_metric": "order_count",
                "dimensions": [],
                "filters": ["create_time >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"],
                "limit": 1000,
            },
            "explanation": "查询最近30天的订单总数（意图示例）",
        },
        {
            "query": "各部门平均薪资",
            "intent": {
                "primary_metric": "avg_salary",
                "dimensions": ["department"],
                "filters": [],
                "limit": 1000,
            },
            "explanation": "按部门统计平均薪资（意图示例）",
        }
    ]


def route_tables(query: str, schema_info: Dict[str, Any], top_k: int = 1) -> List[str]:
    """轻量表路由：按表名/字段名匹配简单打分"""
    q = (query or "").lower()
    if not q or not schema_info.get("tables"):
        return []

    scores: list[tuple[str, int]] = []
    for table, info in schema_info["tables"].items():
        score = 0
        if table.lower() in q:
            score += 2
        for col in info.get("columns", []):
            name = str(col.get("name") or "").lower()
            if name and name in q:
                score += 1
        if score > 0:
            scores.append((table, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores:
        return []
    if len(scores) > 1 and (scores[0][1] - scores[1][1]) < 1:
        return []
    return [t for t, _ in scores[:top_k]]


def _extract_json_robust(text: str) -> dict | None:
    """强化的JSON提取，支持多种格式"""
    if not text:
        return None

    text = text.strip()

    # 方法1: 直接JSON解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 方法2: 提取代码块中的JSON
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 方法3: 使用正则表达式提取JSON对象
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            # 尝试修复常见的JSON格式问题
            json_str = _fix_common_json_issues(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 方法4: 尝试提取数组格式
    array_match = re.search(r'(\[.*\])', text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _fix_common_json_issues(json_str: str) -> str:
    """修复常见的JSON格式问题"""
    # 移除尾随逗号
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # 修复单引号为双引号（简单替换，需要谨慎使用）
    # 这里只修复字符串值，不处理键名
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)

    # 移除多余的换行符和空格
    json_str = re.sub(r'\n\s*', ' ', json_str)
    json_str = re.sub(r'\s+', ' ', json_str)

    return json_str.strip()


def _extract_json(text: str) -> dict | None:
    """保持向后兼容的接口，使用强化的JSON提取"""
    return _extract_json_robust(text)


def _filter_columns(expr: str) -> set[str]:
    if not expr:
        return set()
    try:
        ast = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {expr}", dialect="mysql")
    except Exception:
        return set()
    return {c.name.lower() for c in ast.find_all(exp.Column) if getattr(c, "name", None)}


def _normalize_filters(filters: list[str], metric, time_range: TimeRange | None) -> list[str]:
    if not filters:
        return []
    default_filters = (metric.default_filters or "").lower()
    time_cols = {metric.time_column.lower(), "dt"}
    allowed_cols = set(metric.allowed_dims or [])
    allowed_cols.update(time_cols)
    allowed_cols.update(_filter_columns(metric.default_filters or ""))
    allowed_cols.update(_filter_columns(metric.measure_expr or ""))

    # 定义需要过滤掉的时间相关但非数据库列的关键词
    invalid_time_filters = {
        "time_range", "timeframe", "period", "duration",
        "last_60_days", "last_30_days", "last_week", "last_month",
        "recent_days", "recent_months", "past_days", "past_months"
    }

    out: list[str] = []
    for f in filters:
        f_lower = f.lower()
        cols = _filter_columns(f)
        if cols and not cols.issubset({c.lower() for c in allowed_cols}):
            continue
        if not _filter_is_safe(f):
            continue

        # 跳过真正的数据库时间列过滤
        if cols & time_cols:
            continue

        # 跳过无效的时间范围过滤器
        if any(invalid_term in f_lower for invalid_term in invalid_time_filters):
            continue

        if "status" in default_filters and "status" in f_lower and "paid" in f_lower and "paid" in default_filters:
            continue
        out.append(f)
    return out


def plan_with_llm_enhanced(query: str, hinted_metric: Optional[str],
                           allowed_dims: List[str], use_schema: bool = True,
                           use_examples: bool = True,
                           forced_tables: Optional[List[str]] = None,
                           selected_fields: Optional[List[str]] = None) -> QueryPlan:
    """增强的LLM查询解析，支持Schema Prompting和Few-shot Learning"""
    llm = get_llm()
    tr = infer_time_range(query)

    # 构建系统提示词
    sys_prompt = (
        "你是一个业务查询解析助手。\n"
        "任务：把用户问题解析为 JSON 的查询意图，不要输出 SQL。\n"
        "只输出纯JSON格式，不要包含任何解释、注释或代码块标记。\n"
        "字段：primary_metric（可选）, dimensions, filters, limit, suggested_tables, sql_confidence。\n"
        "dimensions 必须从允许集合中选择；如果不确定，输出空数组。\n"
        "filters 数组中的每一项是 SQL WHERE 的布尔表达式片段（例如 status='paid'）。\n"
        "suggested_tables: 建议使用的数据库表名列表。\n"
        "sql_confidence: SQL生成置信度 (0-1)。\n"
        "重要提示：请参考下方Schema信息中的字段语义推断（# 注释），结合业务逻辑进行映射。\n"
        "重要：输出必须是有效的JSON格式，不能包含任何其他内容。\n"
    )

    # 如果启用Schema Prompting，添加数据库结构信息
    schema_info_full = None
    if use_schema:
        schema_info_full = get_database_schema()
        prompt_schema = schema_info_full
        if forced_tables:
            prompt_schema = _filter_schema(schema_info_full, forced_tables)
        elif not hinted_metric:
            candidate_tables = route_tables(query, schema_info_full, top_k=1)
            if candidate_tables:
                prompt_schema = _filter_schema(schema_info_full, candidate_tables)
        schema_prompt = build_schema_prompt(prompt_schema)
        sys_prompt += f"\n{schema_prompt}\n"

    # 构建用户输入
    user_input = {
        "query": query,
        "hinted_metric": hinted_metric,
        "allowed_dimensions": allowed_dims,
        "time_range_hint": tr.label if tr else None,
        "forced_tables": forced_tables or [],
        "selected_fields": selected_fields or [],
    }

    # 如果启用Few-shot Learning，添加示例
    if use_examples and not hinted_metric:
        examples = get_similar_queries(query)
        sys_prompt += "\n参考示例:\n"
        for i, example in enumerate(examples):
            sys_prompt += f"示例{i + 1}:\n"
            sys_prompt += f"查询: {example['query']}\n"
            sys_prompt += f"意图JSON: {json.dumps(example['intent'], ensure_ascii=False)}\n"
            sys_prompt += f"说明: {example['explanation']}\n\n"

    # 添加基于可用维度的智能示例（如果允许维度不为空）
    if allowed_dims:
        sys_prompt += "\n基于当前指标的可用维度参考:\n"
        dim_examples = []

        for dim in allowed_dims:
            semantic_hint = _infer_field_semantics(dim)
            if semantic_hint:
                dim_examples.append(f"• {dim}: {semantic_hint}")

        if dim_examples:
            sys_prompt += "\n".join(dim_examples) + "\n\n"

        # 添加智能映射指导
        sys_prompt += "智能映射提示:\n"
        sys_prompt += "• 项目相关查询（如'XX项目'）→ 使用 project_name 相关的维度\n"
        sys_prompt += "• 物料类型查询（如'电子元件'）→ 使用 material_type 相关的维度\n"
        sys_prompt += "• 状态查询（如'哪个状态'）→ 使用 status 相关的维度作为分组\n"
        sys_prompt += "• 时间查询 → 使用时间相关的字段\n\n"

    # 调用LLM
    full_prompt = sys_prompt + "\n输入:\n" + json.dumps(user_input, ensure_ascii=False)
    resp = llm.invoke(full_prompt)
    txt = resp.content if hasattr(resp, "content") else str(resp)

    # 使用强化的JSON提取
    raw = _extract_json_robust(txt) or {}
    if raw.get("limit") is None:
        raw.pop("limit", None)

    if "sql_confidence" not in raw:
        raw["sql_confidence"] = 0.5
    # 创建QueryPlan对象
    qp = QueryPlan(**raw)

    # 后处理
    if qp.primary_metric is None:
        qp.primary_metric = hinted_metric

    # 如果有allowed_dims，验证维度
    if allowed_dims:
        qp.dimensions = [d for d in qp.dimensions if d in set(allowed_dims)]

    # 验证建议的表是否存在，必要时做表路由兜底
    if use_schema:
        schema_info = schema_info_full or get_database_schema()
        available_tables = set(schema_info["tables"].keys())
        if forced_tables:
            qp.suggested_tables = [t for t in forced_tables if t in available_tables]
        else:
            if qp.suggested_tables:
                qp.suggested_tables = [t for t in qp.suggested_tables if t in available_tables]
            if not qp.suggested_tables:
                qp.suggested_tables = route_tables(query, schema_info, top_k=1)

    qp.limit = min(int(qp.limit or 1000), settings.max_rows)

    return qp


def plan_with_llm(query: str, hinted_metric: str | None, allowed_dims: list[str]) -> QueryPlan:
    """保持向后兼容的接口，内部使用增强版本"""
    return plan_with_llm_enhanced(query, hinted_metric, allowed_dims, use_schema=True, use_examples=False)


def generate_sql_from_plan(query: str, plan: QueryPlan) -> str:
    """根据QueryPlan生成SQL（无预设指标约束）"""
    if not plan.suggested_tables:
        raise ValueError("无法确定查询表，请明确业务对象或表名。")

    # 构建SQL生成提示词
    sql_prompt = (
        f"基于以下查询意图生成MySQL SQL:\n"
        f"查询: {query}\n"
        f"建议表: {', '.join(plan.suggested_tables)}\n"
        f"维度: {plan.dimensions}\n"
        f"过滤条件: {plan.filters}\n"
        f"限制行数: {plan.limit}\n\n"
        f"要求:\n"
        f"1. 只生成SELECT查询\n"
        f"2. 使用建议的表\n"
        f"3. 严格按过滤条件筛选\n"
        f"4. 如果有维度，进行GROUP BY\n"
        f"5. 添加ORDER BY限制结果数量\n"
        f"6. 只输出SQL，不要解释\n"
        f"7. 输出格式：纯SQL代码，不需要代码块标记\n"
    )

    llm = get_llm()
    resp = llm.invoke(sql_prompt)
    txt = resp.content if hasattr(resp, "content") else str(resp)

    # 清理SQL格式
    txt = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", txt.strip())
    return txt.strip()


def controlled_text_to_sql_enhanced(
    query: str,
    hinted_metric: Optional[str] = None,
    forced_tables: Optional[List[str]] = None,
    allowed_dims_override: Optional[List[str]] = None,
    execute_query: bool = True,
    forced_filters: Optional[List[str]] = None,
) -> ExecutionPlan:
    """增强版的受控文本转SQL，支持真正的无指标生成"""
    # 必须有时间范围
    tr = infer_time_range(query)
    if not tr:
        raise ValueError("缺少时间范围（例如：上周/上个月/最近30天/2025-01-01~2025-02-01）。")

    # 如果有提示指标，尝试获取指标信息
    metric = None
    allowed_dims = []
    if hinted_metric:
        metric = get_metric(hinted_metric)
        if metric:
            allowed_dims = metric.allowed_dims
    if allowed_dims_override:
        allowed_dims = list(allowed_dims_override)

    # 使用增强的LLM解析
    plan = plan_with_llm_enhanced(
        query=query,
        hinted_metric=hinted_metric,
        allowed_dims=allowed_dims,
        use_schema=True,  # 启用Schema Prompting
        use_examples=not hinted_metric,  # 未配置指标时使用示例
        forced_tables=forced_tables,
        selected_fields=allowed_dims_override,
    )
    if forced_filters:
        plan.filters = list(forced_filters) + list(plan.filters or [])
    if plan.sql_confidence < settings.l3_min_sql_confidence:
        raise ValueError("SQL 置信度过低，已阻止执行（仅提供意图解析）。")
    schema_info = get_database_schema()
    if plan.suggested_tables:
        allowed_cols = _allowed_columns_for_tables(schema_info, plan.suggested_tables)
    else:
        allowed_cols = set(schema_info.get("all_columns") or [])
    plan.filters = _sanitize_filters(plan.filters, allowed_cols if allowed_cols else None)

    # 根据是否有预设指标选择生成方式
    if metric:
        # 有预设指标，使用传统方式
        qp = plan  # plan已经是QueryPlan对象
        qp.filters = _normalize_filters(qp.filters, metric, tr)

        sql = compile_metric_sql(
            metric=metric,
            time_range=tr,
            dimensions=qp.dimensions,
            filters=qp.filters,
            limit=qp.limit,
        )
        is_schema_guided = False

    else:
        # 无预设指标，使用纯LLM生成
        sql = generate_sql_from_plan(query, plan)
        is_schema_guided = True

        # 验证生成的SQL
        if not sql.upper().startswith("SELECT"):
            raise ValueError("生成的SQL必须为SELECT查询")
        _enforce_single_table(sql)
        _ensure_time_range(sql, tr)

    # SQL验证和修复循环
    for _ in range(settings.max_repair_rounds + 1):
        ok, reasons, _ = validate_select_only(sql)
        if ok:
            break
        sql = repair_sql_with_llm(sql, "; ".join(reasons))

    sql = _ensure_limit(sql, plan.limit)
    if not metric:
        _enforce_single_table(sql)
        _ensure_time_range(sql, tr)
    # 风险评估
    post = assess_post_risk(sql)
    if post.action == "block":
        raise ValueError("SQL 风险过高，已阻止执行：" + "; ".join(post.reasons))
    if plan.sql_confidence < settings.l3_require_confirmation_sql_confidence and post.level == "low":
        post = RiskAssessment(
            level="medium",
            action="require_confirmation",
            reasons=["SQL置信度偏低，需人工确认"],
        )

    # 执行查询
    exp_rows = explain(sql) if execute_query else []
    rows = fetch_all(sql) if execute_query else []

    return ExecutionPlan(
        sql=sql,
        plan=plan,
        explain=exp_rows,
        post_risk=post.__dict__,
        rows=rows,
        is_schema_guided=is_schema_guided
    )


# 兼容旧接口
def controlled_text_to_sql(
    query: str,
    hinted_metric: str | None = None,
    forced_tables: Optional[List[str]] = None,
    allowed_dims_override: Optional[List[str]] = None,
    execute_query: bool = True,
    forced_filters: Optional[List[str]] = None,
) -> ExecutionPlan:
    """保持向后兼容的接口，内部使用增强版本"""
    return controlled_text_to_sql_enhanced(
        query,
        hinted_metric,
        forced_tables=forced_tables,
        allowed_dims_override=allowed_dims_override,
        execute_query=execute_query,
        forced_filters=forced_filters,
    )


def run_sql(sql: str) -> tuple[list[dict], list[dict]]:
    exp_rows = explain(sql)
    rows = fetch_all(sql)
    return exp_rows, rows


def repair_sql_with_llm(sql: str, error: str) -> str:
    llm = get_llm()
    prompt = (
        "你是 SQL 修复助手。目标：修复 SQL 使其可在 MySQL 执行。\n"
        "要求：只允许 SELECT；不得包含任何写操作；必须保留原意；尽量最小改动。\n"
        f"错误信息: {error}\n"
        f"原 SQL:\n{sql}\n"
        "输出：只输出修复后的 SQL，不要解释。"
    )
    resp = llm.invoke(prompt)
    txt = resp.content if hasattr(resp, "content") else str(resp)
    txt = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", txt.strip())
    return txt.strip()


def _ensure_limit(sql: str, limit: int) -> str:
    cleaned = sql.strip().rstrip(";")
    try:
        ast = sqlglot.parse_one(cleaned, dialect="mysql")
    except Exception:
        return f"{cleaned}\nLIMIT {limit}"
    max_limit = min(int(limit), settings.max_rows)
    existing = None
    lim = ast.find(exp.Limit)
    if lim and isinstance(lim.this, exp.Literal) and lim.this.is_number:
        try:
            existing = int(lim.this.this)
        except Exception:
            existing = None
    if existing is not None:
        max_limit = min(max_limit, existing)
    if list(ast.find_all(exp.Limit)) or list(ast.find_all(exp.Fetch)):
        if isinstance(ast, exp.Select):
            ast.set("limit", exp.Limit(this=exp.Literal.number(max_limit)))
            return ast.sql(dialect="mysql")
        if isinstance(ast, (exp.Union, exp.With, exp.Intersect, exp.Except)):
            wrapped = exp.select("*").from_(exp.Subquery(this=ast).as_("t")).limit(max_limit)
            return wrapped.sql(dialect="mysql")
        return cleaned
    if isinstance(ast, exp.Select):
        ast.set("limit", exp.Limit(this=exp.Literal.number(max_limit)))
        return ast.sql(dialect="mysql")
    if isinstance(ast, (exp.Union, exp.With, exp.Intersect, exp.Except)):
        wrapped = exp.select("*").from_(exp.Subquery(this=ast).as_("t")).limit(max_limit)
        return wrapped.sql(dialect="mysql")
    return f"{cleaned}\nLIMIT {max_limit}"


def _filter_is_safe(expr: str) -> bool:
    if not expr:
        return False
    try:
        ast = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {expr}", dialect="mysql")
    except Exception:
        return False
    where = ast.args.get("where")
    if not where:
        return False
    disallowed = (exp.Or, exp.Subquery, exp.Select, exp.Union, exp.Join, exp.Case)
    if any(where.find_all(disallowed)):
        return False
    allowed = [exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Like, exp.Between, exp.Is]
    is_not = getattr(exp, "IsNot", None)
    if isinstance(is_not, type):
        allowed.append(is_not)
    if not any(where.find_all(tuple(allowed))):
        return False
    return True


def _sanitize_filters(filters: list[str], allowed_cols: set[str] | None) -> list[str]:
    if not filters:
        return []
    out: list[str] = []
    for f in filters:
        if not _filter_is_safe(f):
            continue
        cols = _filter_columns(f)
        if allowed_cols and cols and not cols.issubset({c.lower() for c in allowed_cols}):
            continue
        out.append(f)
    return out


def _allowed_columns_for_tables(schema_info: Dict[str, Any], tables: List[str]) -> set[str]:
    cols: set[str] = set()
    for t in tables:
        info = schema_info.get("tables", {}).get(t)
        if not info:
            continue
        for col in info.get("columns", []):
            name = col.get("name")
            if name:
                cols.add(str(name).lower())
    return cols


def _enforce_single_table(sql: str) -> None:
    ast = sqlglot.parse_one(sql, dialect="mysql")
    tables = {t.name.lower() for t in ast.find_all(exp.Table) if getattr(t, "name", None)}
    if len(tables) != 1:
        raise ValueError("无指标模式仅允许单表查询")
    if list(ast.find_all(exp.Join)) or list(ast.find_all(exp.Subquery)) or list(ast.find_all(exp.Union)) or list(ast.find_all(exp.With)):
        raise ValueError("无指标模式禁止 JOIN/子查询/UNION/CTE")


def _ensure_time_range(sql: str, tr: TimeRange) -> None:
    if not tr:
        return
    if tr.start not in sql or tr.end_exclusive not in sql:
        raise ValueError("无指标模式必须包含时间范围条件")


def refresh_schema_cache():
    """手动刷新schema缓存"""
    _schema_cache_manager.force_refresh()


def get_schema_cache_status() -> Dict[str, Any]:
    """获取schema缓存状态信息"""
    import time
    cached_schema = _schema_cache_manager.get_schema()

    return {
        "has_cache": cached_schema is not None,
        "cache_age": time.time() - _schema_cache_manager._last_update if _schema_cache_manager._last_update > 0 else None,
        "tables_count": len(cached_schema["tables"]) if cached_schema else 0,
        "columns_count": len(cached_schema["all_columns"]) if cached_schema else 0
    }

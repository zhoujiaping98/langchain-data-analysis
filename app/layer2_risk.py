from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import sqlglot
from sqlglot import exp

from .config import settings


RiskLevel = Literal["low", "medium", "high", "critical"]
Action = Literal["allow_with_preview", "require_confirmation", "require_approval", "block"]


@dataclass
class RiskAssessment:
    level: RiskLevel
    action: Action
    reasons: list[str]


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


@lru_cache(maxsize=1)
def _risk_rules() -> dict:
    en = _split_csv(settings.risk_block_keywords)
    zh = _split_csv(settings.risk_block_keywords_zh)
    restricted_kw = _split_csv(settings.risk_restricted_keywords)
    restricted_roles = set(_split_csv(settings.risk_restricted_roles))
    sensitive_columns = {v.lower() for v in _split_csv(settings.risk_sensitive_columns)}

    patterns: list[tuple[re.Pattern, str]] = []
    if en:
        patterns.append((re.compile(r"\\b(" + "|".join(map(re.escape, en)) + r")\\b", re.I), "包含潜在写操作关键词"))
    if zh:
        patterns.append((re.compile("|".join(map(re.escape, zh)), re.I), "包含潜在写操作中文关键词"))

    restricted_pattern = None
    if restricted_kw:
        restricted_pattern = re.compile("|".join(map(re.escape, restricted_kw)), re.I)

    return {
        "patterns": patterns,
        "restricted_pattern": restricted_pattern,
        "restricted_roles": restricted_roles,
        "sensitive_columns": sensitive_columns,
    }


def assess_pre_risk(query: str, role: str) -> RiskAssessment:
    rules = _risk_rules()
    for pat, why in rules["patterns"]:
        if pat.search(query):
            return RiskAssessment(level="critical", action="block", reasons=[why])

    if role in rules["restricted_roles"] and rules["restricted_pattern"]:
        if rules["restricted_pattern"].search(query):
            return RiskAssessment(
                level="high",
                action="require_approval",
                reasons=["角色权限不足以查询敏感业务内容"],
            )

    return RiskAssessment(level="low", action="allow_with_preview", reasons=["默认低风险"])


def validate_select_only(sql: str) -> tuple[bool, list[str]]:
    try:
        ast = sqlglot.parse_one(sql, dialect="mysql")
    except Exception as e:
        return False, [f"SQL解析失败: {e}"]

    # Must be SELECT-like
    if not isinstance(ast, exp.Select) and not isinstance(ast, exp.Subqueryable):
        return False, ["仅允许 SELECT 查询"]

    forbidden_names = [
        "Insert",
        "Update",
        "Delete",
        "Drop",
        "Truncate",
        "TruncateTable",
        "Alter",
        "Create",
    ]
    forbidden = tuple(getattr(exp, name) for name in forbidden_names if hasattr(exp, name))
    if any(ast.find_all(forbidden)):
        return False, ["检测到 DML/DDL 语句"]
    return True, []


def assess_post_risk(sql: str) -> RiskAssessment:
    if settings.risk_block_multiple_statements:
        cleaned = sql.strip()
        if ";" in cleaned.rstrip(";"):
            return RiskAssessment(level="critical", action="block", reasons=["检测到多语句 SQL"])

    if settings.risk_max_query_length and len(sql) > settings.risk_max_query_length:
        return RiskAssessment(
            level="high",
            action="require_approval",
            reasons=["SQL 过长，可能存在性能风险"],
        )

    ok, rs = validate_select_only(sql)
    if not ok:
        return RiskAssessment(level="critical", action="block", reasons=rs)

    rules = _risk_rules()
    reasons: list[str] = []
    level: RiskLevel = "low"

    try:
        ast = sqlglot.parse_one(sql, dialect="mysql")
    except Exception as e:
        return RiskAssessment(level="high", action="require_confirmation", reasons=[f"SQL解析失败: {e}"])

    cols = {c.name.lower() for c in ast.find_all(exp.Column) if getattr(c, "name", None)}
    sensitive_used = sorted({c for c in cols if c in rules["sensitive_columns"]})
    if sensitive_used:
        level = "high"
        reasons.append(f"查询涉及敏感字段: {', '.join(sensitive_used)}")

    tables = list(ast.find_all(exp.Table))
    if settings.risk_max_tables and len(tables) > settings.risk_max_tables:
        level = "high"
        reasons.append("涉及表数量过多，可能存在性能风险")

    joins = list(ast.find_all(exp.Join))
    if settings.risk_max_joins and len(joins) > settings.risk_max_joins:
        level = "high"
        reasons.append("JOIN 过多，可能存在性能风险")

    if settings.risk_block_cross_join:
        for j in joins:
            kind = j.args.get("kind")
            if isinstance(kind, str) and kind.upper() == "CROSS":
                return RiskAssessment(level="high", action="require_approval", reasons=["检测到 CROSS JOIN"])

    unions = list(ast.find_all(exp.Union))
    if settings.risk_max_unions and len(unions) > settings.risk_max_unions:
        level = "high"
        reasons.append("UNION 过多，可能存在性能风险")

    subqueries = list(ast.find_all(exp.Subquery))
    if settings.risk_max_subqueries and len(subqueries) > settings.risk_max_subqueries:
        level = "high"
        reasons.append("子查询过多，可能存在性能风险")

    if settings.risk_require_limit and not list(ast.find_all(exp.Limit)):
        level = "medium" if level == "low" else level
        reasons.append("SQL未包含 LIMIT（系统会强制加上上限）")

    if level == "low":
        action: Action = "allow_with_preview"
    elif level == "medium":
        action = "require_confirmation"
    else:
        action = "require_approval"

    return RiskAssessment(level=level, action=action, reasons=reasons or ["OK"])
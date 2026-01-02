from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional, TypedDict

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


# -----------------------------
# Pre-risk keyword patterns
# -----------------------------

_ZH_CORE_DANGEROUS = ["删除", "更新", "插入", "建表", "改表"]
_ZH_NEGATIONS = ["不", "不要", "未", "无", "没有", "禁止", "勿"]


class _RiskPattern(TypedDict):
    pattern: re.Pattern
    reason: str
    keyword: str | None
    is_zh_action: bool


def _compile_en_word_patterns(keywords: list[str]) -> list[_RiskPattern]:
    """
    English dangerous keywords should match word boundaries.
    """
    if not keywords:
        return []
    # \b(word)\b (case-insensitive)
    pat = re.compile(r"\b(" + "|".join(map(re.escape, keywords)) + r")\b", re.I)
    return [
        {
            "pattern": pat,
            "reason": "包含潜在写操作关键词",
            "keyword": None,
            "is_zh_action": False,
        }
    ]


def _compile_zh_action_patterns(keywords: list[str]) -> list[_RiskPattern]:
    """
    Chinese action words:
    - MUST NOT require right boundary as punctuation/space (otherwise '删除订单' won't match)
    - We only require a reasonable left boundary to reduce false positives.
    - Right side can be anything (including Chinese), but we try to avoid matching inside a longer Chinese word by
      requiring left boundary or start-of-string.

    Left boundary set:
      start OR whitespace OR common punctuations.
    """
    if not keywords:
        return []

    left_boundary = r"(^|[\s，。！？、,.;:()\[\]{}<>《》【】])"
    patterns: list[_RiskPattern] = []
    for kw in keywords:
        if not kw:
            continue
        # left boundary + keyword (no strict right boundary)
        pat = re.compile(left_boundary + re.escape(kw), re.I)
        patterns.append(
            {
                "pattern": pat,
                "reason": f"包含潜在写操作中文关键词: {kw}",
                "keyword": kw,
                "is_zh_action": True,
            }
        )
    return patterns


@lru_cache(maxsize=1)
def _risk_rules() -> dict:
    # configurable keywords
    en = _split_csv(settings.risk_block_keywords)
    zh = _split_csv(settings.risk_block_keywords_zh)

    # ensure core zh actions present only when config is empty
    zh_set = []
    seen = set()
    base_zh = zh if zh else _ZH_CORE_DANGEROUS
    for kw in base_zh:
        if not kw:
            continue
        if kw in seen:
            continue
        seen.add(kw)
        zh_set.append(kw)

    restricted_kw = _split_csv(settings.risk_restricted_keywords)
    restricted_roles = set(_split_csv(settings.risk_restricted_roles))
    sensitive_columns = {v.lower() for v in _split_csv(settings.risk_sensitive_columns)}

    patterns: list[_RiskPattern] = []
    if en:
        patterns.extend(_compile_en_word_patterns(en))
    if zh_set:
        patterns.extend(_compile_zh_action_patterns(zh_set))

    restricted_pattern: Optional[re.Pattern] = None
    if restricted_kw:
        restricted_pattern = re.compile("|".join(map(re.escape, restricted_kw)), re.I)

    return {
        "patterns": patterns,
        "restricted_pattern": restricted_pattern,
        "restricted_roles": restricted_roles,
        "sensitive_columns": sensitive_columns,
    }


def assess_pre_risk(query: str, role: str) -> RiskAssessment:
    """
    Pre-risk is to stop obviously dangerous intent BEFORE calling LLM.
    Keep it conservative but not too strict to avoid misses like '删除订单'.
    """
    rules = _risk_rules()
    q = query or ""
    for rule in rules["patterns"]:
        if rule["pattern"].search(q):
            if rule["is_zh_action"] and rule["keyword"]:
                neg_pat = re.compile(rf"({'|'.join(map(re.escape, _ZH_NEGATIONS))})\s*{re.escape(rule['keyword'])}")
                if neg_pat.search(q):
                    continue
            return RiskAssessment(level="critical", action="block", reasons=[rule["reason"]])

    if role in rules["restricted_roles"] and rules["restricted_pattern"]:
        if rules["restricted_pattern"].search(q):
            return RiskAssessment(
                level="high",
                action="require_approval",
                reasons=["角色权限不足以查询敏感业务内容"],
            )

    return RiskAssessment(level="low", action="allow_with_preview", reasons=["默认低风险"])


# -----------------------------
# Post-risk (SQL) validation
# -----------------------------

_FORBIDDEN_NODE_NAMES = [
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Truncate",
    "TruncateTable",
    "Alter",
    "Create",
    "Replace",
    "Merge",
]


def _has_forbidden_nodes(ast: exp.Expression) -> bool:
    forbidden = tuple(getattr(exp, name) for name in _FORBIDDEN_NODE_NAMES if hasattr(exp, name))
    return any(ast.find_all(forbidden))


def _is_query_root(ast: exp.Expression) -> bool:
    """
    Allow common query roots:
    - SELECT
    - WITH (CTE)
    - UNION/INTERSECT/EXCEPT
    """
    allowed_roots = (exp.Select, exp.With, exp.Union, exp.Intersect, exp.Except)
    if isinstance(ast, allowed_roots):
        return True

    return False


def validate_select_only(sql: str) -> tuple[bool, list[str], Optional[exp.Expression]]:
    """
    Parse and validate SQL is query-only (no DML/DDL).
    Return (ok, reasons, ast).
    """
    try:
        ast = sqlglot.parse_one(sql, dialect="mysql")
    except Exception as e:
        return False, [f"SQL解析失败: {e}"], None

    if not _is_query_root(ast):
        return False, ["仅允许 SELECT/CTE/UNION 等查询语句"], ast

    if _has_forbidden_nodes(ast):
        return False, ["检测到 DML/DDL 语句"], ast

    return True, [], ast


def _detect_multiple_statements(sql: str) -> bool:
    """
    Multi-statement detection:
    - Prefer sqlglot.parse() which returns a list of statements.
    - If parsing fails, be conservative and treat as multi-statement (block).
    """
    try:
        stmts = sqlglot.parse(sql, dialect="mysql")
        return len(stmts) > 1
    except Exception:
        # If we can't reliably parse, block (safer for production)
        return True


def assess_post_risk(sql: str) -> RiskAssessment:
    """
    Post-risk is to ensure the generated SQL is safe + reasonable.
    Parse once, reuse AST, keep deterministic.
    """
    s = (sql or "").strip()
    if not s:
        return RiskAssessment(level="critical", action="block", reasons=["SQL为空"])

    # multi-statement
    if settings.risk_block_multiple_statements:
        if _detect_multiple_statements(s):
            return RiskAssessment(level="critical", action="block", reasons=["检测到多语句 SQL"])

    # length guard
    if settings.risk_max_query_length and len(s) > settings.risk_max_query_length:
        return RiskAssessment(
            level="high",
            action="require_approval",
            reasons=["SQL 过长，可能存在性能风险"],
        )

    ok, rs, ast = validate_select_only(s)
    if not ok:
        return RiskAssessment(level="critical", action="block", reasons=rs)

    assert ast is not None

    rules = _risk_rules()
    reasons: list[str] = []
    level: RiskLevel = "low"

    # sensitive columns
    cols: set[str] = set()
    for c in ast.find_all(exp.Column):
        name = getattr(c, "name", None)
        if not name:
            continue
        name_l = name.lower()
        cols.add(name_l)

        # NOTE: c.table might be alias; keep it but do not rely on it exclusively
        table = getattr(c, "table", None)
        if table:
            cols.add(f"{table.lower()}.{name_l}")

    sensitive_used = sorted({c for c in cols if c in rules["sensitive_columns"]})
    if sensitive_used:
        level = "high"
        reasons.append(f"查询涉及敏感字段: {', '.join(sensitive_used)}")

    # tables/joins/unions/subqueries limits (performance guards)
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

        # Additional guard: joins without ON/USING can also be risky
        for j in joins:
            if not j.args.get("on") and not j.args.get("using"):
                level = "high"
                reasons.append("检测到无 ON/USING 的 JOIN，可能存在性能风险")
                break

    unions = list(ast.find_all(exp.Union))
    if settings.risk_max_unions and len(unions) > settings.risk_max_unions:
        level = "high"
        reasons.append("UNION 过多，可能存在性能风险")

    subqueries = list(ast.find_all(exp.Subquery))
    if settings.risk_max_subqueries and len(subqueries) > settings.risk_max_subqueries:
        level = "high"
        reasons.append("子查询过多，可能存在性能风险")

    # limit
    if settings.risk_require_limit:
        has_limit = bool(list(ast.find_all(exp.Limit)) or list(ast.find_all(exp.Fetch)))
        if not has_limit:
            if level == "low":
                level = "medium"
            reasons.append("SQL未包含 LIMIT（系统会强制加上上限）")

    if level == "low":
        action: Action = "allow_with_preview"
    elif level == "medium":
        action = "require_confirmation"
    else:
        action = "require_approval"

    return RiskAssessment(level=level, action=action, reasons=reasons or ["OK"])

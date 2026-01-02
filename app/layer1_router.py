from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from cachetools import TTLCache

from .assets import list_active_metrics, term_registry
from .config import settings
from .vector_index import retrieve_semantic_assets


@dataclass
class Candidate:
    metric_key: str
    score: float  # unified to 0~1 for comparison
    reason: str
    missing_slots: list[str]


@dataclass
class MatchResult:
    route: str  # TEMPLATE | CONTROLLED_T2SQL | ASK_CLARIFY | BLOCK
    confidence: float  # unified 0~1
    best_metric: str | None
    candidates: list[Candidate]
    normalized_query: str
    missing_slots: list[str]
    explanations: list[str]


class HybridRouter:
    """Layer1: Hybrid Router（缓存→规则/词典→RAG）"""

    def __init__(self):
        self.cache = TTLCache(maxsize=4096, ttl=3600)

        self._metrics = list_active_metrics()
        self._metric_trigger_keywords: dict[str, list[str]] = {}

        for m in self._metrics:
            name_keywords: list[str] = []
            if m.metric_key:
                name_keywords.append(m.metric_key)
            if getattr(m, "metric_name_zh", None):
                name_keywords.append(str(m.metric_name_zh))

            if hasattr(m, "trigger_keywords") and m.trigger_keywords:
                kws = m.trigger_keywords or []
                # keep only valid strings
                kws = [kw for kw in kws if isinstance(kw, str) and kw.strip()]
                if kws:
                    self._metric_trigger_keywords[m.metric_key] = kws
                elif name_keywords:
                    self._metric_trigger_keywords[m.metric_key] = name_keywords
            elif name_keywords:
                self._metric_trigger_keywords[m.metric_key] = name_keywords

        self._terms = term_registry()

        # time detection stays lightweight in router
        self._time_patterns = [
            r"上周|本周|上个月|本月|今年|去年|最近\s*\d+\s*天|近\s*\d+\s*天|\d{4}-\d{2}-\d{2}",
            r"过去\s*\d+\s*天|过去\s*\d+\s*个月|过去\s*\d+\s*年|过去\s*一个月|过去\s*一年",
            r"最近\s*一个月|最近\s*一年|上\s*\d+\s*个月|上\s*\d+\s*年",
        ]

        # strong negation only (do not include "除了/排除")
        self._negative_words = ["不", "非", "不是", "无", "没有", "未", "别"]

        # BLOCK should be conservative: only obvious SQL injection / write attempts
        self._write_keywords = ["drop", "delete", "update", "insert", "alter", "truncate", "replace"]
        self._sql_shape_keywords = ["select", "from", "where", "join", "union", "into", "values", "table"]

        # intent re-rank keywords (no score changes)
        self._money_keywords = set()
        self._count_keywords = set()
        if "gmv" in self._metric_trigger_keywords:
            self._money_keywords.update([k.lower() for k in self._metric_trigger_keywords["gmv"]])
        if "order_count" in self._metric_trigger_keywords:
            self._count_keywords.update(self._metric_trigger_keywords["order_count"])
        self._count_keywords.update(["数量", "次数", "笔数"])
        self._money_keywords.update(["金额", "总额", "总金额"])

    # -----------------------------
    # Normalization & helpers
    # -----------------------------

    def normalize(self, query: str) -> str:
        q = (query or "").strip()
        for term, canonical in self._terms.items():
            if term and term in q:
                q = q.replace(term, canonical)
        return q

    def _has_time(self, q: str) -> bool:
        return any(re.search(p, q) for p in self._time_patterns)

    def _clone_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        return [
            Candidate(
                metric_key=c.metric_key,
                score=float(c.score),
                reason=c.reason,
                missing_slots=list(c.missing_slots),
            )
            for c in candidates
        ]

    def _has_negative_words(self, q: str, keyword: str) -> bool:
        # negation + keyword
        pattern = rf"({'|'.join(self._negative_words)})\s*{re.escape(keyword)}"
        return bool(re.search(pattern, q))

    def _word_boundary_match(self, text: str, keyword: str) -> bool:
        """Boundary for alnum keywords; direct contains for pure Chinese keywords."""
        if not keyword:
            return False
        # if keyword contains ascii letters/digits -> strict boundary
        if re.search(r"[A-Za-z0-9]", keyword):
            pattern = rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])"
            return bool(re.search(pattern, text, re.IGNORECASE))
        # chinese -> contains is OK
        return keyword in text

    def _normalize_score_to_01(self, score: float) -> float:
        """
        Make score comparable (0..1).
        If already 0..1, keep; otherwise squash with abs-based transform + sigmoid.
        """
        try:
            s = float(score)
        except Exception:
            return 0.0
        if -1.0 <= s <= 1.0:
            if s <= 0.0:
                return 0.0
            return min(1.0, s)
        if 0.0 <= s <= 1.0:
            return max(0.0, min(1.0, s))
        base = 1.0 / (1.0 + abs(s))  # (0,1]
        # sigmoid sharpen around 0.5
        norm = 1.0 / (1.0 + math.exp(-12.0 * (base - 0.5)))
        return max(0.0, min(1.0, norm))

    def _rag_conf_with_gap(self, scores01: list[float]) -> float:
        """
        scores01 are normalized to 0..1 (higher better), sorted desc.
        conf = top1 + alpha * gap, clamped
        """
        if not scores01:
            return 0.0
        top1 = scores01[0]
        top2 = scores01[1] if len(scores01) > 1 else 0.0
        gap = max(0.0, top1 - top2)
        # alpha chosen to give meaningful boost when gap is clear
        alpha = 0.25
        conf = top1 + alpha * min(gap, 0.2) / 0.2  # gap cap at 0.2
        return max(0.0, min(1.0, conf))

    def _intent_rerank(self, q: str, candidates: list[Candidate]) -> None:
        """Only rerank; do NOT change scores."""
        if not candidates:
            return
        q_lower = q.lower()

        want_gmv = any(k in q_lower for k in self._money_keywords)
        want_count = any(k in q for k in self._count_keywords)

        def move_first(pred):
            for i, c in enumerate(candidates):
                if pred(c):
                    candidates.insert(0, candidates.pop(i))
                    return

        if want_gmv:
            move_first(lambda c: c.metric_key == "gmv")
        if want_count:
            move_first(lambda c: c.metric_key in {"order_count", "count"})

    def _should_block(self, query: str) -> bool:
        """
        Conservative block:
        - obvious write keyword, OR
        - looks like SQL + contains comment/multi-statement tokens
        """
        q = (query or "").lower()

        # obvious write keyword appears as word boundary
        for kw in self._write_keywords:
            if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", q):
                return True

        # "SQL-shaped" + suspicious tokens
        looks_sql = any(k in q for k in self._sql_shape_keywords)
        suspicious = (";" in q) or ("--" in q) or ("/*" in q) or ("*/" in q)
        if looks_sql and suspicious:
            return True

        return False

    # -----------------------------
    # Main route
    # -----------------------------

    def route(self, query: str, user_id: str, role: str) -> MatchResult:
        # 1) block check
        if self._should_block(query):
            return MatchResult(
                route="BLOCK",
                confidence=1.0,
                best_metric=None,
                candidates=[],
                normalized_query=query,
                missing_slots=[],
                explanations=["疑似写操作/注入形态，已被安全策略阻断"],
            )

        q = self.normalize(query)
        cache_key = f"{user_id}:{role}:{q}"

        # 2) cache
        if cache_key in self.cache:
            cached: MatchResult = self.cache[cache_key]
            candidates = self._clone_candidates(cached.candidates)
            self._intent_rerank(q, candidates)
            # keep stable sorting by score desc
            candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
            self._intent_rerank(q, candidates)

            best_metric = candidates[0].metric_key if candidates else cached.best_metric
            confidence = cached.confidence

            return MatchResult(
                route=cached.route,
                confidence=confidence,
                best_metric=best_metric,
                candidates=candidates,
                normalized_query=cached.normalized_query,
                missing_slots=list(cached.missing_slots),
                explanations=list(cached.explanations),
            )

        # 3) rule match
        rule = self._rule_match(q)
        if rule and rule.confidence >= float(settings.router_conf_rule):
            self.cache[cache_key] = rule
            return rule

        # 4) rag match (returns candidates only, normalized to 0..1)
        rag_candidates, rag_conf = self._rag_candidates_and_conf(q)

        # 5) fallback assemble
        missing: list[str] = []
        if not self._has_time(q):
            missing.append("time_range")

        candidates: list[Candidate] = []
        if rule:
            candidates.extend(rule.candidates)
        if rag_candidates:
            candidates.extend(rag_candidates)

        # dedupe by metric_key keep best score
        best_by_key: dict[str, Candidate] = {}
        for c in candidates:
            if c.metric_key not in best_by_key or c.score > best_by_key[c.metric_key].score:
                best_by_key[c.metric_key] = c
        candidates = sorted(best_by_key.values(), key=lambda x: x.score, reverse=True)

        # intent rerank (no score change)
        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        self._intent_rerank(q, candidates)

        # unified confidence
        rule_conf = float(rule.confidence) if rule else 0.0
        final_confidence = max(rag_conf, rule_conf)

        # decide clarify
        needs_clarify = bool(missing)
        if not needs_clarify and len(candidates) >= 2:
            gap = abs(candidates[0].score - candidates[1].score)
            if gap <= 0.03:
                needs_clarify = True

        if candidates and needs_clarify and final_confidence >= float(settings.router_conf_ask):
            res = MatchResult(
                route="ASK_CLARIFY",
                confidence=final_confidence,
                best_metric=candidates[0].metric_key,
                candidates=candidates[:5],
                normalized_query=q,
                missing_slots=missing,
                explanations=["需要你确认指标或补充关键条件（如时间范围/口径）。"],
            )
            self.cache[cache_key] = res
            return res

        # Prefer TEMPLATE if we are confident enough and no missing
        # (Rule already returned above; here we allow rag_conf to promote TEMPLATE)
        if candidates and not missing and rag_conf >= float(settings.router_conf_rag):
            res = MatchResult(
                route="TEMPLATE",
                confidence=rag_conf,
                best_metric=candidates[0].metric_key,
                candidates=candidates[:5],
                normalized_query=q,
                missing_slots=[],
                explanations=["RAG 高置信命中指标定义（含 top1-gap 校验）"],
            )
            self.cache[cache_key] = res
            return res

        res = MatchResult(
            route="CONTROLLED_T2SQL",
            confidence=final_confidence,
            best_metric=(candidates[0].metric_key if candidates else None),
            candidates=candidates[:5],
            normalized_query=q,
            missing_slots=missing,
            explanations=["未能高置信匹配到已定义指标，将进入受控探索查询流程。"],
        )
        self.cache[cache_key] = res
        return res

    # -----------------------------
    # Rule match
    # -----------------------------

    def _rule_match(self, q: str) -> Optional[MatchResult]:
        hits: dict[str, Candidate] = {}

        for metric_key, trigger_keywords in self._metric_trigger_keywords.items():
            best_score = 0.0
            best_keyword = ""

            for keyword in trigger_keywords:
                if self._word_boundary_match(q, keyword):
                    # if negated, downweight rather than drop (more stable)
                    negated = self._has_negative_words(q, keyword)

                    base_score = 0.95 if keyword.lower() == metric_key.lower() else 0.92
                    semantic_weight = self._calculate_semantic_weight(q, keyword, metric_key)
                    score = min(1.0, max(0.0, base_score + semantic_weight))
                    if negated:
                        score *= 0.7  # keep candidate but penalize

                    if score > best_score:
                        best_score = score
                        best_keyword = keyword

            if best_score > 0.0:
                hits[metric_key] = Candidate(
                    metric_key=metric_key,
                    score=best_score,
                    reason=f"规则命中触发词: {best_keyword}",
                    missing_slots=[],
                )

        if not hits:
            return None

        hits_list = sorted(hits.values(), key=lambda x: x.score, reverse=True)

        missing: list[str] = []
        if not self._has_time(q):
            missing.append("time_range")

        route = "TEMPLATE" if not missing else "ASK_CLARIFY"
        return MatchResult(
            route=route,
            confidence=hits_list[0].score,  # already 0..1
            best_metric=hits_list[0].metric_key,
            candidates=hits_list[:5],
            normalized_query=q,
            missing_slots=missing,
            explanations=[hits_list[0].reason],
        )

    def _calculate_semantic_weight(self, query: str, keyword: str, metric_key: str) -> float:
        weight = 0.0

        pos = query.find(keyword)
        if pos >= 0 and len(query) > 0:
            weight += max(0.0, (len(query) - pos) / len(query) * 0.05)

        if any(w in query for w in ["最多", "最少", "数量", "个数", "统计"]):
            if metric_key in ["material_bom_count", "order_count", "active_projects"]:
                weight += 0.03

        dens = 0.0
        for kw in self._metric_trigger_keywords.get(metric_key, []):
            if kw in query:
                dens += 0.01
        weight += min(dens, 0.05)

        if keyword in ["物料", "物料清单", "BOM清单"] and "物料" in query:
            if metric_key == "material_bom_count":
                weight += 0.04

        return min(weight, 0.1)

    # -----------------------------
    # RAG candidates + conf (top1-gap)
    # -----------------------------

    def _rag_candidates_and_conf(self, q: str) -> tuple[list[Candidate], float]:
        hits = retrieve_semantic_assets(q, top_k=8)
        if not hits:
            return [], 0.0

        # collect raw scores by metric, keep best raw score per metric
        best_raw: dict[str, float] = {}
        raw_by_metric: dict[str, list[float]] = {}
        observed: list[float] = []
        for h in hits:
            if getattr(h, "kind", None) != "metric":
                continue
            mk = (h.payload or {}).get("meta", {}).get("metric_key")
            if not mk:
                continue
            s = float(getattr(h, "score", 0.0) or 0.0)
            observed.append(s)
            raw_by_metric.setdefault(mk, []).append(s)

        if not best_raw:
            return [], 0.0

        # heuristics: if all scores are >=0 and some > 1, treat as distance (lower better)
        is_distance = bool(observed) and min(observed) >= 0.0 and max(observed) > 1.0
        for mk, values in raw_by_metric.items():
            best_raw[mk] = min(values) if is_distance else max(values)

        # normalize each score to 0..1 so candidates sort is meaningful
        candidates: list[Candidate] = []
        scores01: list[float] = []
        for mk, raw in best_raw.items():
            s01 = self._normalize_score_to_01(raw)
            candidates.append(
                Candidate(
                    metric_key=mk,
                    score=s01,
                    reason="RAG 召回指标定义",
                    missing_slots=[],
                )
            )
            scores01.append(s01)

        candidates.sort(key=lambda x: x.score, reverse=True)
        scores01_sorted = [c.score for c in candidates]
        rag_conf = self._rag_conf_with_gap(scores01_sorted)

        return candidates[:5], rag_conf

from __future__ import annotations

import re
from dataclasses import dataclass

from cachetools import TTLCache

from .assets import list_active_metrics, term_registry
from .config import settings
from .vector_index import retrieve_semantic_assets


@dataclass
class Candidate:
    metric_key: str
    score: float
    reason: str
    missing_slots: list[str]


@dataclass
class MatchResult:
    route: str  # TEMPLATE | CONTROLLED_T2SQL | ASK_CLARIFY | BLOCK
    confidence: float
    best_metric: str | None
    candidates: list[Candidate]
    normalized_query: str
    missing_slots: list[str]
    explanations: list[str]


class HybridRouter:
    """层：Hybrid Router（缓存→规则/词典→RAG）"""

    def __init__(self):
        self.cache = TTLCache(maxsize=4096, ttl=3600)

        self._metrics = list_active_metrics()
        self._metric_keywords: dict[str, str] = {}
        for m in self._metrics:
            self._metric_keywords[m.metric_key.lower()] = m.metric_key
            if m.metric_name_zh:
                self._metric_keywords[m.metric_name_zh] = m.metric_key

        self._terms = term_registry()
        self._time_patterns = [
            r"上周|本周|上个月|本月|今年|去年|最近\s*\d+\s*天|近\s*\d+\s*天|\d{4}-\d{2}-\d{2}",
        ]

    def normalize(self, query: str) -> str:
        q = query.strip()
        for term, canonical in self._terms.items():
            if term and term in q:
                q = q.replace(term, canonical)
        return q

    def _has_time(self, q: str) -> bool:
        return any(re.search(p, q) for p in self._time_patterns)

    def _intent_boost(self, q: str, candidates: list[Candidate]) -> None:
        if not candidates:
            return
        money_kw = ["金额", "总额", "总金额", "GMV", "收入", "营收"]
        count_kw = ["数量", "订单数", "次数", "笔数"]

        q_lower = q.lower()
        want_gmv = any(k.lower() in q_lower for k in money_kw)
        want_count = any(k in q for k in count_kw)

        for c in candidates:
            if want_gmv and c.metric_key == "gmv":
                c.score += 0.08
            if want_count and c.metric_key in {"order_count", "count"}:
                c.score += 0.06

        if want_gmv:
            for i, c in enumerate(candidates):
                if c.metric_key == "gmv":
                    candidates.insert(0, candidates.pop(i))
                    break

    def route(self, query: str, user_id: str, role: str) -> MatchResult:
        q = self.normalize(query)
        cache_key = f"{user_id}:{role}:{q}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if cached.candidates:
                self._intent_boost(q, cached.candidates)
                cached.candidates = sorted(cached.candidates, key=lambda x: x.score, reverse=True)
                cached.best_metric = cached.candidates[0].metric_key
            return cached

        # 1) rule match
        rule = self._rule_match(q)
        if rule and rule.confidence >= settings.router_conf_rule:
            self.cache[cache_key] = rule
            return rule

        # 2) rag match
        rag = self._rag_match(q)
        if rag and rag.confidence >= settings.router_conf_rag:
            self.cache[cache_key] = rag
            return rag

        # 3) fallback
        missing = []
        if not self._has_time(q):
            missing.append("time_range")

        candidates = []
        if rag:
            candidates.extend(rag.candidates)
        if rule:
            candidates.extend(rule.candidates)
        # de-dup by metric_key keep max score
        best_by_key = {}
        for c in candidates:
            if c.metric_key not in best_by_key or c.score > best_by_key[c.metric_key].score:
                best_by_key[c.metric_key] = c
        candidates = sorted(best_by_key.values(), key=lambda x: x.score, reverse=True)

        if candidates:
            self._intent_boost(q, candidates)
            candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

        if candidates and (rag.confidence if rag else 0.0) >= settings.router_conf_ask:
            res = MatchResult(
                route="ASK_CLARIFY",
                confidence=(rag.confidence if rag else 0.55),
                best_metric=candidates[0].metric_key,
                candidates=candidates[:5],
                normalized_query=q,
                missing_slots=missing,
                explanations=["低置信度匹配：需要你确认指标或补充条件。"],
            )
            self.cache[cache_key] = res
            return res

        res = MatchResult(
            route="CONTROLLED_T2SQL",
            confidence=max((rag.confidence if rag else 0.0), (rule.confidence if rule else 0.0)),
            best_metric=(candidates[0].metric_key if candidates else None),
            candidates=candidates[:5],
            normalized_query=q,
            missing_slots=missing,
            explanations=["未能高置信度匹配到已定义指标，将进入受控探索查询流程。"],
        )
        self.cache[cache_key] = res
        return res

    def _rule_match(self, q: str) -> MatchResult | None:
        hits: list[Candidate] = []
        for kw, metric_key in self._metric_keywords.items():
            if kw and kw in q:
                score = 0.95 if kw == metric_key else 0.92
                hits.append(
                    Candidate(
                        metric_key=metric_key,
                        score=score,
                        reason=f"规则命中关键词 {kw}",
                        missing_slots=[],
                    )
                )

        if not hits:
            return MatchResult(
                route="CONTROLLED_T2SQL",
                confidence=0.0,
                best_metric=None,
                candidates=[],
                normalized_query=q,
                missing_slots=[],
                explanations=[],
            )

        hits.sort(key=lambda x: x.score, reverse=True)
        missing = []
        if not self._has_time(q):
            missing.append("time_range")

        route = "TEMPLATE" if not missing else "ASK_CLARIFY"
        return MatchResult(
            route=route,
            confidence=hits[0].score,
            best_metric=hits[0].metric_key,
            candidates=hits[:5],
            normalized_query=q,
            missing_slots=missing,
            explanations=[hits[0].reason],
        )

    def _rag_match(self, q: str) -> MatchResult | None:
        hits = retrieve_semantic_assets(q, top_k=8)
        metric_candidates: list[Candidate] = []

        for h in hits:
            if h.kind != "metric":
                continue
            mk = h.payload.get("meta", {}).get("metric_key")
            if not mk:
                continue
            metric_candidates.append(
                Candidate(
                    metric_key=mk,
                    score=h.score,
                    reason="RAG 召回指标定义",
                    missing_slots=[],
                )
            )

        if not metric_candidates:
            return MatchResult(
                route="CONTROLLED_T2SQL",
                confidence=0.0,
                best_metric=None,
                candidates=[],
                normalized_query=q,
                missing_slots=[],
                explanations=[],
            )

        metric_candidates.sort(key=lambda x: x.score, reverse=True)
        best = metric_candidates[0]
        missing = []
        if not self._has_time(q):
            missing.append("time_range")

        route = (
            "TEMPLATE"
            if best.score >= settings.router_conf_rag and not missing
            else ("ASK_CLARIFY" if missing else "CONTROLLED_T2SQL")
        )
        return MatchResult(
            route=route,
            confidence=best.score,
            best_metric=best.metric_key,
            candidates=metric_candidates[:5],
            normalized_query=q,
            missing_slots=missing,
            explanations=["RAG 召回到相似指标定义"],
        )

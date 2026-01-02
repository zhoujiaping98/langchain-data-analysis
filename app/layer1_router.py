from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

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
        self._metric_trigger_keywords: dict[str, list[str]] = {}

        # 构建关键词映射和触发词映射
        for m in self._metrics:
            # 基础关键词映射
            self._metric_keywords[m.metric_key.lower()] = m.metric_key
            if m.metric_name_zh:
                self._metric_keywords[m.metric_name_zh] = m.metric_key

            # 触发关键词映射（从配置文件加载）
            if hasattr(m, 'trigger_keywords') and m.trigger_keywords:
                trigger_keywords = m.trigger_keywords or []
                self._metric_trigger_keywords[m.metric_key] = trigger_keywords
                for kw in trigger_keywords:
                    self._metric_keywords[kw.lower()] = m.metric_key

        self._terms = term_registry()
        self._time_patterns = [
            r"上周|本周|上个月|本月|今年|去年|最近\s*\d+\s*天|近\s*\d+\s*天|\d{4}-\d{2}-\d{2}",
            r"过去\s*\d+\s*天|过去\s*\d+\s*个月|过去\s*\d+\s*年|过去\s*一个月|过去\s*一年",
            r"最近\s*一个月|最近\s*一年|上\s*\d+\s*个月|上\s*\d+\s*年",
        ]

        # 否定词列表
        self._negative_words = ["不", "非", "不是", "无", "没有", "未", "别", "非", "排除", "除", "除了"]

        # 敏感词黑名单
        self._blacklist_keywords = [
            "删除", "drop", "delete", "password", "密码", "身份证", "credit_card",
            "信用卡", "ssn", "社保", "私人", "隐私", "保密", "敏感", "内部"
        ]

        # 构建配置化的关键词集合，用于词边界匹配判断
        self._all_trigger_keywords = set()
        for trigger_keywords in self._metric_trigger_keywords.values():
            self._all_trigger_keywords.update(trigger_keywords)

        # 构建意图关键词集合（从配置中动态提取）
        self._money_keywords = set()
        self._count_keywords = set()

        # 从GMV相关指标提取金额关键词
        if "gmv" in self._metric_trigger_keywords:
            self._money_keywords.update(self._metric_trigger_keywords["gmv"])

        # 从订单相关指标提取数量关键词
        if "order_count" in self._metric_trigger_keywords:
            self._count_keywords.update(self._metric_trigger_keywords["order_count"])

        # 添加通用的数量相关关键词
        self._count_keywords.update(["数量", "次数", "笔数"])
        # 添加通用的金额相关关键词
        self._money_keywords.update(["金额", "总额", "总金额"])

    def normalize(self, query: str) -> str:
        q = query.strip()
        for term, canonical in self._terms.items():
            if term and term in q:
                q = q.replace(term, canonical)
        return q

    def _has_time(self, q: str) -> bool:
        return any(re.search(p, q) for p in self._time_patterns)

    def _has_negative_words(self, q: str, keyword: str) -> bool:
        """检查关键词前是否有否定词"""
        # 使用正则表达式查找否定词 + 关键词的模式
        pattern = rf"({'|'.join(self._negative_words)})\s*{re.escape(keyword)}"
        return bool(re.search(pattern, q))

    def _word_boundary_match(self, text: str, keyword: str) -> bool:
        """使用词边界进行匹配，避免子串误匹配"""
        # 检查是否包含关键词
        if keyword not in text:
            return False

        pos = text.find(keyword)
        if pos < 0:
            return False

        # 获取前后字符
        before_char = text[pos - 1] if pos > 0 else ""
        after_char = text[pos + len(keyword)] if pos + len(keyword) < len(text) else ""

        # 检查是否为词边界的辅助函数
        def is_chinese_char(char):
            """检查字符是否为中文"""
            return ord(char) >= 0x4e00 and ord(char) <= 0x9fff

        def is_english_letter(char):
            """检查字符是否为英文字母"""
            return char.isalpha() and ord(char) < 128

        def is_digit(char):
            """检查字符是否为数字"""
            return char.isdigit()

        # 判断关键词是否为中文
        keyword_is_chinese = all(is_chinese_char(c) for c in keyword if c)

        # 基于配置化的触发关键词进行特殊处理
        if keyword in self._all_trigger_keywords:
            # 对于英文关键词，中文字符不算边界
            if keyword.lower() == 'bom':
                # BOM前面如果是中文是OK的，后面如果是中文也是OK的
                before_ok = True  # BOM前面通常是中文
                after_ok = True  # BOM后面通常是中文
                # 但是如果是连续字母数字则不算边界
                if before_char and before_char.isalnum() and not is_chinese_char(before_char):
                    before_ok = False
                if after_char and after_char.isalnum() and not is_chinese_char(after_char):
                    after_ok = False
                return before_ok or after_ok

            # 对于中文关键词
            if keyword_is_chinese:
                # 中文关键词前后如果是字母数字或英文则不算边界
                before_ok = not (before_char.isalnum() and (is_english_letter(before_char) or is_digit(before_char)))
                after_ok = not (after_char.isalnum() and (is_english_letter(after_char) or is_digit(after_char)))
                return before_ok or after_ok

        # 对于其他关键词，使用简单包含匹配
        return keyword in text

    def _normalize_rag_score(self, rag_score: float) -> float:
        """将RAG分数归一化到0.6-0.95区间，与Rule分数可比"""
        # 处理负数分数和异常值
        if rag_score < -100:  # 非常低的分数，直接设置为最低置信度
            return 0.6

        # 使用sigmoid函数进行归一化，将-200到0的分数映射到0.6-0.95
        # 首先将分数映射到0-1区间
        normalized_input = (rag_score + 200) / 200  # 将-200到0映射到0到1
        normalized_input = max(0, min(1, normalized_input))  # 确保在0-1范围内

        # 使用sigmoid函数
        normalized = 0.6 + 0.35 * (1 / (1 + math.exp(-10 * (normalized_input - 0.5))))
        return min(0.95, max(0.6, normalized))

    def _intent_boost(self, q: str, candidates: list[Candidate]) -> None:
        if not candidates:
            return

        q_lower = q.lower()
        # 使用配置化的意图关键词
        want_gmv = any(k.lower() in q_lower for k in self._money_keywords)
        want_count = any(k in q for k in self._count_keywords)

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

    def _check_blacklist(self, query: str) -> bool:
        """检查查询是否包含敏感词，需要阻断"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self._blacklist_keywords)

    def route(self, query: str, user_id: str, role: str) -> MatchResult:
        # 1. 首先检查黑名单
        if self._check_blacklist(query):
            return MatchResult(
                route="BLOCK",
                confidence=1.0,
                best_metric=None,
                candidates=[],
                normalized_query=query,
                missing_slots=[],
                explanations=["查询包含敏感词汇，已被安全策略阻断"]
            )

        q = self.normalize(query)
        cache_key = f"{user_id}:{role}:{q}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if cached.candidates:
                self._intent_boost(q, cached.candidates)
                cached.candidates = sorted(cached.candidates, key=lambda x: x.score, reverse=True)
                cached.best_metric = cached.candidates[0].metric_key
            return cached

        # 2. 规则匹配
        rule = self._rule_match(q)
        if rule and rule.confidence >= settings.router_conf_rule:
            self.cache[cache_key] = rule
            return rule

        # 3. RAG匹配（带归一化置信度）
        rag = self._rag_match(q)
        if rag:
            # 对RAG分数进行归一化
            rag.confidence = self._normalize_rag_score(rag.confidence)
            if rag.confidence >= settings.router_conf_rag:
                self.cache[cache_key] = rag
                return rag

        # 4. 处理fallback情况
        missing = []
        if not self._has_time(q):
            missing.append("time_range")

        candidates = []
        if rag:
            candidates.extend(rag.candidates)
        if rule:
            candidates.extend(rule.candidates)

        # 去重并保留最高分数的候选
        best_by_key = {}
        for c in candidates:
            if c.metric_key not in best_by_key or c.score > best_by_key[c.metric_key].score:
                best_by_key[c.metric_key] = c
        candidates = sorted(best_by_key.values(), key=lambda x: x.score, reverse=True)

        if candidates:
            self._intent_boost(q, candidates)
            candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

        # 使用归一化后的RAG置信度
        final_confidence = max(
            (rag.confidence if rag else 0.0),
            (rule.confidence if rule else 0.0)
        )

        if candidates and final_confidence >= settings.router_conf_ask:
            res = MatchResult(
                route="ASK_CLARIFY",
                confidence=final_confidence,
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
            confidence=final_confidence,
            best_metric=(candidates[0].metric_key if candidates else None),
            candidates=candidates[:5],
            normalized_query=q,
            missing_slots=missing,
            explanations=["未能高置信度匹配到已定义指标，将进入受控探索查询流程。"],
        )
        self.cache[cache_key] = res
        return res

    def _rule_match(self, q: str) -> Optional[MatchResult]:
        hits: dict[str, Candidate] = {}  # 使用dict避免重复，保留最高分

        # 使用配置化的触发关键词进行匹配
        for metric_key, trigger_keywords in self._metric_trigger_keywords.items():
            best_score = 0.0
            best_keyword = ""

            for keyword in trigger_keywords:
                # 使用词边界匹配，避免子串误匹配
                if self._word_boundary_match(q, keyword):
                    # 检查是否有否定词
                    if self._has_negative_words(q, keyword):
                        continue  # 跳过被否定的关键词

                    # 动态计算分数：基础分数 + 语义权重
                    base_score = 0.95 if keyword.lower() == metric_key.lower() else 0.92

                    # 语义权重计算
                    semantic_weight = self._calculate_semantic_weight(q, keyword, metric_key)
                    final_score = base_score + semantic_weight

                    if final_score > best_score:
                        best_score = final_score
                        best_keyword = keyword

            # 如果该指标有匹配的关键词，添加到结果中
            if best_score > 0.0:
                # 检查是否已存在该指标，保留最高分
                if metric_key not in hits or best_score > hits[metric_key].score:
                    hits[metric_key] = Candidate(
                        metric_key=metric_key,
                        score=best_score,
                        reason=f"配置化规则命中关键词 {best_keyword}",
                        missing_slots=[],
                    )

        # 如果没有命中任何规则，返回None而不是兜底结果
        if not hits:
            return None

        # 转换为列表并排序
        hits_list = list(hits.values())
        hits_list.sort(key=lambda x: x.score, reverse=True)

        missing = []
        if not self._has_time(q):
            missing.append("time_range")

        route = "TEMPLATE" if not missing else "ASK_CLARIFY"
        return MatchResult(
            route=route,
            confidence=hits_list[0].score,
            best_metric=hits_list[0].metric_key,
            candidates=hits_list[:5],
            normalized_query=q,
            missing_slots=missing,
            explanations=[hits_list[0].reason],
        )

    def _calculate_semantic_weight(self, query: str, keyword: str, metric_key: str) -> float:
        """计算语义权重，提高查询理解准确性"""
        weight = 0.0

        # 1. 关键词位置权重（前面的关键词更重要）
        keyword_pos = query.find(keyword)
        if keyword_pos >= 0:
            # 位置越靠前权重越高
            position_weight = max(0, (len(query) - keyword_pos) / len(query) * 0.05)
            weight += position_weight

        # 2. 语义相关性权重
        # 数量相关查询更倾向于material_bom_count等计数指标
        if any(count_word in query for count_word in ["最多", "最少", "数量", "个数", "统计"]):
            if metric_key in ["material_bom_count", "order_count", "active_projects"]:
                weight += 0.03

        # 3. 关键词密度权重（一个查询中同一指标关键词越多，权重越高）
        keyword_density = 0.0
        for kw in self._metric_trigger_keywords.get(metric_key, []):
            if kw in query:
                keyword_density += 0.01  # 每个匹配的关键词增加0.01

        weight += min(keyword_density, 0.05)  # 最多增加0.05

        # 4. 特殊处理"物料"类关键词的优先级
        if keyword in ["物料", "物料清单", "BOM清单"] and "物料" in query:
            if metric_key == "material_bom_count":
                weight += 0.04

        # 5. 避免过度权重
        return min(weight, 0.1)  # 总权重不超过0.1

    def _rag_match(self, q: str) -> Optional[MatchResult]:
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
            return None

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
            confidence=best.score,  # 原始分数，后续会归一化
            best_metric=best.metric_key,
            candidates=metric_candidates[:5],
            normalized_query=q,
            missing_slots=missing,
            explanations=["RAG 召回到相似指标定义"],
        )
"""Guardrail 服务 — Responsible Agentic AI 的核心。

实现 4 类防护:
  1. 输入验证 (input validation): city/dates/days 合法性
  2. PII 红act (Personal Identifiable Information): 手机/身份证/邮箱/银行卡
  3. Prompt injection 检测: "ignore previous"/"system prompt"/"forget all"
  4. 敏感主题黑名单: 政治敏感 / 暴力 / 违法

输入路径: 在 plan_trip 入口先过 InputGuardrail.check
输出路径: 在 _assemble_plan 后过 OutputGuardrail.check (验证 LLM 没引入未授权字段)

设计为纯算法,零外部依赖,可单独测试。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..models.schemas import TripPlan, TripRequest

logger = logging.getLogger(__name__)


# ============ 数据结构 ============

@dataclass
class GuardrailResult:
    """单次防护检查结果。"""
    passed: bool
    severity: str  # "ok" / "warn" / "block"
    violations: List[str] = field(default_factory=list)
    sanitized_text: Optional[str] = None
    redacted_count: int = 0

    @property
    def should_block(self) -> bool:
        return self.severity == "block"

    @property
    def has_warning(self) -> bool:
        return self.severity == "warn" or bool(self.violations)


# ============ 规则库 ============

# PII 模式 (中国)
_PII_PATTERNS = [
    (re.compile(r"1[3-9]\d{9}"), "[手机号已隐藏]"),
    (re.compile(r"\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"), "[身份证已隐藏]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[邮箱已隐藏]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[银行卡已隐藏]"),
]

# Prompt injection 关键词 (中英双语)
_INJECTION_KEYWORDS = [
    # 经典越狱模式
    "ignore previous", "ignore above", "disregard prior",
    "system prompt", "system message", "you are now", "you must now",
    "forget all", "forget previous", "new instructions",
    "act as", "pretend to be",
    # 中文
    "忽略以上", "忽略之前", "无视前面",
    "系统提示", "系统消息", "你现在是", "你必须是",
    "扮演", "假装是", "新指令",
    # JSON / role 注入
    '"role":"system"', '"system_prompt"',
]

# 敏感主题黑名单
_SENSITIVE_TOPICS = [
    "黄赌毒", "枪支", "毒品交易", "炸药", "自杀", "杀人",
    "drug deal", "weapon", "explosive", "suicide method",
]

# 城市名白名单 (国内大陆主流城市,逐步扩展)
_CITY_WHITELIST = {
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都", "重庆",
    "西安", "武汉", "天津", "青岛", "厦门", "三亚", "丽江", "桂林", "长沙",
    "济南", "郑州", "沈阳", "大连", "哈尔滨", "昆明", "贵阳", "南宁", "福州",
    "南昌", "合肥", "石家庄", "太原", "兰州", "西宁", "银川", "乌鲁木齐",
    "拉萨", "海口", "呼和浩特", "长春", "无锡", "宁波", "佛山", "东莞",
    "珠海", "中山", "汕头", "湖州", "嘉兴", "绍兴", "金华", "台州", "温州",
    "黄山", "九江", "洛阳", "开封", "曲阜", "敦煌", "张家界",
}


# ============ Input Guardrail ============

class InputGuardrail:
    """检查用户的 TripRequest 是否安全合法。"""

    def check(self, request: TripRequest) -> GuardrailResult:
        violations: List[str] = []
        severity = "ok"

        # 1. 城市合法性 (允许白名单外但 warn)
        if request.city and request.city not in _CITY_WHITELIST:
            violations.append(f"非热门城市'{request.city}',结果可能不准")
            severity = "warn"

        # 2. 旅行天数 (1-15 合理,超过当作 block)
        if request.travel_days < 1 or request.travel_days > 15:
            violations.append(f"travel_days={request.travel_days} 超出合理范围 [1,15]")
            severity = "block"

        # 3. 日期格式 + 不能在过去
        from datetime import datetime, date
        try:
            start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
            if start < date.today():
                violations.append("start_date 在过去")
                severity = "warn"  # 历史日期 warn 不 block (天气数据可能仍可用)
        except (ValueError, TypeError):
            violations.append("start_date 格式非法")
            severity = "block"

        # 4. free_text 检查 PII / injection / 敏感
        free_text = request.free_text_input or ""
        sanitized, redacted_count = self._redact_pii(free_text)
        if redacted_count > 0:
            violations.append(f"自由文本含 {redacted_count} 处 PII,已脱敏")
            if severity == "ok":
                severity = "warn"

        if self._has_injection(free_text):
            violations.append("自由文本疑似 Prompt Injection,已拦截")
            sanitized = ""  # 直接清空,避免污染 LLM
            severity = "block"

        if self._has_sensitive_topic(free_text):
            violations.append("自由文本含敏感主题,已拒绝")
            severity = "block"

        result = GuardrailResult(
            passed=(severity != "block"),
            severity=severity,
            violations=violations,
            sanitized_text=sanitized if sanitized != free_text else None,
            redacted_count=redacted_count,
        )
        if result.has_warning:
            logger.warning(
                "🛡️ InputGuardrail [%s]: %s",
                severity, "; ".join(violations),
            )
        return result

    @staticmethod
    def _redact_pii(text: str) -> tuple:
        if not text:
            return text, 0
        out = text
        count = 0
        for pat, replacement in _PII_PATTERNS:
            new_out, n = pat.subn(replacement, out)
            count += n
            out = new_out
        return out, count

    @staticmethod
    def _has_injection(text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        return any(kw in lower for kw in _INJECTION_KEYWORDS)

    @staticmethod
    def _has_sensitive_topic(text: str) -> bool:
        if not text:
            return False
        return any(kw in text for kw in _SENSITIVE_TOPICS)


# ============ Output Guardrail ============

class OutputGuardrail:
    """检查 LLM 生成的 TripPlan 是否被篡改/越权。

    LLM 应该只改 description/meals/overall_suggestions,不能引入未授权字段。
    特别检查"description 语义一致性": LLM 不能在 description 里描述本天景点列表
    之外的地点(防止 LLM 抄 few-shot 示例或被 free_text 带偏)。
    """

    # 中文地点名匹配正则: 2-8 字 + 常见后缀
    _LOCATION_PATTERN = re.compile(
        r"[一-龥]{2,8}(?:门|宫|城|塔|寺|园|湖|山|场|路|街|巷|楼|馆|阁|台|院)"
    )

    def check(self, plan: TripPlan, allowed_poi_ids: set) -> GuardrailResult:
        violations: List[str] = []
        severity = "ok"

        for day_idx, day in enumerate(plan.days):
            attraction_names = [a.name for a in day.attractions]
            day_no = day_idx + 1

            for attr in day.attractions:
                # LLM 不能引入新景点 (poi_id 必须在白名单)
                if attr.poi_id and attr.poi_id not in allowed_poi_ids:
                    violations.append(f"Day{day_no} 出现未授权景点 poi_id={attr.poi_id}")
                    severity = "warn"
                # 价格上下限
                if attr.ticket_price < 0 or attr.ticket_price > 5000:
                    violations.append(
                        f"景点 '{attr.name}' 票价 ¥{attr.ticket_price} 不合理"
                    )
                    severity = "warn"

            # 餐饮估价
            for meal in day.meals:
                if meal.estimated_cost < 0 or meal.estimated_cost > 2000:
                    violations.append(
                        f"Day{day_no} {meal.type} '{meal.name}' 估价 ¥{meal.estimated_cost} 不合理"
                    )
                    severity = "warn"

            # ⚠️ description 语义一致性检查 (防 LLM 脱离数据)
            inconsistency = self._check_description_consistency(day.description, attraction_names)
            if inconsistency:
                violations.append(
                    f"Day{day_no} description 提到 {inconsistency} 但本天景点是 {attraction_names}"
                )
                severity = "warn"

        # 总体建议长度
        if plan.overall_suggestions and len(plan.overall_suggestions) > 2000:
            violations.append("overall_suggestions 过长,可能 LLM 注入")
            severity = "warn"

        result = GuardrailResult(
            passed=(severity != "block"),
            severity=severity,
            violations=violations,
        )
        if result.has_warning:
            logger.warning(
                "🛡️ OutputGuardrail [%s]: %s",
                severity, "; ".join(violations[:5]),
            )
        return result

    @classmethod
    def _check_description_consistency(
        cls, description: Optional[str], attraction_names: List[str],
    ) -> List[str]:
        """检查 description 提到的地名是否都在 attraction_names 里。

        返回不一致的"未授权地名"列表(去重)。
        """
        if not description or not attraction_names:
            return []
        # 提取 description 中的所有"地名候选"
        candidates = set(cls._LOCATION_PATTERN.findall(description))
        if not candidates:
            return []
        # 已授权地名集合(包括 attraction 名字本身和它的 2-gram 子串)
        authorized = set()
        for name in attraction_names:
            authorized.add(name)
            # 加入 2-4 字子串匹配 (e.g. "故宫博物院" → 故宫)
            for length in range(2, min(5, len(name) + 1)):
                for i in range(len(name) - length + 1):
                    authorized.add(name[i:i + length])

        unmatched = []
        for cand in candidates:
            # 如果 candidate 是任一授权地名的子串/超串,视为匹配
            ok = any(cand in auth or auth in cand for auth in authorized)
            if not ok:
                unmatched.append(cand)

        # 只报告 ≥2 处不一致才告警(允许 1 个误报,如"酒店"会被误抓)
        return unmatched if len(unmatched) >= 2 else []


# ============ 单例 ============

_input_guardrail: Optional[InputGuardrail] = None
_output_guardrail: Optional[OutputGuardrail] = None


def get_input_guardrail() -> InputGuardrail:
    global _input_guardrail
    if _input_guardrail is None:
        _input_guardrail = InputGuardrail()
    return _input_guardrail


def get_output_guardrail() -> OutputGuardrail:
    global _output_guardrail
    if _output_guardrail is None:
        _output_guardrail = OutputGuardrail()
    return _output_guardrail

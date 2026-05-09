"""多智能体旅行规划系统 (4 Agent 协作版)。

四个 Agent 协作完成行程规划:

  ┌────────────────────────┐  ┌────────────────────────┐
  │ AttractionSearchAgent  │  │ HotelRecommendAgent    │
  │ (算法,无 LLM 调用)      │  │ (算法,无 LLM 调用)      │
  │ poi_aggregator 拉数据   │  │ around_search 找酒店    │
  └───────────┬────────────┘  └───────────┬────────────┘
              │                            │
              ↓                            ↓
  ┌────────────────────────┐  ┌────────────────────────┐
  │ WeatherQueryAgent      │  │ TripPlannerAgent       │
  │ (LLM + MCP 工具)       │  │ (LLM, 仅做文案润色)     │
  │ maps_weather 拿天气    │  │ 严格不改景点/坐标/价格 │
  └────────────────────────┘  └────────────────────────┘

设计原则:
- 算法 Agent 用真实数据决策（避免 LLM 幻觉）
- LLM Agent 仅做"包装"职责（文案润色、工具调用代理）
- 中间还有不属于任何 Agent 的算法层: itinerary_optimizer (聚类+TSP) + 硬规则后处理
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from hello_agents import SimpleAgent
from hello_agents.tools import MCPTool

logger = logging.getLogger(__name__)

from ..config import get_settings
from ..models.schemas import (
    Attraction,
    DayPlan,
    Hotel,
    Location,
    Meal,
    POIInfo,
    TripPlan,
    TripRequest,
    WeatherInfo,
)
from ..data.keywords import (
    HOTEL_TIER_KEYWORDS,
    infer_hotel_tier,
    is_blacklisted_hotel,
)
from ..data.curated_food import get_curated_food
from ..services.amap_service import get_amap_service
from ..services.evaluation_service import get_evaluator
from ..services.guardrail_service import get_input_guardrail, get_output_guardrail
from ..services.hotel_pricing import estimate_hotel, normalize_tier
from ..services.image_cache import get_image_cache
from ..services.itinerary_optimizer import optimize
from ..services.llm_service import get_llm
from ..services.poi_aggregator import collect_attractions, _enrich_with_detail
from ..services.rag_service import get_rag
from ..services.amap_rest_service import get_weather_rest, search_pois_around_rest

# ============ Agent提示词 ============

WEATHER_AGENT_PROMPT = """你是天气查询专家。你的任务是查询指定城市的天气信息。

**重要提示:**
你必须使用工具来查询天气!不要自己编造天气信息!

**工具调用格式:**
使用maps_weather工具时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_weather:city=城市名]`

**示例:**
用户: "查询北京天气"
你的回复: [TOOL_CALL:amap_maps_weather:city=北京]
"""


PLANNER_AGENT_PROMPT = """你是行程规划文案专家。你的任务是为已经确定好的行程生成生动的描述文字、餐饮建议和总体建议。

**核心约束（必须严格遵守）:**
1. 我会给你"已经选定的景点列表（含 poi_id、坐标、评分、地址）"和"已经选定的酒店（含名称、地址、价格区间、预订链接）"。
   你**只能为它们生成文案**，绝对不能新增景点、不能修改名称、不能修改任何坐标或价格字段。
2. 你的回复必须是合法 JSON,严格按照下面的"输出格式"。
3. 餐饮（meals）由你结合景点位置就近推荐，每天必须包含早中晚三餐，给出 estimated_cost。
4. weather_info 数组每天一项，已经在输入里给你了，请原样回填，不要修改温度数字（不要带°C等单位）。
5. **若输入中存在 free_text_input（用户的特殊诉求,如"想看升旗仪式"）**：
   - 必须在 description 或 overall_suggestions 中**显式落地这个诉求**（如"第1天清晨5点前到天安门广场观看升旗仪式（夏季约5:00,冬季约7:00）"）
   - 如果诉求需要特定时间安排,要在对应天的 description 里说明
   - 不要忽略这个字段,不要敷衍带过

**输出格式:**
```json
{
  "city": "城市",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述,一段连贯的文字",
      "transportation": "交通方式建议",
      "accommodation": "住宿类型",
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "...", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "...", "estimated_cost": 60},
        {"type": "dinner", "name": "晚餐推荐", "description": "...", "estimated_cost": 100}
      ],
      "attractions": [
        {"poi_id": "xxx", "ticket_price": 60, "visit_duration": 120, "description": "为该景点写一段50字以内的简介"}
      ]
    }
  ],
  "overall_suggestions": "总体建议:着装/出行/避坑..."
}
```

**注意事项:**
1. attractions 数组里只输出 poi_id + ticket_price + visit_duration + description 四个字段，
   其它信息我会从我的候选数据里回填，不要改 poi_id。
2. ticket_price 可以根据知名景点的常识给出整数估算（不知道则给 0），单位是元。
3. visit_duration 是建议游览时间，单位分钟，常见在 60-180 之间。
4. description 文案紧扣景点特色，避免泛泛而谈。

⚠️ **最重要的硬约束 — 防止 LLM 脱离数据幻觉:**
- 每天的 description **只能描述本次输入 attractions 列表里的景点**,
  哪怕用户的 free_text_input 提到了其他地方,也不要写进 description。
- 例: 如果当天 attractions 是 [颐和园, 国家自然博物馆],
      description 就只能围绕"颐和园 + 国家自然博物馆"展开,
      绝对不能引入"故宫""天安门"等本天列表里没有的景点。
- description 中提到的地名必须 100% 是当天景点列表里的名字。
- 如果用户特殊诉求(如"想看升旗")在某天的景点列表里能落地,就在该天 description
  里写时间安排; 如果完全不在任何天的景点列表里,就在 overall_suggestions 提一句
  "本次行程未安排此项,建议另行规划",不要硬塞进 description。

**Few-shot 输出格式范例 (仅展示 JSON 结构,不要复用任何具体场景):**

输入示例: 某天 attractions=[POI_X, POI_Y], hotel=ABC
输出格式:
{
  "day_index": 0,
  "description": "上午 8:30 抵达 [POI_X 实际名称],建议先去 [POI_X 主要看点],游览约 [时长];
                  中午在 ABC 酒店或周边用餐;下午 14:00 前往 [POI_Y 实际名称],
                  重点 [POI_Y 主要看点],傍晚返回酒店。",
  "transportation": "[根据距离推荐合适交通方式]",
  "meals": [
    {"type": "breakfast", "name": "...", "description": "...", "estimated_cost": ...},
    {"type": "lunch", "name": "...", "description": "...", "estimated_cost": ...},
    {"type": "dinner", "name": "...", "description": "...", "estimated_cost": ...}
  ],
  "attractions": [
    {"poi_id": "POI_X 的真实 id", "ticket_price": ..., "visit_duration": ..., "description": "..."},
    {"poi_id": "POI_Y 的真实 id", "ticket_price": ..., "visit_duration": ..., "description": "..."}
  ]
}

description 写作要求:
- ✓ 含具体时间(如"8:30 到""14:00 前往")
- ✓ 含价格/票价提示(如"门票 ¥60""停车 ¥10")
- ✓ 含动线建议(从 X 出口出 → 步行 N 米到 Y)
- ✓ 含避坑/小技巧(如"周一闭馆""提前网上预约")
- ✗ 不要泛泛而谈("景色很美""不容错过")
- ✗ 不要引入本天 attractions 列表外的景点
"""


# ============ 算法 Agent 内部使用的辅助函数 ============

def _pick_best_hotel(candidates: List[POIInfo], target_tier: str) -> Optional[POIInfo]:
    """按 rating + 档位匹配 + 知名连锁品牌 综合打分,选最优酒店。"""
    if not candidates:
        return None

    def score(p: POIInfo) -> float:
        s = 0.0
        # rating 权重最高
        if p.rating is not None:
            s += p.rating * 2.0
        else:
            s += 6.0  # 缺失给基础分
        # 档位匹配 +5
        inferred_tier = infer_hotel_tier(p.name, fallback_pref="")
        if inferred_tier and inferred_tier == target_tier:
            s += 5.0
        # 含知名连锁品牌词 +3
        for tier_kws in HOTEL_TIER_KEYWORDS.values():
            if any(kw in p.name for kw in tier_kws):
                s += 3.0
                break
        # 名字含"酒店"二字 +1
        if "酒店" in p.name:
            s += 1.0
        return s

    return sorted(candidates, key=score, reverse=True)[0]


def _select_hotels_for_days(
    amap_service,
    days_pois: List[List[POIInfo]],
    request: TripRequest,
) -> List[Optional[Hotel]]:
    """对每天景点的质心做 around_search 找酒店,补全 detail,综合打分选优,补档位估算价。"""
    from concurrent.futures import ThreadPoolExecutor

    hotels: List[Optional[Hotel]] = []
    target_tier = normalize_tier(request.accommodation or "舒适型")

    def fetch_candidates(centroid: Location, radius: int) -> List[POIInfo]:
        raw = search_pois_around_rest(
            centroid, keywords="酒店", city=request.city, radius=radius, limit=15
        )
        if not raw and get_settings().enable_mcp_tools:
            try:
                raw = amap_service.around_search(
                    centroid, keywords="酒店", radius=radius
                )
            except Exception as exc:
                logger.warning("酒店周边搜索失败 radius=%d err=%s", radius, exc)
                return []
        # 黑名单过滤（驿站/招待所/民居等明显非主流）
        filtered = [p for p in raw if p.name and not is_blacklisted_hotel(p.name)]
        need_detail = [p for p in filtered[:8] if p.location is None or p.rating is None]
        no_detail = [p for p in filtered[:8] if p not in need_detail]
        if need_detail:
            with ThreadPoolExecutor(max_workers=4) as pool:
                enriched = no_detail + list(pool.map(
                    lambda p: _enrich_with_detail(p, amap_service),
                    need_detail,
                ))
        else:
            enriched = no_detail
        # 必须有 location（用于距离展示）
        return [p for p in enriched if p.location is not None]

    for i, day_pois in enumerate(days_pois):
        if not day_pois:
            hotels.append(None)
            continue
        # 当日景点质心
        n = len(day_pois)
        centroid = Location(
            longitude=sum(p.location.longitude for p in day_pois) / n,
            latitude=sum(p.location.latitude for p in day_pois) / n,
        )

        # 半径 2km → 失败再扩到 5km
        candidates = fetch_candidates(centroid, 2000)
        if not candidates:
            candidates = fetch_candidates(centroid, 5000)

        chosen = _pick_best_hotel(candidates, target_tier)
        if chosen is None:
            anchor = day_pois[0].name if day_pois else request.city
            pricing = estimate_hotel(
                hotel_name=f"{request.city}{anchor}周边酒店",
                city=request.city,
                accommodation_pref=request.accommodation or "舒适型",
            )
            hotels.append(Hotel(
                name=f"{anchor}周边住宿区域",
                address=f"建议在{anchor}周边2-5公里或地铁沿线筛选",
                location=centroid,
                price_range=pricing["price_range"],
                rating="",
                distance="按当日景点中心筛选,以预订平台实时距离为准",
                type=pricing["tier"],
                estimated_cost=pricing["estimated_cost"],
                price_source="reference",
                booking_url=pricing["booking_url"],
            ))
            continue

        pricing = estimate_hotel(
            hotel_name=chosen.name,
            city=request.city,
            accommodation_pref=request.accommodation or "舒适型",
        )
        from ..services.itinerary_optimizer import haversine
        distance_km = haversine(centroid, chosen.location)
        hotels.append(Hotel(
            name=chosen.name,
            address=chosen.address or "",
            location=chosen.location,
            price_range=pricing["price_range"],
            rating=str(chosen.rating) if chosen.rating is not None else "",
            distance=f"距当日景点中心约{distance_km:.1f}公里",
            type=pricing["tier"],
            estimated_cost=pricing["estimated_cost"],
            price_source=pricing["price_source"],
            booking_url=pricing["booking_url"],
        ))
        logger.info(
            "第%d天选定酒店: %s (rating=%s, tier=%s, 距质心=%.1fkm)",
            i + 1, chosen.name, chosen.rating, pricing["tier"], distance_km,
        )
    return hotels


def _fetch_weather(amap_service, request: TripRequest) -> List[WeatherInfo]:
    """通过 amap_service 拉天气,按 travel_days 补齐。

    高德 maps_weather 仅返回 4 天预报(当天+未来3天)。超出范围时复用最后一天 +
    标记"(参考)",而不是返回空温度让前端显示 0°C。
    """
    weather = get_weather_rest(request.city)
    if not weather and get_settings().enable_mcp_tools:
        try:
            weather = amap_service.get_weather(request.city)
        except Exception as exc:
            logger.warning("天气查询失败: %s", exc)
            weather = []
    if not weather:
        logger.warning("天气 API 暂无可用结果,使用城市季节气候参考")
        return _build_climate_reference_weather(request)

    out: List[WeatherInfo] = []
    start = datetime.strptime(request.start_date, "%Y-%m-%d")
    fallback_used = 0

    for i in range(request.travel_days):
        target_date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        match = next((w for w in weather if w.date == target_date), None)
        if match is not None:
            out.append(match)
            continue

        # 日期对不上但还在 4 天预报内 — 按顺序映射
        if i < len(weather):
            w = weather[i]
            out.append(WeatherInfo(
                date=target_date,
                day_weather=w.day_weather,
                night_weather=w.night_weather,
                day_temp=w.day_temp,
                night_temp=w.night_temp,
                wind_direction=w.wind_direction,
                wind_power=w.wind_power,
            ))
            continue

        # 超出 4 天预报范围 — 复用 out 中最后一个有效天 + "(参考)" 标记
        fallback_used += 1
        ref = next((w for w in reversed(out) if w.day_weather), None)
        if ref is not None:
            out.append(WeatherInfo(
                date=target_date,
                day_weather=_with_reference_suffix(ref.day_weather),
                night_weather=_with_reference_suffix(ref.night_weather),
                day_temp=ref.day_temp,
                night_temp=ref.night_temp,
                wind_direction=ref.wind_direction,
                wind_power=ref.wind_power,
            ))
        else:
            # 极端情况: 高德也没返回有效数据
            out.append(WeatherInfo(
                date=target_date,
                day_weather="预报暂无",
                night_weather="预报暂无",
            ))

    if fallback_used > 0:
        logger.info("天气: 高德仅返回 %d 天预报, %d 天用最近一天作参考", len(weather), fallback_used)
    return out


def _with_reference_suffix(text: str) -> str:
    base = (text or "天气待更新").replace(" (参考)", "").replace("（参考）", "")
    if "参考" in base:
        return base
    return f"{base}（参考）"


_CITY_MONTH_CLIMATE: Dict[str, Dict[int, tuple]] = {
    # city -> month -> (day_weather, night_weather, day_temp, night_temp)
    "北京": {
        1: ("晴冷", "晴冷", 2, -8), 2: ("晴冷", "晴冷", 6, -5), 3: ("晴到多云", "多云", 13, 1),
        4: ("晴到多云", "多云", 21, 8), 5: ("晴到多云", "多云", 27, 15), 6: ("多云", "多云", 31, 20),
        7: ("多云有雷阵雨可能", "多云", 32, 23), 8: ("多云有阵雨可能", "多云", 30, 22),
        9: ("晴到多云", "多云", 26, 16), 10: ("晴到多云", "晴", 19, 8), 11: ("晴冷", "晴冷", 10, 0), 12: ("晴冷", "晴冷", 3, -6),
    },
    "上海": {
        1: ("阴到多云", "阴到多云", 9, 3), 2: ("阴到多云", "阴到多云", 11, 5), 3: ("多云有小雨可能", "阴", 15, 8),
        4: ("多云", "多云", 21, 13), 5: ("多云到阴", "多云", 26, 18), 6: ("阴雨或梅雨", "阴雨", 28, 22),
        7: ("多云炎热", "多云", 34, 27), 8: ("多云炎热", "多云", 33, 27), 9: ("多云", "多云", 29, 23),
        10: ("多云", "多云", 24, 17), 11: ("多云", "多云", 18, 11), 12: ("多云偏冷", "多云", 11, 5),
    },
    "西安": {
        1: ("晴冷", "晴冷", 5, -4), 2: ("晴到多云", "晴冷", 9, -1), 3: ("晴到多云", "多云", 16, 5),
        4: ("晴到多云", "多云", 23, 11), 5: ("晴到多云", "多云", 28, 16), 6: ("晴热", "多云", 33, 21),
        7: ("多云炎热", "多云", 35, 24), 8: ("多云炎热", "多云", 33, 23), 9: ("多云", "多云", 26, 18),
        10: ("晴到多云", "多云", 20, 11), 11: ("晴冷", "晴冷", 12, 3), 12: ("晴冷", "晴冷", 6, -3),
    },
    "杭州": {
        1: ("阴到多云", "阴到多云", 9, 2), 2: ("阴到多云", "阴", 11, 4), 3: ("多云有小雨可能", "阴", 16, 8),
        4: ("多云", "多云", 22, 13), 5: ("多云", "多云", 27, 18), 6: ("阴雨或梅雨", "阴雨", 29, 22),
        7: ("多云炎热", "多云", 35, 27), 8: ("多云炎热", "多云", 34, 26), 9: ("多云", "多云", 30, 22),
        10: ("多云", "多云", 24, 16), 11: ("多云偏凉", "多云", 18, 10), 12: ("多云偏冷", "多云", 11, 4),
    },
    "成都": {
        1: ("阴到多云", "阴", 10, 4), 2: ("阴到多云", "阴", 13, 6), 3: ("多云", "阴", 18, 10),
        4: ("多云", "阴", 24, 15), 5: ("多云", "阴", 28, 19), 6: ("多云有阵雨可能", "阴", 30, 22),
        7: ("多云有阵雨可能", "阴", 32, 24), 8: ("多云有阵雨可能", "阴", 32, 24), 9: ("多云", "阴", 26, 20),
        10: ("阴到多云", "阴", 21, 15), 11: ("阴到多云", "阴", 16, 10), 12: ("阴到多云", "阴", 11, 5),
    },
}


def _build_climate_reference_weather(request: TripRequest) -> List[WeatherInfo]:
    start = datetime.strptime(request.start_date, "%Y-%m-%d")
    default_by_month = {
        1: ("多云偏冷", "多云偏冷", 8, 0), 2: ("多云偏冷", "多云偏冷", 10, 2),
        3: ("多云", "多云", 16, 7), 4: ("多云", "多云", 22, 13),
        5: ("多云", "多云", 27, 18), 6: ("多云有阵雨可能", "多云", 30, 22),
        7: ("多云炎热", "多云", 33, 25), 8: ("多云炎热", "多云", 32, 25),
        9: ("多云", "多云", 28, 20), 10: ("多云", "多云", 23, 14),
        11: ("多云偏凉", "多云", 16, 7), 12: ("多云偏冷", "多云", 10, 2),
    }
    out: List[WeatherInfo] = []
    city_table = _CITY_MONTH_CLIMATE.get(request.city, {})
    for i in range(request.travel_days):
        current = start + timedelta(days=i)
        day_weather, night_weather, day_temp, night_temp = city_table.get(
            current.month,
            default_by_month[current.month],
        )
        out.append(WeatherInfo(
            date=current.strftime("%Y-%m-%d"),
            day_weather=f"{day_weather}（气候参考）",
            night_weather=f"{night_weather}（气候参考）",
            day_temp=day_temp,
            night_temp=night_temp,
            wind_direction="以实时预报为准",
            wind_power="以实时预报为准",
        ))
    return out


# ============ Agent 基类 ============

class BaseAgent:
    """所有 Agent 的基类,统一日志格式让 4 Agent 协作清晰可见。"""

    name: str = "Agent"

    def run(self, *args, **kwargs):
        logger.info("🤖 %s 开始工作...", self.name)
        try:
            result = self._run(*args, **kwargs)
            logger.info("✅ %s 完成", self.name)
            return result
        except Exception:
            logger.exception("❌ %s 失败", self.name)
            raise

    def _run(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


# ============ 4 个具体 Agent ============

class AttractionSearchAgent(BaseAgent):
    """景点搜索 Agent (算法 Agent,不调 LLM)。
    内部委托给 poi_aggregator: 高德 POI 拉数据 → 评分排序 → A 级加权 → free_text 强制保留。
    """
    name = "景点搜索专家"

    def _run(self, request: TripRequest) -> List[POIInfo]:
        top_n = max(request.travel_days * 4 + 6, 12)
        return collect_attractions(
            city=request.city,
            preferences=request.preferences,
            free_text=request.free_text_input,
            top_n=top_n,
        )


class HotelRecommendAgent(BaseAgent):
    """酒店推荐 Agent (算法 Agent,不调 LLM)。
    内部用 amap.around_search 找质心附近真实酒店 → 黑名单过滤 → detail 补 rating
    → 综合打分(评分 + 档位匹配 + 连锁品牌)。
    """
    name = "酒店推荐专家"

    def __init__(self, amap_service):
        self.amap_service = amap_service

    def _run(
        self,
        days_pois: List[List[POIInfo]],
        request: TripRequest,
    ) -> List[Optional[Hotel]]:
        return _select_hotels_for_days(self.amap_service, days_pois, request)


class WeatherQueryAgent(BaseAgent):
    """天气查询 Agent (LLM + MCP 工具)。
    保留 SimpleAgent + maps_weather 的可调用入口,但默认走 amap_service.get_weather
    直接拿结构化数据(更可靠,避免 LLM 解析 MCP 字符串出错)。
    """
    name = "天气查询专家"

    def __init__(self, llm, amap_tool, amap_service):
        self.llm = llm
        self.amap_tool = amap_tool
        self.amap_service = amap_service
        self.simple_agent = None
        if llm is not None and amap_tool is not None:
            # SimpleAgent 仅作课程演示入口;真实查询走 REST/amap_service
            self.simple_agent = SimpleAgent(
                name=self.name,
                llm=llm,
                system_prompt=WEATHER_AGENT_PROMPT,
            )
            self.simple_agent.add_tool(amap_tool)

    def _run(self, request: TripRequest) -> List[WeatherInfo]:
        return _fetch_weather(self.amap_service, request)


class TripPlannerAgent(BaseAgent):
    """行程文案 Agent (LLM 包装 + Agentic RAG)。
    给 LLM 已经选定的景点/酒店/天气,让它仅生成 description / meals /
    overall_suggestions 等"软"字段。生成时通过 RAG 检索旅行知识库,
    把"升旗时间""穿衣建议""防坑指南"等真实知识注入 prompt。
    """
    name = "行程文案专家"

    def __init__(self, llm):
        self.llm = llm
        self.simple_agent = None
        if llm is not None:
            self.simple_agent = SimpleAgent(
                name=self.name,
                llm=llm,
                system_prompt=PLANNER_AGENT_PROMPT,
            )

    def _run(self, prompt: str) -> str:
        if self.simple_agent is None:
            raise RuntimeError("LLM planner is disabled")
        return self.simple_agent.run(prompt)


# ============ Responsible AI / Reasoning / Learning Agents ============

class GuardrailAgent(BaseAgent):
    """护栏 Agent (Responsible AI)。
    输入端: 验证 + PII 红act + Prompt injection 拦截
    输出端: 验证 LLM 没引入未授权字段 / 价格不离谱
    """
    name = "护栏专家"

    def __init__(self):
        self.input_gr = get_input_guardrail()
        self.output_gr = get_output_guardrail()

    def _run(self, mode: str, payload):
        if mode == "input":
            return self.input_gr.check(payload)
        if mode == "output":
            plan, allowed_ids = payload
            return self.output_gr.check(plan, allowed_ids)
        raise ValueError(f"unknown mode: {mode}")


class OrchestratorAgent(BaseAgent):
    """调度 Agent (Reason 阶段核心)。
    职责:
      1. Intent classification: 判断用户意图 (深度文化游 / 网红打卡 / 亲子 / 商务等)
      2. Task decomposition (CoT): 把"3 天北京"拆成可执行步骤
      3. Workflow routing: 决定 agents 调用顺序和参数

    用 LLM-free 的规则引擎实现,避免引入额外 LLM 延迟。
    后续可平滑升级为 LLM-driven (system prompt + few-shot)。
    """
    name = "意图调度专家"

    INTENT_RULES = [
        # (关键词, intent_label, 推荐 preferences 加权)
        (["升旗", "故宫", "长城", "古迹", "历史"], "深度文化游", ["历史文化"]),
        (["拍照", "网红", "打卡", "出片"], "网红打卡", ["艺术", "夜景"]),
        (["亲子", "孩子", "宝宝", "小孩"], "亲子游", ["亲子"]),
        (["美食", "好吃", "小吃"], "美食探店", ["美食"]),
        (["登山", "徒步", "户外"], "户外探险", ["自然", "运动"]),
        (["商务", "出差", "会议"], "商务出行", []),
    ]

    def _run(self, request: TripRequest) -> Dict[str, Any]:
        free_text = request.free_text_input or ""
        prefs = list(request.preferences or [])

        # 1. Intent classification
        intent = "通用观光"
        boost_prefs: List[str] = []
        for keywords, label, boost in self.INTENT_RULES:
            if any(kw in free_text for kw in keywords):
                intent = label
                boost_prefs = boost
                break

        # 2. 偏好增强 (CoT-style: 已识别意图 → 推导补充偏好)
        merged_prefs = list(dict.fromkeys(prefs + boost_prefs))  # 去重保序

        # 3. Workflow plan (Chain-of-Thought 推导出的执行步骤)
        steps = self._decompose(intent, request)

        plan = {
            "intent": intent,
            "merged_preferences": merged_prefs,
            "execution_steps": steps,
            "agent_sequence": [
                "GuardrailAgent(input)",
                "AttractionSearchAgent",
                "itinerary_optimizer + 硬规则",
                "HotelRecommendAgent",
                "WeatherQueryAgent",
                "TripPlannerAgent (with RAG)",
                "GuardrailAgent(output)",
                "EvaluationAgent",
            ],
        }
        logger.info(
            "🧠 意图识别: %s | 增强偏好: %s | %d 个执行步骤",
            intent, merged_prefs, len(steps),
        )
        return plan

    @staticmethod
    def _decompose(intent: str, request: TripRequest) -> List[str]:
        """CoT 风格的任务拆解 (展示给用户/日志,可扩展成 LLM-driven)。"""
        return [
            f"识别意图为「{intent}」,目的地={request.city},天数={request.travel_days}",
            "并发拉取高德 POI 候选,按 5A>4A>评分综合打分",
            "若用户提及特殊景点关键词,强制纳入候选池(如升旗→天安门广场)",
            "k-means 地理聚类成 N 天,每天 TSP 求最短路径",
            "硬规则后处理:升旗仪式必排第 1 天首位",
            "对每天景点质心做 around_search,按品牌/档位综合打分选酒店",
            "MCP 工具拉真实天气预报",
            "RAG 检索旅行知识库 → LLM 生成可执行文案(含具体时间/票价/防坑)",
            "Output Guardrail 验证 LLM 没越权",
            "自动评估: 知名度/紧凑度/酒店覆盖/图片注入/诉求响应",
        ]


class EvaluationAgent(BaseAgent):
    """自动评估 Agent (Learn 阶段)。
    给生成的 TripPlan 打分,把分数写回 TripPlan 元数据,
    并通过日志输出(供 observability 监控)。
    低分会触发 warnings,可作为后续 RL fine-tune 的 feedback signal。
    """
    name = "质量评估专家"

    def __init__(self):
        self.evaluator = get_evaluator()

    def _run(self, request: TripRequest, plan: TripPlan):
        return self.evaluator.evaluate(request, plan)


# ============ 主类 ============

class MultiAgentTripPlanner:
    """多智能体旅行规划系统。"""

    def __init__(self):
        logger.info("初始化多智能体旅行规划系统 (4 Agent 协作)...")

        try:
            settings = get_settings()
            self.llm = (
                get_llm()
                if settings.enable_llm_planner or settings.enable_mcp_tools
                else None
            )
            self.amap_service = get_amap_service()

            self.amap_tool = None
            if settings.enable_mcp_tools:
                # 创建共享的 MCP 工具(课程演示用,主链路默认 REST-first)
                self.amap_tool = MCPTool(
                    name="amap",
                    description="高德地图服务",
                    server_command=["uvx", "amap-mcp-server"],
                    env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
                    auto_expand=True,
                )
                self.amap_tool.expandable = True

            # 7 个 Agent 协作:
            #   核心算法 Agent (无 LLM):
            self.attraction_agent = AttractionSearchAgent()
            self.hotel_agent = HotelRecommendAgent(self.amap_service)
            #   LLM Agent:
            self.weather_agent = WeatherQueryAgent(
                self.llm, self.amap_tool, self.amap_service
            )
            self.planner_agent = TripPlannerAgent(self.llm)
            #   Reason / Responsible / Learn:
            self.orchestrator_agent = OrchestratorAgent()
            self.guardrail_agent = GuardrailAgent()
            self.evaluation_agent = EvaluationAgent()
            self.name = "多智能体旅行规划系统"

            # RAG 单例预热 (启动时加载知识库)
            get_rag()

            logger.info(
                "多智能体系统初始化成功 (7 Agent: %s)",
                " / ".join([
                    self.orchestrator_agent.name,
                    self.guardrail_agent.name,
                    self.attraction_agent.name,
                    self.hotel_agent.name,
                    self.weather_agent.name,
                    self.planner_agent.name,
                    self.evaluation_agent.name,
                ]),
            )

        except Exception:
            logger.exception("多智能体系统初始化失败")
            raise

    # ============ 主流程 ============

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """主入口 — 完整 Agentic AI 4 stage cycle。

        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Perceive │→ │  Reason  │→ │  Action  │→ │   Learn  │
        └──────────┘  └──────────┘  └──────────┘  └──────────┘
            │              │              │              │
        Guardrail     Orchestrator   Attraction +    Evaluation
        + RAG ctx     intent +CoT    Hotel + Weather  + warnings
                                     + Planner LLM
        """
        try:
            logger.info(
                "═══ 开始规划 city=%s 日期=%s~%s days=%d 偏好=%s ═══",
                request.city, request.start_date, request.end_date,
                request.travel_days,
                ",".join(request.preferences) if request.preferences else "无",
            )

            # ──────── PERCEIVE 阶段: 输入护栏 + 上下文检索 ────────
            logger.info("🔵 [Perceive] 输入护栏检查...")
            input_check = self.guardrail_agent.run("input", request)
            if not input_check.passed:
                logger.error("🛑 输入被护栏拦截: %s", input_check.violations)
                return self._create_fallback_plan(
                    request,
                    reason="输入未通过安全检查: " + "; ".join(input_check.violations),
                )
            # PII 脱敏后的 free_text 替换原值
            if input_check.sanitized_text is not None:
                request = request.model_copy(
                    update={"free_text_input": input_check.sanitized_text}
                )

            # ──────── REASON 阶段: 意图识别 + 任务分解 (CoT) ────────
            logger.info("🟡 [Reason] 调度 Agent 做意图识别 + 任务分解...")
            routing = self.orchestrator_agent.run(request)
            # 用增强后的 preferences 替换原 request
            request = request.model_copy(
                update={"preferences": routing["merged_preferences"]}
            )

            # ──────── ACTION 阶段: 4 个 specialist agent 协作 ────────
            logger.info("🟢 [Action] 4 个 specialist Agent 并行/顺序协作...")

            # A1: 景点 Agent
            candidates = self.attraction_agent.run(request)
            if not candidates:
                logger.warning("未拉到任何景点候选，回落到 fallback 计划")
                return self._create_fallback_plan(request)

            # 中间层算法: 聚类 + TSP + 硬规则 (不属于任何 Agent)
            logger.info("中间层: 聚类 + TSP + 硬规则")
            days_pois = optimize(
                candidates,
                n_days=request.travel_days,
                max_per_day=3,
            )
            themed_days = self._compose_themed_days(candidates, request)
            if themed_days:
                days_pois = themed_days
            days_pois = self._apply_must_first_rules(days_pois, request)
            # 必去地标打散到不同天 (避免聚类把超级地标全堆 day0,后面冷清)
            if not themed_days:
                days_pois = self._spread_must_visit_across_days(days_pois, request)
            days_pois = self._isolate_remote_attractions(days_pois)
            from ..services.itinerary_optimizer import order_within_day
            days_pois = [order_within_day(day) for day in days_pois]
            days_pois = self._apply_must_first_rules(days_pois, request)
            days_pois = self._isolate_remote_attractions(days_pois)
            days_pois = self._cap_daily_attractions(days_pois, max_per_day=3)

            # A2 + A3 并行: Hotel 与 Weather 互不依赖,并行可省 5-10s
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_hotel = pool.submit(self.hotel_agent.run, days_pois, request)
                fut_weather = pool.submit(self.weather_agent.run, request)
                day_hotels = fut_hotel.result()
                weather_list = fut_weather.result()

            # A4: 文案生成。默认走确定性模板,可配置开启 LLM 润色且有超时保护。
            settings = get_settings()
            llm_data: Dict[str, Any] = {}
            if settings.enable_llm_planner:
                planner_query = self._build_planner_query(
                    request, days_pois, day_hotels, weather_list, routing,
                )
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                executor = ThreadPoolExecutor(max_workers=1)
                fut = executor.submit(self.planner_agent.run, planner_query)
                try:
                    planner_response = fut.result(
                        timeout=max(1, settings.llm_planner_timeout_seconds)
                    )
                    logger.debug("文案生成结果片段: %s", planner_response[:200])
                    llm_data = self._extract_json_from_llm(planner_response) or {}
                except TimeoutError:
                    logger.warning(
                        "LLM 文案生成超过 %ds,改用确定性模板",
                        settings.llm_planner_timeout_seconds,
                    )
                except Exception as exc:
                    logger.warning("LLM 文案生成失败,改用确定性模板: %s", exc)
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

            if not llm_data:
                llm_data = self._generate_plan_copy(
                    request, days_pois, day_hotels, weather_list, routing,
                )

            # 拼装 TripPlan
            trip_plan = self._assemble_plan(
                request, days_pois, day_hotels, weather_list, llm_data
            )

            # 输出 Guardrail (验证 LLM 没引入未授权景点)
            allowed_ids = {p.id for p in candidates if p.id}
            output_check = self.guardrail_agent.run("output", (trip_plan, allowed_ids))
            if output_check.violations:
                logger.warning("Output 护栏发现 %d 处问题(已记录,未阻断)",
                               len(output_check.violations))

            # ──────── LEARN 阶段: 评估 + 反馈 ────────
            logger.info("🟣 [Learn] 自动质量评估...")
            eval_report = self.evaluation_agent.run(request, trip_plan)
            # 评估元数据回写 overall_suggestions 末尾(不影响主体)
            if eval_report.warnings:
                trip_plan.overall_suggestions = (
                    trip_plan.overall_suggestions or ""
                ) + f"\n\n[质量评估 {eval_report.grade} ({eval_report.overall_score:.0f}/100)]"

            logger.info(
                "═══ ✅ 旅行计划生成完成 city=%s days=%d 评级=%s ═══",
                request.city, request.travel_days, eval_report.grade,
            )
            return trip_plan

        except Exception:
            logger.exception("生成旅行计划失败")
            return self._create_fallback_plan(request)

    # ============ 子流程 ============

    @staticmethod
    def _compose_themed_days(
        candidates: List[POIInfo],
        request: TripRequest,
    ) -> Optional[List[List[POIInfo]]]:
        """按真实旅行主题组织每天景点。

        聚类适合控制距离,但纯聚类会把旅行体验变成“同类景点堆叠”。这里对热门城市
        先按常见旅游日主题选点,再让后续硬规则控制远郊/容量。
        """
        if request.city not in ("北京", "上海") or not candidates:
            return None

        free_text = request.free_text_input or ""
        prefs = set(request.preferences or [])
        relaxed = any(kw in free_text for kw in ("不太累", "不要太累", "轻松", "慢游"))
        if request.city == "上海":
            return MultiAgentTripPlanner._compose_shanghai_themed_days(
                candidates, request, prefs, free_text, relaxed
            )

        want_wall = any(kw in free_text for kw in ("长城", "八达岭", "慕田峪", "爬长城"))
        want_food = "美食" in prefs or "美食" in free_text or "好吃" in free_text
        want_night = "夜景" in prefs or "夜景" in free_text or "晚上" in free_text

        themes: List[tuple] = []
        if "升旗" in free_text:
            themes.append(("中轴线升旗与前门老城", ["天安门广场", "中国国家博物馆", "前门大街", "王府井"]))
        else:
            themes.append(("天安门与老城中轴线", ["天安门广场", "中国国家博物馆", "前门大街"]))

        themes.append(("故宫深度与登高看城", ["故宫博物院", "景山公园", "北海公园"]))
        themes.append(("皇家园林慢游", ["颐和园", "圆明园", "恭王府"]))
        themes.append(("胡同寺庙与市井体验", ["什刹海", "南锣鼓巷", "雍和宫", "恭王府"]))
        if want_wall or request.travel_days >= 3:
            themes.append(("长城远郊半日/一日", ["八达岭长城", "慕田峪长城"]))
        if want_food or want_night or request.travel_days >= 4:
            themes.append(("街区美食与夜游", ["前门大街", "王府井步行街", "南锣鼓巷", "什刹海"]))
        if request.travel_days >= 5:
            themes.append(("现代北京与艺术区", ["798艺术区", "奥林匹克公园", "王府井步行街"]))

        if want_wall:
            # 长城不要被挤掉。3 天以内时压缩皇家园林/胡同,保留长城日。
            wall_theme = next((t for t in themes if "长城" in t[0]), None)
            non_wall = [t for t in themes if "长城" not in t[0]]
            if wall_theme:
                themes = non_wall[:max(0, request.travel_days - 1)] + [wall_theme]

        themes = themes[:request.travel_days]
        if len(themes) < request.travel_days:
            themes.extend([("城市弹性慢游", ["北海公园", "景山公园", "王府井步行街"])] * (request.travel_days - len(themes)))

        def find_poi(keyword: str, used: set) -> Optional[POIInfo]:
            for poi in candidates:
                if poi.id in used:
                    continue
                if keyword in poi.name or poi.name in keyword:
                    return poi
            for poi in candidates:
                if poi.id in used:
                    continue
                if any(part and part in poi.name for part in keyword.split()):
                    return poi
            return None

        used_ids: set = set()
        days: List[List[POIInfo]] = []
        max_per_day = 2 if relaxed and request.travel_days >= 4 else 3
        for theme_name, keywords in themes:
            day: List[POIInfo] = []
            limit = 1 if "长城" in theme_name else max_per_day
            for kw in keywords:
                poi = find_poi(kw, used_ids)
                if poi is None:
                    continue
                day.append(poi)
                used_ids.add(poi.id)
                if len(day) >= limit:
                    break
            if not day:
                for poi in candidates:
                    if poi.id not in used_ids:
                        day.append(poi)
                        used_ids.add(poi.id)
                        break
            days.append(day)

        # 如果核心必去没被主题命中,补进较短的非远郊日。
        must_keywords = ["天安门广场", "故宫博物院", "颐和园"]
        if want_wall:
            must_keywords.append("八达岭长城")
        for kw in must_keywords:
            if any(any(kw in p.name or p.name in kw for p in day) for day in days):
                continue
            poi = find_poi(kw, used_ids)
            if poi is None:
                continue
            target_idx = min(
                range(len(days)),
                key=lambda idx: (999 if any("长城" in p.name for p in days[idx]) else len(days[idx])),
            )
            if len(days[target_idx]) < 3:
                days[target_idx].append(poi)
                used_ids.add(poi.id)

        logger.info(
            "主题化行程: %s",
            " | ".join(
                f"Day{i + 1} {themes[i][0]}: " + "/".join(p.name for p in day)
                for i, day in enumerate(days)
            ),
        )
        return days

    @staticmethod
    def _compose_shanghai_themed_days(
        candidates: List[POIInfo],
        request: TripRequest,
        prefs: set,
        free_text: str,
        relaxed: bool,
    ) -> Optional[List[List[POIInfo]]]:
        want_disney = "迪士尼" in free_text or "亲子" in prefs
        want_art = "艺术" in prefs or "美术馆" in free_text or "展览" in free_text
        want_food = "美食" in prefs or "美食" in free_text or "好吃" in free_text
        want_night = "夜景" in prefs or "夜景" in free_text or "晚上" in free_text
        want_old_town = any(kw in free_text for kw in ("古镇", "朱家角", "水乡"))

        themes: List[tuple] = [
            ("外滩陆家嘴夜景", ["外滩", "外白渡桥", "东方明珠", "上海中心大厦", "陆家嘴中心绿地"]),
            ("人民广场博物馆与南京路", ["上海博物馆", "人民广场", "南京路步行街"]),
            ("豫园城隍庙与老城厢", ["豫园", "城隍庙旅游区", "上海新天地"]),
            ("衡复街区城市漫步", ["武康大楼", "思南公馆", "愚园路历史风貌区", "田子坊"]),
        ]
        if want_art or request.travel_days >= 4:
            themes.append(("滨江西岸与当代艺术", ["西岸美术馆", "浦东美术馆", "上海中心大厦"]))
        if want_disney:
            themes.append(("迪士尼一日", ["上海迪士尼乐园"]))
        elif want_old_town or request.travel_days >= 5:
            themes.append(("朱家角水乡半日", ["朱家角古镇"]))
        if want_food or want_night:
            themes.append(("本帮菜与夜生活", ["新天地", "田子坊", "南京路步行街", "外滩"]))

        # 不强制每个地标都出现，按天数取最匹配的主题。迪士尼/古镇只在用户意图或天数足够时加入。
        themes = themes[:request.travel_days]
        if len(themes) < request.travel_days:
            themes.extend([("弹性街区慢游", ["愚园路历史风貌区", "思南公馆", "田子坊"])] * (request.travel_days - len(themes)))

        def find_poi(keyword: str, used: set) -> Optional[POIInfo]:
            for poi in candidates:
                if poi.id in used:
                    continue
                if keyword in poi.name or poi.name in keyword:
                    return poi
            return None

        used_ids: set = set()
        days: List[List[POIInfo]] = []
        max_per_day = 2 if relaxed and request.travel_days >= 4 else 3
        for theme_name, keywords in themes:
            day: List[POIInfo] = []
            limit = 1 if "迪士尼" in theme_name else max_per_day
            if "朱家角" in theme_name:
                limit = 1 if relaxed else 2
            for kw in keywords:
                poi = find_poi(kw, used_ids)
                if poi is None:
                    continue
                day.append(poi)
                used_ids.add(poi.id)
                if len(day) >= limit:
                    break
            if not day:
                for poi in candidates:
                    if poi.id not in used_ids:
                        day.append(poi)
                        used_ids.add(poi.id)
                        break
            days.append(day)

        logger.info(
            "上海主题化行程: %s",
            " | ".join(
                f"Day{i + 1} {themes[i][0]}: " + "/".join(p.name for p in day)
                for i, day in enumerate(days)
            ),
        )
        return days

    def _generate_plan_copy(
        self,
        request: TripRequest,
        days_pois: List[List[POIInfo]],
        day_hotels: List[Optional[Hotel]],
        weather_list: List[WeatherInfo],
        routing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """确定性行程文案生成。

        这个方法不依赖 LLM,用于主链路提速和兜底。它只基于已选真实 POI/酒店/天气生成
        description、meals、ticket_price、visit_duration、overall_suggestions。
        """
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        days: List[Dict[str, Any]] = []
        used_food_ids: set = set()
        for i, day_pois in enumerate(days_pois):
            hotel = day_hotels[i] if i < len(day_hotels) else None
            date_str = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            days.append({
                "date": date_str,
                "day_index": i,
                "description": self._build_day_description(i, day_pois, hotel, request),
                "transportation": self._transportation_hint(day_pois, request),
                "accommodation": request.accommodation,
                "meals": self._specific_meals(day_pois, hotel, request.city, used_food_ids),
                "attractions": [
                    {
                        "poi_id": p.id,
                        "ticket_price": self._estimate_ticket_price(p),
                        "visit_duration": self._estimate_visit_duration(p),
                        "description": self._attraction_blurb(p),
                    }
                    for p in day_pois
                ],
            })

        return {
            "city": request.city,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "days": days,
            "weather_info": [
                {
                    "date": w.date,
                    "day_weather": w.day_weather,
                    "night_weather": w.night_weather,
                    "day_temp": w.day_temp,
                    "night_temp": w.night_temp,
                    "wind_direction": w.wind_direction,
                    "wind_power": w.wind_power,
                }
                for w in weather_list
            ],
            "overall_suggestions": self._build_overall_suggestions(
                request, days_pois, weather_list, routing
            ),
        }

    def _build_day_description(
        self,
        day_index: int,
        day_pois: List[POIInfo],
        hotel: Optional[Hotel],
        request: TripRequest,
    ) -> str:
        if not day_pois:
            return f"第{day_index + 1}天建议放慢节奏,根据天气和体力安排城市休整。"

        parts: List[str] = []
        theme = self._infer_day_theme(day_pois)
        if theme:
            parts.append(f"第{day_index + 1}天主线: {theme},同片区顺路游览,避免来回折返。")
        cursor_hour = 8
        cursor_min = 30
        free_text = request.free_text_input or ""
        if day_index == 0 and "升旗" in free_text and "天安门" in day_pois[0].name:
            parts.append(
                f"清晨提前抵达{day_pois[0].name}观看升旗仪式,旺季建议至少提前60-90分钟到达安检口。"
            )
            cursor_hour, cursor_min = 8, 30

        for idx, poi in enumerate(day_pois):
            if idx > 0:
                prev = day_pois[idx - 1]
                commute = self._commute_text(prev, poi, request.transportation)
                parts.append(f"{commute}前往{poi.name}。")
            time_label = f"{cursor_hour:02d}:{cursor_min:02d}"
            duration = self._estimate_visit_duration(poi)
            ticket = self._estimate_ticket_price(poi)
            ticket_text = "免票或按官方公示" if ticket == 0 else f"门票约¥{ticket}"
            parts.append(
                f"{time_label}游览{poi.name},建议停留{duration // 60 if duration >= 60 else 1}"
                f"{'小时' if duration >= 60 else '小时以内'}, {ticket_text}。"
            )
            cursor_hour += max(1, round(duration / 60)) + (1 if idx < len(day_pois) - 1 else 0)
            if cursor_hour >= 18:
                cursor_hour = 17
                cursor_min = 0

        if hotel is not None:
            parts.append(f"傍晚回到{hotel.name},酒店参考价{hotel.price_range},适合按当天动线就近入住。")
        else:
            parts.append("傍晚建议回到市中心或交通便利区域入住,方便第二天出发。")
        return "".join(parts)

    @staticmethod
    def _infer_day_theme(day_pois: List[POIInfo]) -> str:
        text = " ".join(p.name for p in day_pois)
        if "长城" in text or "八达岭" in text or "慕田峪" in text:
            return "远郊长城"
        if "天安门" in text or "国家博物馆" in text or "前门" in text:
            return "中轴线老城"
        if "故宫" in text or "景山" in text or "北海" in text:
            return "宫城登高"
        if "颐和园" in text or "圆明园" in text:
            return "园林慢游"
        if "什刹海" in text or "南锣鼓巷" in text or "恭王府" in text or "雍和宫" in text:
            return "胡同市井"
        if "798" in text or "奥林匹克" in text or "王府井" in text:
            return "现代艺术夜游"
        if "外滩" in text or "东方明珠" in text or "上海中心" in text or "陆家嘴" in text:
            return "浦江夜景"
        if "上海博物馆" in text or "人民广场" in text or "南京路" in text:
            return "人民广场南京路"
        if "豫园" in text or "城隍庙" in text:
            return "老城厢小吃"
        if "武康" in text or "思南" in text or "愚园" in text or "田子坊" in text:
            return "衡复街区漫步"
        if "西岸" in text or "浦东美术馆" in text:
            return "滨江艺术"
        if "朱家角" in text:
            return "水乡古镇"
        if "迪士尼" in text:
            return "迪士尼一日"
        return "城市顺路慢游"

    @staticmethod
    def _commute_text(prev: POIInfo, nxt: POIInfo, mode: str) -> str:
        from ..services.itinerary_optimizer import estimate_commute
        commute = estimate_commute(prev.location, nxt.location, mode="driving")
        dist = commute["distance_km"]
        minutes = max(10, int(commute["duration_min"]))
        if dist <= 1.2:
            return f"步行约{int(dist * 1000)}米"
        if "公共" in (mode or "") or "公交" in (mode or "") or "地铁" in (mode or ""):
            return f"公共交通/打车约{minutes}-{minutes + 10}分钟"
        return f"车程约{minutes}分钟"

    @staticmethod
    def _transportation_hint(day_pois: List[POIInfo], request: TripRequest) -> str:
        if len(day_pois) <= 1:
            return request.transportation or "市内交通"
        from ..services.itinerary_optimizer import haversine
        max_dist = max(
            haversine(day_pois[i].location, day_pois[i + 1].location)
            for i in range(len(day_pois) - 1)
        )
        if max_dist <= 1.2:
            return "步行串联为主"
        if max_dist <= 8:
            return "地铁/打车结合"
        return "建议打车或包车衔接,预留跨区通勤时间"

    @staticmethod
    def _specific_meals(
        day_pois: List[POIInfo],
        hotel: Optional[Hotel],
        city: str,
        used_food_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        foods = get_curated_food(city)
        if not foods:
            return MultiAgentTripPlanner._default_meals(day_pois, hotel, city)

        from ..services.itinerary_optimizer import haversine

        used_ids: set = set()
        global_used = used_food_ids if used_food_ids is not None else set()
        context_tags = MultiAgentTripPlanner._food_context_tags(day_pois)

        def anchor_for(meal_type: str) -> Optional[Location]:
            if meal_type == "breakfast":
                if hotel and hotel.location:
                    return hotel.location
                return day_pois[0].location if day_pois else None
            if meal_type == "lunch":
                if day_pois:
                    # 旅行里午餐通常跟上午第一个核心景点绑定；两景点日尤其如此。
                    idx = 0 if len(day_pois) <= 2 else 1
                    return day_pois[idx].location
                return hotel.location if hotel and hotel.location else None
            if day_pois:
                return day_pois[-1].location
            return hotel.location if hotel and hotel.location else None

        def pick(meal_type: str) -> Optional[POIInfo]:
            anchor = anchor_for(meal_type)
            compatible = [
                f for f in foods
                if f.id not in used_ids
                and meal_type in getattr(f, "_meal_types", [])
            ]
            if not compatible:
                compatible = [f for f in foods if f.id not in used_ids and f.id not in global_used]
            if not compatible:
                compatible = [f for f in foods if f.id not in used_ids]
            if not compatible:
                return None
            def score(food: POIInfo) -> float:
                areas = set(getattr(food, "_areas", []))
                tag_match = len(areas & context_tags)
                d = haversine(anchor, food.location) if anchor and food.location else 8.0
                brand_key = MultiAgentTripPlanner._food_brand_key(food.name)
                s = d
                s -= tag_match * 8.0
                if food.id in global_used:
                    s += 8.0
                if brand_key in global_used:
                    s += 10.0
                if meal_type == "breakfast" and "quick" in areas:
                    s -= 2.0
                if meal_type == "dinner" and ("night" in areas or "hotpot" in areas):
                    s -= 2.0
                return s

            chosen = min(compatible, key=score)
            used_ids.add(chosen.id)
            global_used.add(chosen.id)
            global_used.add(MultiAgentTripPlanner._food_brand_key(chosen.name))
            return chosen

        meals: List[Dict[str, Any]] = []
        labels = {
            "breakfast": "早餐",
            "lunch": "午餐",
            "dinner": "晚餐",
        }
        for meal_type in ("breakfast", "lunch", "dinner"):
            food = pick(meal_type)
            if food is None:
                continue
            anchor = anchor_for(meal_type)
            distance_text = ""
            if anchor and food.location:
                dist = haversine(anchor, food.location)
                if dist <= 15:
                    distance_text = f"距当段行程锚点约{dist:.1f}公里, "
                elif meal_type == "dinner":
                    distance_text = "建议返城后或回酒店前用餐, "
            signature = getattr(food, "_signature", "招牌菜")
            note = getattr(food, "_note", "当地人气餐厅")
            cost = MultiAgentTripPlanner._safe_int(food.cost, default=80)
            meals.append({
                "type": meal_type,
                "name": food.name,
                "address": food.address,
                "location": {
                    "longitude": food.location.longitude,
                    "latitude": food.location.latitude,
                } if food.location else None,
                "description": (
                    f"{labels[meal_type]}推荐: {signature}; {distance_text}"
                    f"{note} 地址: {food.address}"
                ),
                "estimated_cost": cost,
            })

        if len(meals) < 3:
            fallback_by_type = {m["type"]: m for m in MultiAgentTripPlanner._default_meals(day_pois, hotel, city)}
            existing = {m["type"] for m in meals}
            for meal_type in ("breakfast", "lunch", "dinner"):
                if meal_type not in existing:
                    meals.append(fallback_by_type[meal_type])
        return meals

    @staticmethod
    def _food_brand_key(name: str) -> str:
        if not name:
            return ""
        base = name.split("(")[0].split("（")[0]
        for suffix in ("总店", "本店", "分店", "店"):
            base = base.replace(suffix, "")
        return base.strip()

    @staticmethod
    def _food_context_tags(day_pois: List[POIInfo]) -> set:
        text = " ".join(p.name for p in day_pois)
        tags = set()
        if any(kw in text for kw in ("天安门", "国家博物馆", "前门", "王府井")):
            tags.update(["central", "qianmen", "wangfujing"])
        if any(kw in text for kw in ("故宫", "景山", "北海")):
            tags.update(["palace", "central", "duck"])
        if any(kw in text for kw in ("颐和园", "圆明园")):
            tags.update(["garden", "west"])
        if any(kw in text for kw in ("什刹海", "南锣鼓巷", "恭王府", "雍和宫")):
            tags.update(["hutong", "local"])
        if any(kw in text for kw in ("长城", "八达岭", "慕田峪")):
            tags.update(["wall", "badaling", "quick"])
        if any(kw in text for kw in ("798", "奥林匹克", "鸟巢", "水立方")):
            tags.update(["modern", "art", "olympic", "night"])
        if any(kw in text for kw in ("外滩", "外白渡桥")):
            tags.update(["bund", "night", "view"])
        if any(kw in text for kw in ("东方明珠", "上海中心", "陆家嘴", "浦东美术馆")):
            tags.update(["pudong", "luiiazui", "night", "modern"])
        if any(kw in text for kw in ("南京路", "人民广场", "上海博物馆")):
            tags.update(["nanjing", "people", "central", "snack"])
        if any(kw in text for kw in ("豫园", "城隍庙")):
            tags.update(["yuyuan", "oldtown", "snack"])
        if any(kw in text for kw in ("武康", "思南", "愚园", "田子坊", "新天地")):
            tags.update(["fuxing", "wukang", "xintiandi", "tianzifang", "benbang"])
        if any(kw in text for kw in ("西岸", "美术馆")):
            tags.update(["westbund", "art", "modern"])
        if any(kw in text for kw in ("朱家角", "古镇")):
            tags.update(["zhujiajiao", "oldtown", "local"])
        if any(kw in text for kw in ("迪士尼",)):
            tags.update(["disney", "quick"])
        return tags

    @staticmethod
    def _default_meals(day_pois: List[POIInfo], hotel: Optional[Hotel], city: str) -> List[Dict[str, Any]]:
        anchor = day_pois[0].name if day_pois else (hotel.name if hotel else city)
        dinner_anchor = day_pois[-1].name if day_pois else anchor
        return [
            {
                "type": "breakfast",
                "name": f"{city}本地早餐店",
                "description": "优先选择酒店附近评分较高的早餐店,点当地特色主食,避免上午景点排队前空腹赶路。",
                "estimated_cost": 30,
            },
            {
                "type": "lunch",
                "name": f"{anchor}附近评分餐厅",
                "description": "建议在地图上按评分和排队情况二次确认,优先选景区外步行10-15分钟范围内餐厅。",
                "estimated_cost": 70,
            },
            {
                "type": "dinner",
                "name": f"{dinner_anchor}附近本地菜馆",
                "description": f"晚餐安排在{dinner_anchor}或酒店周边,选择当地菜或老字号,方便结束行程后返程休息。",
                "estimated_cost": 110,
            },
        ]

    @staticmethod
    def _estimate_visit_duration(poi: POIInfo) -> int:
        name = poi.name or ""
        type_text = f"{poi.type or ''}{poi.biz_type or ''}"
        if any(kw in name for kw in ("迪士尼", "长城", "黄山", "兵马俑")):
            return 240
        if any(kw in name for kw in ("故宫", "颐和园", "西湖", "博物馆", "美术馆", "动物园")):
            return 180
        if any(kw in type_text for kw in ("博物馆", "风景名胜")):
            return 120
        if any(kw in name for kw in ("步行街", "巷", "街", "外滩", "广场")):
            return 90
        return 120

    @staticmethod
    def _estimate_ticket_price(poi: POIInfo) -> int:
        name = poi.name or ""
        known = [
            (("故宫",), 60),
            (("颐和园",), 30),
            (("天坛",), 34),
            (("长城", "八达岭", "慕田峪"), 40),
            (("东方明珠",), 199),
            (("迪士尼",), 475),
            (("豫园",), 40),
            (("兵马俑",), 120),
            (("西安城墙",), 54),
            (("大雁塔",), 40),
            (("雷峰塔",), 40),
            (("灵隐寺",), 75),
            (("熊猫",), 55),
            (("武侯祠", "杜甫草堂"), 50),
        ]
        for kws, price in known:
            if any(kw in name for kw in kws):
                return price
        if any(kw in name for kw in ("博物馆", "美术馆", "广场", "外滩", "步行街", "新天地")):
            return 0
        return 0

    @staticmethod
    def _attraction_blurb(poi: POIInfo) -> str:
        name = poi.name or "该景点"
        if poi.level in ("AAAAA", "5A"):
            return f"{name}是当地高知名度地标,建议提前预约并错峰入场。"
        if "博物馆" in name or "馆" in (poi.biz_type or ""):
            return f"{name}适合安排较完整的参观时间,重点看常设展和代表性展品。"
        if any(kw in name for kw in ("街", "巷", "路", "外滩", "广场")):
            return f"{name}适合步行游览和拍照,傍晚到夜间体验更完整。"
        return f"{name}适合按当天动线顺路游览,建议关注开放时间和预约要求。"

    def _build_overall_suggestions(
        self,
        request: TripRequest,
        days_pois: List[List[POIInfo]],
        weather_list: List[WeatherInfo],
        routing: Optional[Dict[str, Any]] = None,
    ) -> str:
        names = "、".join(p.name for day in days_pois for p in day)
        tips = [
            f"本次行程按{request.city}热门地标和地理动线组合,核心景点包括{names}。",
            "热门景点建议提前1-3天预约,每天上午优先安排预约制或排队压力大的景点。",
            "市内跨区移动优先地铁加短途打车,远郊景点单独留半天以上,不要和市中心密集景点硬拼。"
        ]
        if weather_list:
            w = weather_list[0]
            tips.append(f"首日天气参考: 白天{w.day_weather}, {w.day_temp}°C, 出门前再确认实时预报。")
        free_text = request.free_text_input or ""
        if "升旗" in free_text:
            if "天安门" in names:
                tips.append("升旗仪式已安排在第1天清晨,具体升旗时间随日出变化,出发前请查当天官方时间。")
            else:
                tips.append("你提到想看升旗,但本次实际景点未包含天安门广场,建议另行预留清晨时段。")
        if routing and routing.get("intent"):
            tips.append(f"整体节奏按“{routing['intent']}”处理,保留午餐和跨点通勤缓冲。")
        return "\n".join(tips)

    # ============ planner 输入/输出 ============

    def _build_planner_query(
        self,
        request: TripRequest,
        days_pois: List[List[POIInfo]],
        day_hotels: List[Optional[Hotel]],
        weather_list: List[WeatherInfo],
        routing: Optional[Dict[str, Any]] = None,
    ) -> str:
        """给 planner_agent 喂结构化数据，让它仅做文案。"""
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        days_payload: List[Dict[str, Any]] = []
        for i, day_pois in enumerate(days_pois):
            date_str = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            attractions_payload = [
                {
                    "poi_id": p.id,
                    "name": p.name,
                    "address": p.address,
                    "location": {"longitude": p.location.longitude, "latitude": p.location.latitude},
                    "rating": p.rating,
                    "category": p.biz_type or "景点",
                }
                for p in day_pois
            ]
            hotel = day_hotels[i] if i < len(day_hotels) else None
            hotel_payload = None
            if hotel is not None:
                hotel_payload = {
                    "name": hotel.name,
                    "address": hotel.address,
                    "type": hotel.type,
                    "price_range": hotel.price_range,
                    "estimated_cost": hotel.estimated_cost,
                }
            days_payload.append({
                "date": date_str,
                "day_index": i,
                "attractions": attractions_payload,
                "hotel": hotel_payload,
            })

        weather_payload = [
            {
                "date": w.date,
                "day_weather": w.day_weather,
                "night_weather": w.night_weather,
                "day_temp": w.day_temp,
                "night_temp": w.night_temp,
                "wind_direction": w.wind_direction,
                "wind_power": w.wind_power,
            }
            for w in weather_list
        ]

        payload = {
            "city": request.city,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "transportation_pref": request.transportation,
            "accommodation_pref": request.accommodation,
            "preferences": request.preferences,
            "free_text_input": request.free_text_input or "",
            "days": days_payload,
            "weather_info": weather_payload,
        }

        # 意图识别结果 (Reason 阶段产物)
        intent_hint = ""
        if routing:
            intent_hint = (
                f"\n\n🧠 **意图识别 (Orchestrator)**: {routing.get('intent', '通用观光')}\n"
                f"请把这个意图作为文案风格的指导原则。"
            )

        # 用户特殊诉求
        free_text_hint = ""
        if request.free_text_input:
            free_text_hint = (
                f"\n\n⚠️ **用户特殊诉求**: \"{request.free_text_input}\"\n"
                f"必须在 description 或 overall_suggestions 中**显式回应这个诉求**,"
                f"包括具体时间、地点提示等可执行细节,不要忽略不要敷衍。"
            )

        # Agentic RAG: 检索旅行知识库
        rag_hint = ""
        try:
            rag = get_rag()
            query = " ".join([
                request.city or "",
                request.free_text_input or "",
                " ".join(p.name for day in days_pois for p in day[:2]),  # 用前几个景点名作辅助检索词
            ])
            kb_docs = rag.search(query, city=request.city, top_k=3)
            if kb_docs:
                rag_context = rag.format_context(kb_docs, max_chars=1200)
                rag_hint = (
                    f"\n\n📚 **旅行知识库检索结果 (Agentic RAG)** — 请把以下知识融入 description 和 "
                    f"overall_suggestions,提供可执行的具体细节(时间/价格/避坑等):\n{rag_context}"
                )
        except Exception as exc:
            logger.warning("RAG 检索失败,跳过: %s", exc)

        # ⚠️ 显式列出每天景点 — 防止 LLM 脱离数据自由发挥
        per_day_summary = "\n".join(
            f"  Day {i+1} ({d['date']}): " + (
                " / ".join(a["name"] for a in d["attractions"])
                if d["attractions"] else "(无景点)"
            )
            for i, d in enumerate(days_payload)
        )

        # 检查 free_text 是否在任何天的景点里能落地
        free_text_uncovered_hint = ""
        if request.free_text_input:
            all_names = " ".join(
                a["name"] for d in days_payload for a in d["attractions"]
            )
            # 简单关键词检测: 用户 free_text 里的名词是否在景点名出现
            import re as _re
            user_kws = set(_re.findall(r"[一-龥]{2,5}(?:门|宫|城|塔|寺|园|湖|山|场)", request.free_text_input))
            uncovered = [kw for kw in user_kws if kw not in all_names]
            if uncovered:
                free_text_uncovered_hint = (
                    f"\n\n⚠️ **重要提示**: 用户 free_text 提到的 {uncovered} "
                    f"在本次行程的实际景点列表中**没有**安排。description 不要包含这些地点,"
                    f"如需对用户解释,可在 overall_suggestions 里说一句"
                    f"\"本次行程未安排 X,可作为下次旅行考虑\"。"
                )

        return (
            "请为下面这份已经规划好的行程生成文案。\n\n"
            "🔒 **不可越权修改的字段**: poi_id、坐标、酒店名、价格 — 这些都是后端算法已确定的真实数据。\n"
            "📝 **你只输出**: description / meals / ticket_price / visit_duration / "
            "overall_suggestions 这些文案与数值字段。\n\n"
            f"📍 **本次行程的实际景点列表 (description 必须严格围绕这些景点写,不要引入其他地方)**:\n"
            f"{per_day_summary}"
            f"{intent_hint}{free_text_hint}{rag_hint}{free_text_uncovered_hint}\n\n"
            f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
        )

    def _extract_json_from_llm(self, response: str) -> Optional[Dict[str, Any]]:
        """从 LLM 响应里抽 JSON。"""
        if not response:
            return None
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                return json.loads(response[start:end].strip())
            if "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                return json.loads(response[start:end].strip())
            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                return json.loads(response[start:end])
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("LLM 响应 JSON 解析失败: %s", exc)
        return None

    # ============ 拼装最终 TripPlan ============

    def _assemble_plan(
        self,
        request: TripRequest,
        days_pois: List[List[POIInfo]],
        day_hotels: List[Optional[Hotel]],
        weather_list: List[WeatherInfo],
        llm_data: Dict[str, Any],
    ) -> TripPlan:
        """把"已确定的景点/酒店"+"LLM 文案"合并为最终 TripPlan。"""
        llm_days = llm_data.get("days") or []
        # 按 day_index 建索引
        llm_day_by_index: Dict[int, Dict[str, Any]] = {}
        for d in llm_days:
            if isinstance(d, dict) and isinstance(d.get("day_index"), int):
                llm_day_by_index[d["day_index"]] = d

        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        days: List[DayPlan] = []
        budget_attractions = 0
        budget_meals = 0
        budget_hotels = 0

        for i, day_pois in enumerate(days_pois):
            date_str = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            llm_day = llm_day_by_index.get(i, {})

            # 景点：从候选透传，仅注入 LLM 给的 ticket_price/visit_duration/description
            llm_attr_by_id: Dict[str, Dict[str, Any]] = {}
            for la in llm_day.get("attractions") or []:
                if isinstance(la, dict) and la.get("poi_id"):
                    llm_attr_by_id[la["poi_id"]] = la

            attractions: List[Attraction] = []
            for poi in day_pois:
                la = llm_attr_by_id.get(poi.id, {})
                ticket = self._safe_int(la.get("ticket_price"), default=0)
                duration = self._safe_int(la.get("visit_duration"), default=120)
                desc = la.get("description") or f"{request.city}的著名景点 {poi.name}"
                image_url = self._resolve_image(poi, request.city)
                attractions.append(Attraction(
                    name=poi.name,
                    address=poi.address,
                    location=poi.location,
                    visit_duration=duration,
                    description=desc,
                    category=poi.biz_type or "景点",
                    rating=poi.rating,
                    photos=poi.photos,
                    poi_id=poi.id,
                    image_url=image_url,
                    ticket_price=ticket,
                ))
                budget_attractions += ticket

            # 餐饮：直接用 LLM 输出（如果没给则给个保底）
            meals_payload = llm_day.get("meals") or []
            meals: List[Meal] = []
            for m in meals_payload:
                if not isinstance(m, dict):
                    continue
                cost = self._safe_int(m.get("estimated_cost"), default=0)
                meals.append(Meal(
                    type=m.get("type") or "lunch",
                    name=m.get("name") or "用餐推荐",
                    address=m.get("address"),
                    location=self._parse_optional_location(m.get("location")),
                    description=m.get("description") or "",
                    estimated_cost=cost,
                ))
                budget_meals += cost
            if not meals:
                meals = [
                    Meal(type="breakfast", name="酒店早餐", description="", estimated_cost=30),
                    Meal(type="lunch", name="景区周边午餐", description="", estimated_cost=60),
                    Meal(type="dinner", name="本地特色晚餐", description="", estimated_cost=100),
                ]
                budget_meals += 190

            # 酒店：从后端透传（最后一天可省）
            hotel = day_hotels[i] if i < len(day_hotels) else None
            if hotel and hotel.estimated_cost:
                budget_hotels += hotel.estimated_cost

            description = (
                llm_day.get("description")
                or f"第{i+1}天：游览 " + "、".join(p.name for p in day_pois)
            )
            transportation = llm_day.get("transportation") or request.transportation

            days.append(DayPlan(
                date=date_str,
                day_index=i,
                description=description,
                transportation=transportation,
                accommodation=request.accommodation,
                hotel=hotel,
                attractions=attractions,
                meals=meals,
            ))

        overall = (
            llm_data.get("overall_suggestions")
            or f"祝您在{request.city}度过愉快的{request.travel_days}天行程！"
        )

        # 简单预算：交通先按 100/天估
        budget_transport = 100 * request.travel_days
        from ..models.schemas import Budget
        budget = Budget(
            total_attractions=budget_attractions,
            total_hotels=budget_hotels,
            total_meals=budget_meals,
            total_transportation=budget_transport,
            total=budget_attractions + budget_hotels + budget_meals + budget_transport,
        )

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=weather_list,
            overall_suggestions=overall,
            budget=budget,
        )

    # ============ 工具函数 ============

    @staticmethod
    def _resolve_image(poi: POIInfo, city: str) -> Optional[str]:
        """为景点解析图片 URL（拼装阶段使用,要求"零外网请求、零延迟"）。

        优先级: 高德 photos → 图片缓存 → None
        百度百科 / Unsplash 等外网调用**留给前端的 /poi/photo 接口异步兜底**,
        避免 _assemble_plan 里串行 N 次外网请求拖慢整体响应导致 LLM 超时。
        """
        # 1) 高德精简版偶尔会给 photos
        if poi.photos:
            return poi.photos[0]

        # 2) 查图片缓存（用户之前访问过的景点会有）
        try:
            cache = get_image_cache()
            cached = cache.get(poi.name, city)
            if cached and cached.get("url"):
                return cached["url"]
        except Exception:
            pass

        # 3) 留给前端 /poi/photo 接口异步兜底（包含百度百科 + Unsplash）
        return None

    @staticmethod
    def _apply_must_first_rules(
        days_pois: List[List[POIInfo]],
        request: TripRequest,
    ) -> List[List[POIInfo]]:
        """对用户的强诉求做硬规则后处理,把特定景点强制安排到第 1 天首位。

        当前规则:
        - free_text 含"升旗" → 天安门广场必须在第 1 天的第 1 个景点（清晨观礼）
        """
        if not days_pois:
            return days_pois
        free_text = request.free_text_input or ""

        rules: List[tuple] = [
            # (触发词, 目标景点关键词)
            (("升旗",), "天安门广场"),
        ]

        for triggers, target_kw in rules:
            if not any(t in free_text for t in triggers):
                continue
            # 已经在 day[0][0] 就跳过
            if days_pois[0] and target_kw in days_pois[0][0].name:
                continue
            # 找出该景点（可能在某天的某位置）
            found_poi: Optional[POIInfo] = None
            for day_idx, day in enumerate(days_pois):
                for i, p in enumerate(day):
                    if target_kw in p.name:
                        found_poi = day.pop(i)
                        logger.info(
                            "硬规则: 把 %s 从第%d天位置%d移到第1天首位 (触发=%s)",
                            p.name, day_idx + 1, i + 1, triggers,
                        )
                        break
                if found_poi is not None:
                    break
            if found_poi is not None:
                days_pois[0].insert(0, found_poi)
                # 升旗通常是清晨市中心活动,不要和长城等远郊景点硬塞同一天。
                from ..services.itinerary_optimizer import haversine
                if len(days_pois[0]) > 1:
                    far_idx = max(
                        range(1, len(days_pois[0])),
                        key=lambda idx: haversine(found_poi.location, days_pois[0][idx].location),
                    )
                    far_dist = haversine(found_poi.location, days_pois[0][far_idx].location)
                    if far_dist > 25:
                        best: Optional[tuple] = None
                        for di in range(1, len(days_pois)):
                            for pi, candidate in enumerate(days_pois[di]):
                                dist = haversine(found_poi.location, candidate.location)
                                if best is None or dist < best[0]:
                                    best = (dist, di, pi)
                        if best is not None and best[0] <= 12:
                            _, repl_di, repl_pi = best
                            days_pois[0][far_idx], days_pois[repl_di][repl_pi] = (
                                days_pois[repl_di][repl_pi],
                                days_pois[0][far_idx],
                            )
                            logger.info(
                                "硬规则: 升旗当天换入 %s,把远郊 %s 调到 Day%d",
                                days_pois[0][far_idx].name,
                                days_pois[repl_di][repl_pi].name,
                                repl_di + 1,
                            )
                while len(days_pois[0]) > 3:
                    mover = days_pois[0].pop()
                    remote_keywords = ("长城", "八达岭", "慕田峪", "司马台", "金山岭")
                    candidates = [
                        idx for idx in range(1, len(days_pois))
                        if not any(
                            any(kw in p.name for kw in remote_keywords)
                            for p in days_pois[idx]
                        )
                    ] or list(range(1, len(days_pois)))
                    target_di = min(
                        candidates,
                        key=lambda idx: len(days_pois[idx]),
                    )
                    days_pois[target_di].append(mover)
                    logger.info(
                        "硬规则: 升旗首日控制在 3 个景点内,把 %s 调到 Day%d",
                        mover.name, target_di + 1,
                    )

        return days_pois

    @staticmethod
    def _spread_must_visit_across_days(
        days_pois: List[List[POIInfo]],
        request: TripRequest,
    ) -> List[List[POIInfo]]:
        """让 city_must_visit 命中的景点尽量分散到不同天 (不堆在一天)。

        例: 上海必去 = [外滩, 东方明珠, 豫园], 5 天行程。
        如果聚类把 3 个全堆到 day 0,后面 4 天全是冷门 → 给一种"前热后冷"的失衡感。
        本方法把这些"超级地标"打散,每天最多放 1-2 个,余下天更均衡。

        注意: 仅做轻量调换 (跨天 swap),不破坏 itinerary_optimizer 的 TSP 顺序。
        """
        from ..data.keywords import get_must_visit
        if not days_pois:
            return days_pois
        must_kws = get_must_visit(request.city)
        if not must_kws:
            return days_pois

        # 找出每天命中必去的景点索引 (day_idx, poi_idx, kw)
        hits_per_day: Dict[int, List[tuple]] = {}
        for di, day in enumerate(days_pois):
            for pi, p in enumerate(day):
                for kw in must_kws:
                    if kw in p.name:
                        hits_per_day.setdefault(di, []).append((pi, kw, p))
                        break

        if not hits_per_day:
            return days_pois

        # 找"必去过载"的天 (>1 个) 与 "必去为零"的天
        protect_day0 = "升旗" in (request.free_text_input or "")
        overloaded = [
            di for di, hits in hits_per_day.items()
            if len(hits) > 1 and not (protect_day0 and di == 0)
        ]
        empty = [di for di in range(len(days_pois)) if di not in hits_per_day]

        if not overloaded or not empty:
            return days_pois

        # 把过载天的多余必去景点,跟空白天的最后一个景点 swap (保留每天景点数不变)
        for over_di in overloaded:
            extras = hits_per_day[over_di][1:]  # 保留第 1 个,余下要打散
            for _poi_idx, _kw, must_poi in extras:
                if not empty:
                    break
                target_di = empty.pop(0)
                if not days_pois[target_di]:
                    continue
                # 拿出 must_poi
                # 注意: 因为之前可能已 pop 改变了 index, 重新查找
                src_idx = next(
                    (i for i, p in enumerate(days_pois[over_di]) if p.id == must_poi.id),
                    None,
                )
                if src_idx is None:
                    continue
                must_poi_actual = days_pois[over_di].pop(src_idx)
                # 把目标天最后一个景点(最不重要的) 移到过载天填空
                victim = days_pois[target_di].pop()
                days_pois[over_di].append(victim)
                days_pois[target_di].insert(0, must_poi_actual)
                logger.info(
                    "🔀 必去打散: %s 从 Day%d 调到 Day%d (换走 %s)",
                    must_poi_actual.name, over_di + 1, target_di + 1, victim.name,
                )

        return days_pois

    @staticmethod
    def _isolate_remote_attractions(
        days_pois: List[List[POIInfo]],
    ) -> List[List[POIInfo]]:
        """把长城等远郊大景点尽量单独成半日/一日,避免和市中心景点硬拼。"""
        if not days_pois:
            return days_pois
        remote_keywords = ("长城", "八达岭", "慕田峪", "司马台", "金山岭")
        for di, day in enumerate(days_pois):
            remote_idx = next(
                (idx for idx, p in enumerate(day) if any(kw in p.name for kw in remote_keywords)),
                None,
            )
            if remote_idx is None or len(day) <= 1:
                continue
            remote = day[remote_idx]
            movers = [p for idx, p in enumerate(day) if idx != remote_idx]
            days_pois[di] = [remote]
            for poi in movers:
                target_di = min(
                    (idx for idx in range(len(days_pois)) if idx != di),
                    key=lambda idx: len(days_pois[idx]),
                )
                days_pois[target_di].append(poi)
                logger.info(
                    "远郊规则: %s 单独安排,把 %s 调到 Day%d",
                    remote.name, poi.name, target_di + 1,
                )
        return days_pois

    @staticmethod
    def _cap_daily_attractions(
        days_pois: List[List[POIInfo]],
        max_per_day: int = 3,
    ) -> List[List[POIInfo]]:
        """控制每日景点数,避免为了塞满候选导致行程过载。"""
        if not days_pois:
            return days_pois
        remote_keywords = ("长城", "八达岭", "慕田峪", "司马台", "金山岭")

        def has_remote(day: List[POIInfo]) -> bool:
            return any(any(kw in p.name for kw in remote_keywords) for p in day)

        for di, day in enumerate(days_pois):
            while len(day) > max_per_day:
                mover = day.pop()
                targets = [
                    idx for idx, other in enumerate(days_pois)
                    if idx != di and len(other) < max_per_day and not has_remote(other)
                ]
                if targets:
                    target_di = min(targets, key=lambda idx: len(days_pois[idx]))
                    days_pois[target_di].append(mover)
                    logger.info(
                        "日程容量: 把 %s 从 Day%d 调到 Day%d",
                        mover.name, di + 1, target_di + 1,
                    )
                else:
                    logger.info(
                        "日程容量: Day%d 已超过 %d 个景点,舍弃低优先级候选 %s",
                        di + 1, max_per_day, mover.name,
                    )
        return days_pois

    @staticmethod
    def _safe_int(v: Any, default: int = 0) -> int:
        try:
            if v is None or v == "":
                return default
            return int(float(v))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _parse_optional_location(v: Any) -> Optional[Location]:
        if not isinstance(v, dict):
            return None
        try:
            lng = v.get("longitude")
            lat = v.get("latitude")
            if lng is None or lat is None:
                return None
            return Location(longitude=float(lng), latitude=float(lat))
        except (ValueError, TypeError):
            return None

    # ============ Fallback ============

    def _create_fallback_plan(
        self,
        request: TripRequest,
        reason: Optional[str] = None,
    ) -> TripPlan:
        """无候选数据 / 输入被护栏拦截时的备用计划。"""
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        days: List[DayPlan] = []
        for i in range(request.travel_days):
            current_date = start + timedelta(days=i)
            days.append(DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐"),
                ],
            ))

        if reason:
            suggestions = f"⚠️ {reason}"
        else:
            suggestions = (
                f"⚠️ 本次规划未能从地图服务获取到足够的景点候选，请稍后重试。"
                f"目标城市：{request.city}，计划天数：{request.travel_days}天。"
            )
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=suggestions,
        )


# ============ 单例 ============

_multi_agent_planner: Optional[MultiAgentTripPlanner] = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)。"""
    global _multi_agent_planner
    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()
    return _multi_agent_planner

"""高德地图MCP服务封装"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Union

from hello_agents.tools import MCPTool

from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo

logger = logging.getLogger(__name__)

# 全局MCP工具实例
_amap_mcp_tool = None


def get_amap_mcp_tool() -> MCPTool:
    """获取高德地图MCP工具实例(单例模式)"""
    global _amap_mcp_tool

    if _amap_mcp_tool is None:
        settings = get_settings()

        if not settings.amap_api_key:
            raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")

        _amap_mcp_tool = MCPTool(
            name="amap",
            description="高德地图服务,支持POI搜索、路线规划、天气查询等功能",
            server_command=["uvx", "amap-mcp-server"],
            env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
            auto_expand=True
        )

        tool_count = len(_amap_mcp_tool._available_tools)
        logger.info("高德地图MCP工具初始化成功，工具数量=%d", tool_count)
        if _amap_mcp_tool._available_tools:
            preview = [t.get("name", "unknown") for t in _amap_mcp_tool._available_tools[:5]]
            logger.info("可用工具预览: %s%s", preview, f" ...还有 {tool_count - 5} 个" if tool_count > 5 else "")

    return _amap_mcp_tool


# ============ 解析工具函数 ============

def _strip_mcp_envelope(raw: str) -> str:
    """剥离 HelloAgents MCPTool 包裹的 `工具 'xxx' 执行结果:\\n` 前缀。"""
    if not isinstance(raw, str):
        return raw
    # 匹配 "工具 'xxx' 执行结果:" 行（中英文冒号都兼容）
    m = re.match(r"^工具\s+'[^']*'\s+执行结果[:：]\s*", raw)
    if m:
        return raw[m.end():]
    return raw


def _extract_json(raw: str) -> Optional[Union[dict, list]]:
    """
    从可能含前缀文本/markdown 围栏的字符串里抽取首个 JSON 对象或数组。
    返回 dict / list / None。
    """
    if not raw:
        return None
    text = _strip_mcp_envelope(raw).strip()
    if not text:
        return None

    # 优先尝试整体直解
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass

    # 去 ```json ... ``` 围栏后再试
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except (ValueError, TypeError):
            pass

    # 用括号计数定位首个完整 JSON。按 `{` / `[` 在文本中首次出现的位置决定优先级，
    # 避免在 "[ {a} ]" 这种数组里被内部 dict 抢先匹配。
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        idx = text.find(opener)
        if idx != -1:
            candidates.append((idx, opener, closer))
    candidates.sort(key=lambda x: x[0])

    for start, opener, closer in candidates:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    snippet = text[start:i + 1]
                    try:
                        return json.loads(snippet)
                    except (ValueError, TypeError):
                        break
    return None


def _parse_location(loc: Any) -> Optional[Location]:
    """解析高德返回的 location 字段。可能是 'lng,lat' 字符串或 {longitude, latitude} dict。"""
    if not loc:
        return None
    try:
        if isinstance(loc, str):
            parts = loc.split(",")
            if len(parts) != 2:
                return None
            return Location(longitude=float(parts[0].strip()), latitude=float(parts[1].strip()))
        if isinstance(loc, dict):
            lng = loc.get("longitude") or loc.get("lng") or loc.get("lon")
            lat = loc.get("latitude") or loc.get("lat")
            if lng is None or lat is None:
                return None
            return Location(longitude=float(lng), latitude=float(lat))
    except (ValueError, TypeError):
        return None
    return None


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, list):
        # 高德有些字段返回 [] 表示空
        return None if not v else str(v)
    s = str(v).strip()
    return s if s else None


def _parse_photos(raw: Any) -> List[str]:
    """photos 字段可能是 [{title, url}, ...] / [str, ...] / str。"""
    if not raw:
        return []
    out: List[str] = []
    if isinstance(raw, str):
        return [raw] if raw.startswith("http") else []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                url = item.get("url") or item.get("photo_url") or item.get("src")
                if url:
                    out.append(url)
            elif isinstance(item, str) and item.startswith("http"):
                out.append(item)
    return out


def _parse_poi_item(item: Dict[str, Any]) -> Optional[POIInfo]:
    """解析单个 POI 项。仅 name 缺失才返回 None,其它字段尽量保留以待 detail 补全。

    兼容两种 amap 返回:
    - 高德 V3 完整版: rating/cost 在 biz_ext 子字典里
    - amap-mcp-server 精简版: rating/cost 在顶层,且无 location/biz_ext/photos
    """
    if not isinstance(item, dict):
        return None
    name = _safe_str(item.get("name"))
    if not name:
        return None

    location = _parse_location(item.get("location"))  # 可能为 None,由上层补全

    biz_ext = item.get("biz_ext")
    if not isinstance(biz_ext, dict):
        biz_ext = {}

    # 精简版 rating/cost 在顶层;完整版在 biz_ext;两边都试
    rating = _safe_float(item.get("rating") if item.get("rating") not in (None, "", []) else biz_ext.get("rating"))
    cost = _safe_str(item.get("cost") if item.get("cost") not in (None, "", []) else biz_ext.get("cost"))

    type_str = _safe_str(item.get("type")) or ""
    typecode = _safe_str(item.get("typecode"))
    biz_type = type_str.split(";")[0] if type_str else None

    return POIInfo(
        id=_safe_str(item.get("id")) or "",
        name=name,
        type=type_str,
        address=_safe_str(item.get("address")) or "",
        location=location,
        tel=_safe_str(item.get("tel")),
        rating=rating,
        cost=cost,
        photos=_parse_photos(item.get("photos")),
        biz_type=biz_type,
        typecode=typecode,
        level=_safe_str(item.get("level")),
    )


def _parse_pois(data: Any) -> List[POIInfo]:
    """从 JSON 中提取 POI 列表。data 可能是 dict 或 list。"""
    if data is None:
        return []
    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # 高德 V3: {"pois": [...]} ; MCP 简化版可能用 results/data
        for key in ("pois", "results", "data", "list"):
            v = data.get(key)
            if isinstance(v, list):
                items = v
                break

    out: List[POIInfo] = []
    for it in items:
        poi = _parse_poi_item(it)
        if poi is not None:
            out.append(poi)
    return out


def _parse_weather_cast(item: Dict[str, Any], date_fallback: str = "") -> Optional[WeatherInfo]:
    """单条天气预报解析。兼容高德 V3 / amap-mcp-server / 实时天气 等多种字段命名。"""
    if not isinstance(item, dict):
        return None
    # 字段名兼容映射：日期
    date_val = (
        item.get("date") or item.get("forecast_date") or item.get("reporttime") or date_fallback
    )
    # 白天/夜间天气描述
    day_weather = (
        item.get("dayweather") or item.get("day_weather") or
        item.get("weather") or item.get("day_text") or ""
    )
    night_weather = (
        item.get("nightweather") or item.get("night_weather") or
        item.get("night_text") or ""
    )
    # 温度
    day_temp = (
        item.get("daytemp") or item.get("day_temp") or
        item.get("temperature") or item.get("temp_day") or 0
    )
    night_temp = (
        item.get("nighttemp") or item.get("night_temp") or
        item.get("temp_night") or 0
    )
    # 风向 / 风力
    wind_dir = (
        item.get("daywind") or item.get("wind_direction") or
        item.get("winddirection") or ""
    )
    wind_power = (
        item.get("daypower") or item.get("wind_power") or
        item.get("windpower") or ""
    )
    return WeatherInfo(
        date=_safe_str(date_val) or date_fallback,
        day_weather=_safe_str(day_weather) or "",
        night_weather=_safe_str(night_weather) or "",
        day_temp=day_temp or 0,
        night_temp=night_temp or 0,
        wind_direction=_safe_str(wind_dir) or "",
        wind_power=_safe_str(wind_power) or "",
    )


def _parse_weather(data: Any) -> List[WeatherInfo]:
    """从 weather 接口返回中提取每日预报列表,兼容多种 mcp-server 实现。"""
    if data is None:
        return []
    casts: List[Dict[str, Any]] = []
    if isinstance(data, list):
        casts = data
    elif isinstance(data, dict):
        forecasts = data.get("forecasts")
        if isinstance(forecasts, list) and forecasts:
            first = forecasts[0]
            if isinstance(first, dict):
                # 高德 V3 标准: {"forecasts":[{"casts":[...]}]}
                nested = first.get("casts") or first.get("forecast")
                if isinstance(nested, list) and nested:
                    casts = nested
                # amap-mcp-server: forecasts 数组本身就是每日预报（无 casts 嵌套）
                # 判断方式: first 直接含 date/dayweather 等字段
                elif any(k in first for k in ("date", "dayweather", "weather")):
                    casts = forecasts
        # 兜底其它字段名
        if not casts:
            for key in ("casts", "forecast", "lives", "data", "return", "list"):
                v = data.get(key)
                if isinstance(v, list) and v:
                    casts = v
                    break
        # 实时天气只有一条记录,用根级字段
        if not casts and any(k in data for k in ("weather", "temperature", "winddirection")):
            casts = [data]

    out: List[WeatherInfo] = []
    for c in casts:
        w = _parse_weather_cast(c)
        if w is not None:
            out.append(w)
    return out


def _parse_route(data: Any) -> Dict[str, Any]:
    """解析路线规划。返回 {distance_m, duration_s, route_type, steps}。"""
    if not isinstance(data, dict):
        return {}
    # 高德返回结构: {"route": {"paths": [{"distance": "...", "duration": "...", "steps": [...]}]}}
    route = data.get("route") if isinstance(data.get("route"), dict) else data
    paths = route.get("paths") if isinstance(route, dict) else None
    if not isinstance(paths, list) or not paths:
        return {}
    first = paths[0]
    if not isinstance(first, dict):
        return {}
    try:
        distance_m = float(first.get("distance") or 0)
    except (ValueError, TypeError):
        distance_m = 0.0
    try:
        duration_s = float(first.get("duration") or 0)
    except (ValueError, TypeError):
        duration_s = 0.0
    return {
        "distance_m": distance_m,
        "duration_s": duration_s,
        "steps": first.get("steps") or [],
    }


# ============ 服务类 ============

class AmapService:
    """高德地图服务封装类"""

    def __init__(self):
        self.mcp_tool = None

    def _call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """统一的 MCP 调用入口。"""
        if self.mcp_tool is None:
            self.mcp_tool = get_amap_mcp_tool()
        return self.mcp_tool.run({
            "action": "call_tool",
            "tool_name": tool_name,
            "arguments": arguments,
        })

    def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """关键词搜索 POI。"""
        try:
            raw = self._call("maps_text_search", {
                "keywords": keywords,
                "city": city,
                "citylimit": str(citylimit).lower(),
            })
            data = _extract_json(raw)
            pois = _parse_pois(data)
            logger.info("POI[%s@%s] 解析得 %d 条", keywords, city, len(pois))
            return pois
        except Exception as e:
            logger.warning("POI搜索失败 keywords=%s city=%s err=%s", keywords, city, e)
            return []

    def around_search(
        self,
        location: Location,
        keywords: str,
        radius: int = 3000,
    ) -> List[POIInfo]:
        """以坐标为中心搜索周边 POI（如靠近景点的酒店）。"""
        try:
            loc_str = f"{location.longitude},{location.latitude}"
            raw = self._call("maps_around_search", {
                "location": loc_str,
                "keywords": keywords,
                "radius": str(radius),
            })
            data = _extract_json(raw)
            pois = _parse_pois(data)
            logger.info("Around[%s@%s,r=%d] 解析得 %d 条", keywords, loc_str, radius, len(pois))
            return pois
        except Exception as e:
            logger.warning("周边搜索失败 keywords=%s err=%s", keywords, e)
            return []

    def get_weather(self, city: str) -> List[WeatherInfo]:
        """查询城市天气预报。"""
        try:
            raw = self._call("maps_weather", {"city": city})
            data = _extract_json(raw)
            weather = _parse_weather(data)
            if not weather:
                # 解析得 0 天时记录 raw 便于诊断字段命名差异
                logger.warning(
                    "Weather[%s] 解析得 0 天,raw 前 500 字: %s",
                    city, (raw or "")[:500],
                )
            else:
                logger.info("Weather[%s] 解析得 %d 天", city, len(weather))
            return weather
        except Exception as e:
            logger.warning("天气查询失败 city=%s err=%s", city, e)
            return []

    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking",
    ) -> Dict[str, Any]:
        """按地址规划路线。"""
        tool_map = {
            "walking": "maps_direction_walking_by_address",
            "driving": "maps_direction_driving_by_address",
            "transit": "maps_direction_transit_integrated_by_address",
        }
        tool_name = tool_map.get(route_type, "maps_direction_walking_by_address")

        arguments: Dict[str, Any] = {
            "origin_address": origin_address,
            "destination_address": destination_address,
        }
        if origin_city:
            arguments["origin_city"] = origin_city
        if destination_city:
            arguments["destination_city"] = destination_city

        try:
            raw = self._call(tool_name, arguments)
            data = _extract_json(raw)
            result = _parse_route(data)
            result["route_type"] = route_type
            return result
        except Exception as e:
            logger.warning("路线规划失败 mode=%s err=%s", route_type, e)
            return {"route_type": route_type}

    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """地址转坐标。amap-mcp-server 用 'return' 字段,高德 V3 原版用 'geocodes'。"""
        try:
            arguments: Dict[str, Any] = {"address": address}
            if city:
                arguments["city"] = city
            raw = self._call("maps_geo", arguments)
            data = _extract_json(raw)
            if not isinstance(data, dict):
                return None
            items = data.get("return") or data.get("geocodes") or data.get("results") or []
            if not items:
                return None
            first = items[0] if isinstance(items[0], dict) else None
            if first is None:
                return None
            return _parse_location(first.get("location"))
        except Exception as e:
            logger.warning("地理编码失败 address=%s err=%s", address, e)
            return None

    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """获取 POI 详情。amap-mcp-server 直接返回扁平 dict;高德 V3 原版可能包在 pois/poi 里。"""
        try:
            raw = self._call("maps_search_detail", {"id": poi_id})
            data = _extract_json(raw)
            if not isinstance(data, dict):
                return {"raw": raw}

            # 兼容老格式: 如果有 pois/poi 包裹就解一层
            poi: Dict[str, Any] = data
            if "pois" in data and isinstance(data["pois"], list) and data["pois"]:
                poi = data["pois"][0]
            elif "poi" in data and isinstance(data["poi"], dict):
                poi = data["poi"]

            normalized = dict(poi)
            # photos 字段精简版没有,这里只在源 dict 有时才标准化
            if poi.get("photos"):
                normalized["photos"] = [{"url": u} for u in _parse_photos(poi.get("photos"))]
            return normalized
        except Exception as e:
            logger.warning("获取POI详情失败 poi_id=%s err=%s", poi_id, e)
            return {}


# 创建全局服务实例
_amap_service = None


def get_amap_service() -> AmapService:
    """获取高德地图服务实例(单例模式)"""
    global _amap_service

    if _amap_service is None:
        _amap_service = AmapService()

    return _amap_service

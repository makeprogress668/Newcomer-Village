"""高德 Web 服务 RESTful API 直接调用。

为什么不通过 mcp-server？
- mcp-server 启动 uvx 子进程 + stdio 通信,每次调用 ~3-5s
- HTTP 直连 ~0.5-1s,**快 5-10x**
- 且 extensions=all 一次返回 location + rating + photos 全部字段,
  无需二次 search_detail 调用

这里只用 requests 同步调,不依赖任何 MCP 框架。
"""

import logging
from typing import List, Optional

import requests

from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo

logger = logging.getLogger(__name__)


_AMAP_PLACE_TEXT = "https://restapi.amap.com/v3/place/text"
_AMAP_PLACE_AROUND = "https://restapi.amap.com/v3/place/around"
_AMAP_PLACE_DETAIL = "https://restapi.amap.com/v3/place/detail"
_AMAP_GEOCODE = "https://restapi.amap.com/v3/geocode/geo"
_AMAP_WEATHER = "https://restapi.amap.com/v3/weather/weatherInfo"


def _api_key() -> str:
    return get_settings().amap_api_key or ""


def get_poi_photos(name: str, city: str = "", limit: int = 5, timeout: int = 8) -> List[str]:
    """从高德 RESTful 文本搜索拉取首条 POI 的实景照片 URL 列表。"""
    key = _api_key()
    if not key or not name:
        return []
    try:
        resp = requests.get(
            _AMAP_PLACE_TEXT,
            params={
                "key": key,
                "keywords": name,
                "city": city or "",
                "extensions": "all",  # 关键: 返回 photos / biz_ext 等丰富字段
                "offset": str(min(limit, 25)),
                "page": "1",
                "output": "json",
            },
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("高德 REST 调用异常 [%s@%s]: %s", name, city, exc)
        return []

    if resp.status_code != 200:
        logger.warning("高德 REST 非 200 [%s@%s]: %d", name, city, resp.status_code)
        return []

    try:
        data = resp.json()
    except ValueError:
        return []

    if str(data.get("status")) != "1":
        logger.warning(
            "高德 REST status != 1 [%s@%s]: %s (info=%s,infocode=%s)",
            name, city, data.get("status"), data.get("info"), data.get("infocode"),
        )
        return []

    pois = data.get("pois") or []
    if not pois:
        return []
    # 取第一个 POI 的 photos
    first = pois[0]
    photos = first.get("photos") or []
    urls: List[str] = []
    if isinstance(photos, list):
        for p in photos:
            if isinstance(p, dict):
                u = p.get("url")
                if u and isinstance(u, str) and u.startswith("http"):
                    urls.append(u)
    return urls


def get_first_poi_photo(name: str, city: str = "") -> Optional[str]:
    """便捷接口: 拿第一张景点实景照。"""
    urls = get_poi_photos(name, city, limit=3)
    return urls[0] if urls else None


def search_pois_rest(
    keywords: str,
    city: str = "",
    limit: int = 10,
    timeout: int = 8,
) -> List[POIInfo]:
    """REST 直连版 POI 搜索 — 一次调用返回完整 POIInfo (含 location/rating/photos)。

    比 mcp-server 快 5-10x:
    - mcp 启动 uvx 子进程 + stdio 通信 ~3-5s/次
    - REST HTTP 直连 ~0.5-1s/次
    - extensions=all 直接返回 location/biz_ext.rating/photos,无需 detail 二次补全
    """
    key = _api_key()
    if not key or not keywords:
        return []
    try:
        resp = requests.get(
            _AMAP_PLACE_TEXT,
            params={
                "key": key,
                "keywords": keywords,
                "city": city or "",
                "extensions": "all",
                "offset": str(min(limit, 25)),
                "page": "1",
                "output": "json",
                "citylimit": "true",  # 限制在城市内
            },
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("REST POI 搜索异常 [%s@%s]: %s", keywords, city, exc)
        return []

    if resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    if str(data.get("status")) != "1":
        return []

    pois_raw = data.get("pois") or []
    out: List[POIInfo] = []
    for item in pois_raw:
        poi = _parse_rest_poi_item(item)
        if poi is not None:
            out.append(poi)
    return out


def search_pois_around_rest(
    location: Location,
    keywords: str,
    city: str = "",
    radius: int = 3000,
    limit: int = 20,
    timeout: int = 5,
) -> List[POIInfo]:
    """REST 直连周边搜索。用于酒店推荐,避免走 MCP around_search 的秒级开销。"""
    key = _api_key()
    if not key or not keywords or location is None:
        return []
    try:
        resp = requests.get(
            _AMAP_PLACE_AROUND,
            params={
                "key": key,
                "location": f"{location.longitude},{location.latitude}",
                "keywords": keywords,
                "city": city or "",
                "radius": str(radius),
                "extensions": "all",
                "offset": str(min(limit, 25)),
                "page": "1",
                "output": "json",
            },
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("REST 周边搜索异常 [%s,r=%s]: %s", keywords, radius, exc)
        return []

    if resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    if str(data.get("status")) != "1":
        return []

    out: List[POIInfo] = []
    for item in data.get("pois") or []:
        poi = _parse_rest_poi_item(item)
        if poi is not None:
            out.append(poi)
    return out


def get_weather_rest(city: str, timeout: int = 5) -> List[WeatherInfo]:
    """REST 直连天气预报。高德返回当天 + 未来 3 天。"""
    key = _api_key()
    if not key or not city:
        return []
    try:
        resp = requests.get(
            _AMAP_WEATHER,
            params={
                "key": key,
                "city": city,
                "extensions": "all",
                "output": "json",
            },
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("REST 天气查询异常 [%s]: %s", city, exc)
        return []

    if resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    if str(data.get("status")) != "1":
        return []

    forecasts = data.get("forecasts") or []
    if not forecasts or not isinstance(forecasts[0], dict):
        return []
    casts = forecasts[0].get("casts") or []
    out: List[WeatherInfo] = []
    for item in casts:
        if not isinstance(item, dict):
            continue
        out.append(WeatherInfo(
            date=str(item.get("date") or ""),
            day_weather=str(item.get("dayweather") or ""),
            night_weather=str(item.get("nightweather") or ""),
            day_temp=item.get("daytemp") or 0,
            night_temp=item.get("nighttemp") or 0,
            wind_direction=str(item.get("daywind") or ""),
            wind_power=str(item.get("daypower") or ""),
        ))
    return out


def _parse_rest_poi_item(item: dict) -> Optional[POIInfo]:
    """把高德 REST 返回的单个 POI 解析为 POIInfo。"""
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    if not name or not isinstance(name, str):
        return None

    # location: "lng,lat" 字符串
    location: Optional[Location] = None
    loc_str = item.get("location")
    if isinstance(loc_str, str) and "," in loc_str:
        try:
            lng_s, lat_s = loc_str.split(",", 1)
            location = Location(longitude=float(lng_s), latitude=float(lat_s))
        except (ValueError, TypeError):
            location = None

    # biz_ext 可能是 dict 或 [] (空)
    biz_ext = item.get("biz_ext")
    if not isinstance(biz_ext, dict):
        biz_ext = {}

    rating: Optional[float] = None
    rating_raw = biz_ext.get("rating")
    if rating_raw not in (None, "", []):
        try:
            rating = float(rating_raw)
        except (ValueError, TypeError):
            rating = None

    cost = biz_ext.get("cost")
    if not isinstance(cost, str) or not cost.strip():
        cost = None

    # photos: list of {"title", "url"}
    photos_raw = item.get("photos") or []
    photos: List[str] = []
    if isinstance(photos_raw, list):
        for p in photos_raw:
            if isinstance(p, dict):
                u = p.get("url")
                if isinstance(u, str) and u.startswith("http"):
                    photos.append(u)

    type_str = item.get("type") if isinstance(item.get("type"), str) else ""
    typecode = item.get("typecode") if isinstance(item.get("typecode"), str) else ""
    biz_type = type_str.split(";")[0] if type_str else None

    address = item.get("address")
    if not isinstance(address, str):
        address = ""

    return POIInfo(
        id=item.get("id") or "",
        name=name,
        type=type_str,
        address=address,
        location=location,
        rating=rating,
        cost=cost,
        photos=photos,
        biz_type=biz_type,
        typecode=typecode,
    )


def geocode_rest(address: str, city: str = "", timeout: int = 5):
    """直接调高德 RESTful geocode (HTTP 直连,比 mcp-server stdio 快 5x)。

    返回 (longitude, latitude) 元组,失败返回 None。
    必传 city 参数避免"外滩"被命中其他城市同名地址。
    """
    key = _api_key()
    if not key or not address:
        return None
    try:
        resp = requests.get(
            _AMAP_GEOCODE,
            params={
                "key": key,
                "address": address,
                "city": city or "",
                "output": "json",
            },
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("REST geocode 调用异常 [%s@%s]: %s", address, city, exc)
        return None

    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if str(data.get("status")) != "1":
        return None
    geocodes = data.get("geocodes") or []
    if not geocodes:
        return None
    first = geocodes[0]
    if not isinstance(first, dict):
        return None
    loc_str = first.get("location")
    if not isinstance(loc_str, str):
        return None
    parts = loc_str.split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, TypeError):
        return None

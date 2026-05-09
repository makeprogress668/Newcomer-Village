"""POI相关API路由。"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...data.keywords import to_english_query
from ...services.amap_rest_service import get_first_poi_photo as get_amap_rest_photo
from ...services.amap_service import get_amap_service
from ...services.baike_service import get_baike_photo
from ...services.image_cache import get_cache_stats, get_image_cache
from ...services.unsplash_service import get_unsplash_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/poi", tags=["POI"])


class POIDetailResponse(BaseModel):
    """POI详情响应。"""
    success: bool
    message: str
    data: Optional[dict] = None


@router.get(
    "/detail/{poi_id}",
    response_model=POIDetailResponse,
    summary="获取POI详情",
    description="根据POI ID获取详细信息,包括图片",
)
async def get_poi_detail(poi_id: str):
    try:
        amap_service = get_amap_service()
        result = amap_service.get_poi_detail(poi_id)
        return POIDetailResponse(success=True, message="获取POI详情成功", data=result)
    except Exception as e:
        logger.exception("获取POI详情失败 poi_id=%s", poi_id)
        raise HTTPException(status_code=500, detail=f"获取POI详情失败: {str(e)}")


@router.get(
    "/search",
    summary="搜索POI",
    description="根据关键词搜索POI",
)
async def search_poi(keywords: str, city: str = "北京"):
    try:
        amap_service = get_amap_service()
        result = amap_service.search_poi(keywords, city)
        return {"success": True, "message": "搜索成功", "data": result}
    except Exception as e:
        logger.exception("搜索POI失败 keywords=%s city=%s", keywords, city)
        raise HTTPException(status_code=500, detail=f"搜索POI失败: {str(e)}")


def _amap_first_photo(name: str, city: str) -> Optional[str]:
    """从高德拉景点实景照片。

    优先级:
    1) 高德 RESTful Web 服务 API (extensions=all) — 直接返回真实实景照
       (mcp-server 精简版去掉了 photos 字段,所以必须走 RESTful)
    2) 回退: mcp-server search_poi → get_poi_detail (兼容老格式)
    """
    # 1) 高德 RESTful (主源,拿真实实景照,非 logo)
    try:
        url = get_amap_rest_photo(name, city)
        if url:
            return url
    except Exception as exc:
        logger.warning("高德 REST 取图异常 [%s@%s]: %s", name, city, exc)

    # 2) 回退: mcp-server (大概率没图,但保留兼容)
    try:
        amap_service = get_amap_service()
        poi_list = amap_service.search_poi(name, city or "全国")
        if not poi_list:
            return None
        first = poi_list[0]
        if first.photos:
            return first.photos[0]
        if first.id:
            detail = amap_service.get_poi_detail(first.id)
            photos = detail.get("photos") or []
            if photos:
                first_photo = photos[0]
                if isinstance(first_photo, dict):
                    return first_photo.get("url") or first_photo.get("photo_url")
                if isinstance(first_photo, str):
                    return first_photo
    except Exception as exc:
        logger.warning("高德 mcp 取图失败 [%s@%s]: %s", name, city, exc)
    return None


def _unsplash_photo(name: str, city: str) -> Optional[str]:
    """Unsplash 兜底。Unsplash 中文搜索命中率极低,优先用英文专名。"""
    try:
        unsplash_service = get_unsplash_service()
        queries: list = []

        # 第一优先：中→英映射的专有名词（如"故宫博物院"→"Forbidden City Beijing"）
        en_query = to_english_query(name, city)
        if en_query and en_query != f"{name} {city} landmark".strip():
            # 命中映射的英文专名,最准确
            queries.append(en_query)

        # 第二优先：name + city + landmark
        if city:
            queries.append(f"{name} {city} landmark")
        queries.append(f"{name} landmark")

        # 第三优先（兜底）：城市 + architecture
        if city:
            queries.append(f"{city} architecture")

        for q in queries:
            url = unsplash_service.get_photo_url(q)
            if url:
                logger.info("Unsplash 命中 [%s@%s] query=%s", name, city, q)
                return url
    except Exception as exc:
        logger.warning("Unsplash 取图失败 [%s@%s]: %s", name, city, exc)
    return None


@router.get(
    "/photo",
    summary="获取景点图片",
    description="高德主源 + Unsplash 兜底 + SQLite 缓存。",
)
async def get_attraction_photo(name: str, city: str = ""):
    """
    取图链路:
      1) SQLite 缓存命中 → 直接返回
      2) amap.search_poi 取 photos[0]
      3) Unsplash 兜底（多级查询）
      4) 写入缓存
    """
    if not name:
        raise HTTPException(status_code=400, detail="景点名称不能为空")

    cache = get_image_cache()
    cached = cache.get(name, city)
    if cached:
        return {
            "success": True,
            "message": "获取图片成功(缓存)",
            "data": {"name": name, "city": city, "photo_url": cached["url"], "source": cached["source"]},
        }

    # 取图链路: 高德 photos → 百度百科 → Unsplash 兜底
    photo_url: Optional[str] = _amap_first_photo(name, city)
    source = "amap"
    if not photo_url:
        photo_url = get_baike_photo(name, city)
        source = "baike"
    if not photo_url:
        photo_url = _unsplash_photo(name, city)
        source = "unsplash"

    if photo_url:
        cache.set(name, city, photo_url, source)

    return {
        "success": True,
        "message": "获取图片成功" if photo_url else "未找到图片",
        "data": {"name": name, "city": city, "photo_url": photo_url, "source": source if photo_url else None},
    }


@router.get(
    "/photo/stats",
    summary="图片缓存命中率统计",
    description="返回当前进程的命中/未命中/写入次数和命中率",
)
async def photo_stats():
    return {"success": True, "data": get_cache_stats()}

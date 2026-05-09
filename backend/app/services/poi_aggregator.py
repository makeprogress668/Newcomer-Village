"""景点聚合：用关键词派发 + 高德拉数据 + 详情补全 + 评分排序。

由于 amap-mcp-server 的 maps_text_search 返回精简字段（仅 id/name/address/typecode），
本模块在去重后会对 top 候选并发调 maps_search_detail 二次补全 location/rating/level/photos。

流程:
1. preferences → 关键词候选（含保底词），见 data.keywords.expand_preferences
2. ThreadPoolExecutor 并发调 amap.search_poi
3. 按 poi_id 去重
4. 粗筛：typecode 过滤掉餐饮/购物/住宿等非景点；type 字符串老规则也保留（兼容完整版）
5. 粗排序：主景点（无 '-' 子点位）+ A 级景区 优先
6. 对 top_n*2 候选并发拉 detail，补全 location/rating/level/type/photos
7. 丢弃仍无 location 的（没法做行程优化）
8. rating 过滤 + 综合分数最终排序
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ..data.keywords import (
    expand_free_text,
    expand_preferences,
    get_must_visit,
    is_non_attraction,
    is_within_city,
    landmark_priority,
)
from ..data.curated_pois import get_curated_pois
from ..models.schemas import POIInfo
from ..config import get_settings
from .amap_service import (
    _parse_location,
    _parse_photos,
    _safe_float,
    _safe_str,
    get_amap_service,
)

logger = logging.getLogger(__name__)


# ============ typecode 过滤 ============

# 高德 typecode 前缀 → 明显非景点
_NON_ATTRACTION_PREFIXES = (
    "05",  # 餐饮服务
    "06",  # 购物服务
    "07",  # 生活服务
    "10",  # 住宿服务
    "12",  # 商务住宅
    "13",  # 政府机构及社会团体
    "15",  # 交通设施服务
    "16",  # 金融保险服务
    "17",  # 公司企业
    "18",  # 道路附属设施
    "19",  # 地名地址信息（行政区/街道）
    "20",  # 公共设施
    "22",  # 通行设施
    "23",  # 室内设施
    "24",  # 室内地标
    "25",  # 摆渡停靠点
    "97",  # 室内楼层
    "99",  # 占位
)

# 明确属于景点的前缀
_ATTRACTION_PREFIXES = (
    "11",   # 风景名胜
    "14",   # 科教文化服务（含博物馆 1401、纪念馆 1402、文化场馆 1403 等）
    "080",  # 体育休闲服务（公园广场也包含一些景观）
)


def _is_attraction_typecode(typecode: Optional[str]) -> bool:
    """根据高德 typecode 粗筛是否景点。typecode 可用 '|' 分隔多分类码。"""
    if not typecode:
        return True  # 缺失则不过滤，留给后续 type 字符串判断
    codes = [c.strip() for c in typecode.split("|") if c.strip()]
    if not codes:
        return True
    # 任一码命中"景点类前缀"就保留
    for code in codes:
        if any(code.startswith(p) for p in _ATTRACTION_PREFIXES):
            return True
    # 否则检查是否全部命中"非景点前缀"
    if all(any(code.startswith(p) for p in _NON_ATTRACTION_PREFIXES) for code in codes):
        return False
    # 含未知前缀（既不在景点也不在非景点列表）→ 保留
    return True


# ============ 排序权重 ============

_LEVEL_WEIGHT = {
    "AAAAA": 2.0, "5A": 2.0,   # 5A 是绝对地标 (故宫/长城/颐和园),必须强势加权
    "AAAA": 1.0, "4A": 1.0,
    "AAA": 0.4, "3A": 0.4,
    "AA": 0.15, "2A": 0.15,
    "A": 0.05, "1A": 0.05,
}


def _level_weight(level: Optional[str]) -> float:
    if not level:
        return 0.0
    return _LEVEL_WEIGHT.get(level.upper().strip(), 0.0)


def _is_main_branch(name: str) -> bool:
    """是否是主景点（不带 '-' 子点位、不带括号）。"""
    return ("-" not in name) and ("(" not in name) and ("（" not in name)


def _pre_score(poi: POIInfo) -> float:
    """补全前的粗排分：决定哪些候选先去拉详情。"""
    score = 0.0
    if _is_main_branch(poi.name):
        score += 1.0
    score += _level_weight(poi.level)
    return score


_POI_NAME_BLACKLIST = (
    "停车场", "售票处", "票务", "入口", "出口", "卫生间", "厕所", "游客中心",
    "服务中心", "管理处", "派出所", "办公室", "公交站", "地铁站", "码头售票",
)


def _is_noise_poi_name(name: str) -> bool:
    if not name:
        return True
    return any(kw in name for kw in _POI_NAME_BLACKLIST)


def _final_score(poi: POIInfo, city: str = "") -> float:
    """补全 detail 后的最终排序分数。"""
    rating = poi.rating if poi.rating is not None else 3.5
    rating_weight = 1.0 if poi.rating is not None else 0.7
    score = rating * 0.6 * rating_weight
    score += _level_weight(poi.level) * 1.5  # A 级景区是非常强的信号
    score += landmark_priority(city, poi.name)
    if _is_main_branch(poi.name):
        score += 0.5
    else:
        score -= 0.6
    score += math.log1p(len(poi.photos)) * 0.1
    return score


def _category_key(poi: POIInfo) -> str:
    text = f"{poi.name}{poi.type or ''}{poi.biz_type or ''}"
    if any(kw in text for kw in ("博物馆", "美术馆", "艺术", "展览", "文化")):
        return "文化场馆"
    if any(kw in text for kw in ("街", "巷", "路", "商圈", "步行街", "古镇", "新天地")):
        return "街区体验"
    if any(kw in text for kw in ("公园", "湖", "山", "湿地", "海", "湾")):
        return "自然休闲"
    if any(kw in text for kw in ("寺", "宫", "城", "塔", "祠", "陵", "古迹")):
        return "历史古迹"
    if any(kw in text for kw in ("乐园", "动物园", "熊猫", "主题")):
        return "亲子娱乐"
    return "城市地标"


def _diversified_top(pois: List[POIInfo], limit: int, city: str) -> List[POIInfo]:
    """在保持分数优先的前提下控制类别单一问题。"""
    selected: List[POIInfo] = []
    category_counts: Dict[str, int] = {}
    soft_cap = 3 if limit >= 9 else 2

    for poi in pois:
        cat = _category_key(poi)
        is_landmark = landmark_priority(city, poi.name) >= 3.0
        if category_counts.get(cat, 0) >= soft_cap and not is_landmark:
            continue
        selected.append(poi)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if len(selected) >= limit:
            return selected

    selected_ids = {p.id for p in selected}
    for poi in pois:
        if poi.id in selected_ids:
            continue
        selected.append(poi)
        if len(selected) >= limit:
            break
    return selected


# ============ detail 补全 ============

def _enrich_with_detail(poi: POIInfo, amap_service) -> POIInfo:
    """用 maps_search_detail 补全缺失的 location/rating/level/type/photos。"""
    if poi.location is not None and poi.rating is not None and poi.level is not None:
        return poi
    if not poi.id:
        return poi
    try:
        detail = amap_service.get_poi_detail(poi.id)
    except Exception as exc:
        logger.warning("详情补全失败 poi_id=%s err=%s", poi.id, exc)
        return poi
    if not detail:
        return poi

    if poi.location is None:
        poi.location = _parse_location(detail.get("location"))
    if poi.rating is None:
        poi.rating = _safe_float(detail.get("rating"))
    if not poi.type:
        type_str = _safe_str(detail.get("type")) or ""
        poi.type = type_str
        if type_str and not poi.biz_type:
            poi.biz_type = type_str.split(";")[0]
    if not poi.level:
        poi.level = _safe_str(detail.get("level"))
    if not poi.cost:
        poi.cost = _safe_str(detail.get("cost"))
    if not poi.photos:
        poi.photos = _parse_photos(detail.get("photos"))
    return poi


# ============ 入口 ============

def collect_attractions(
    city: str,
    preferences: Optional[List[str]] = None,
    free_text: Optional[str] = None,
    top_n: int = 15,
    min_rating: float = 4.0,
    per_keyword_limit: int = 20,
    max_workers: int = 4,
    detail_workers: int = 5,
    detail_pool_factor: int = 2,
    must_include: Optional[List[str]] = None,
) -> List[POIInfo]:
    """
    聚合城市景点。

    Args:
        city: 目的地城市
        preferences: 用户偏好标签列表
        free_text: 用户自由输入文本（如"想看升旗仪式"），会被解析为额外景点关键词
        top_n: 最终返回的景点数
        min_rating: 评分门槛（评分缺失不过滤但降权）
        per_keyword_limit: 每个关键词最多保留多少条
        max_workers: 关键词并发数
        detail_workers: detail 补全并发数
        detail_pool_factor: 拉详情时取 top_n * factor 作为候选
        must_include: 强制保留的景点关键词（来自 free_text 的高优先诉求会被强制保留在最终列表）

    Returns:
        按综合评分降序的 POIInfo 列表，长度 <= top_n
    """
    keywords = expand_preferences(preferences or [])

    # 把 free_text 中识别出的景点关键词追加进搜索集合 (用户特殊诉求)
    free_text_kws = expand_free_text(free_text)
    city_must = get_must_visit(city)
    # 只有用户明确提到的地点才强制保留；城市地标只作为搜索词和排序加权。
    must_include = list(must_include or []) + free_text_kws
    for kw in free_text_kws:
        if kw not in keywords:
            keywords.append(kw)

    # 城市高热度地标作为额外搜索词,并在最终排序里加权；不硬塞进最终行程。
    for kw in city_must:
        if kw not in keywords:
            keywords.append(kw)

    # P20: 关键词收敛上限,防止 12+ 个关键词拖慢主流程
    if len(keywords) > 8:
        keywords = keywords[:8]
        logger.info("关键词收敛到 %d 个 (避免过多 API 调用)", len(keywords))

    if not keywords:
        return []
    if city_must:
        logger.info("📌 城市保底地标已加入搜索关键词: %s", city_must)

    # P20: 优先 REST 直连 (快 5-10x), mcp 作为 fallback
    from .amap_rest_service import search_pois_rest
    amap = get_amap_service()  # 仅 detail 补全 fallback 用

    def _fetch(kw: str) -> List[POIInfo]:
        # 1) 优先 REST: HTTP 直连 + 一次返回完整字段
        try:
            pois = search_pois_rest(kw, city, limit=per_keyword_limit)
            if pois:
                return pois
        except Exception as exc:
            logger.warning("REST 搜索 [%s] 异常: %s", kw, exc)
        if not get_settings().enable_mcp_tools:
            return []
        # 2) Fallback: mcp (网络不通时兜底)
        try:
            pois = amap.search_poi(kw, city, citylimit=True)
        except Exception as exc:
            logger.warning("mcp fallback 搜索 [%s] 异常: %s", kw, exc)
            return []
        return pois[:per_keyword_limit]

    raw_buckets: List[List[POIInfo]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(_fetch, keywords):
            raw_buckets.append(result)
    curated = get_curated_pois(city, keywords)
    if curated:
        raw_buckets.append(curated)

    # 去重 by id（无 id 的退化用 name+address 作 key）
    merged: Dict[str, POIInfo] = {}
    seen_names: set = set()
    raw_count = 0
    for bucket in raw_buckets:
        for poi in bucket:
            raw_count += 1
            name_key = poi.name.strip()
            if name_key in seen_names:
                continue
            key = poi.id or f"name:{poi.name}|{poi.address}"
            if key not in merged:
                merged[key] = poi
                seen_names.add(name_key)

    # 粗筛：typecode 过滤 + 兼容老 type 字符串
    filtered: List[POIInfo] = []
    for poi in merged.values():
        if _is_noise_poi_name(poi.name):
            continue
        if poi.type and is_non_attraction(poi.type):
            continue
        if not _is_attraction_typecode(poi.typecode):
            continue
        filtered.append(poi)

    # 粗排序：主景点 + A 级 优先（精简返回里没 level，但 type 含 "5A级景区" 的会先排上）
    filtered.sort(key=_pre_score, reverse=True)

    # 取候选去拉详情
    candidate_count = max(top_n * detail_pool_factor, 20)
    candidates = filtered[:candidate_count]

    # must_include 命中的 POI 即使粗排靠后也要强制进入 candidates（否则没 location 就被淘汰）
    if must_include:
        candidate_ids = {p.id for p in candidates}
        for kw in must_include:
            for p in filtered:
                if p.id in candidate_ids:
                    continue
                if kw in p.name:
                    candidates.insert(0, p)  # 放最前面优先补全
                    candidate_ids.add(p.id)
                    logger.info("强制纳入 candidates (free_text 命中): %s [%s]", p.name, kw)
                    break  # 每关键词最多 1 个

    # P20: 智能 detail 补全 - 大部分 POI 走 REST 后已有 location/rating/photos,
    # 只对仍缺关键字段的 POI 调 detail (从全量 30 个 → 通常 0-3 个)
    need_enrich = [
        p for p in candidates
        if p.location is None or (p.rating is None and not p.photos)
    ]
    no_enrich = [p for p in candidates if p not in need_enrich]
    if need_enrich:
        logger.info("仅 %d/%d 个 POI 需要 detail 补全 (REST 已覆盖大部分)",
                    len(need_enrich), len(candidates))
        def _enrich(p: POIInfo) -> POIInfo:
            return _enrich_with_detail(p, amap)
        with ThreadPoolExecutor(max_workers=detail_workers) as pool:
            enriched_part = list(pool.map(_enrich, need_enrich))
        candidates = no_enrich + enriched_part
    # else: 完全跳过 detail 补全 (REST 已经够用,可省 20-30s)

    # 必须有 location 才能进行行程优化
    enriched = [p for p in candidates if p.location is not None]

    # 跨城过滤: 把坐标在 100km 城市范围外的 POI 剔除
    # (防御性: 即使高德/geocode 偶尔返回错误坐标也能拦住)
    if city:
        before = len(enriched)
        enriched = [
            p for p in enriched
            if is_within_city(p.location.longitude, p.location.latitude, city)
        ]
        if len(enriched) < before:
            logger.warning(
                "🚫 跨城过滤剔除 %d 个不在 %s 范围的 POI", before - len(enriched), city,
            )

    # rating 过滤（明确低于门槛才过滤；缺失保留）
    survived = [p for p in enriched if not (p.rating is not None and p.rating < min_rating)]

    # 最终排序
    survived.sort(key=lambda p: _final_score(p, city), reverse=True)
    selected = _diversified_top(survived, top_n, city)

    # P20: 移除 _rescue_must_include_pois 调用 — REST 直连后 99% POI 已有 location,
    # 不再需要 geocode 救回。如果 must_include 个别词没命中,接受"自然排序",
    # 让 _final_score 选出真正受欢迎的景点 (用户反馈"不需要那么固定")。

    # 强制保留: 在 survived(已有 location) 中找 must_include 命中并放到头部。
    # 这样能稳定命中超级地标,避免纯评分排序把冷门博物馆/子点位排到前面。
    if must_include:
        selected_ids = {p.id for p in selected}
        hits: List[POIInfo] = []
        for kw in must_include:
            for p in survived:
                if p.id in selected_ids:
                    continue
                if kw in p.name:
                    hits.append(p)
                    selected_ids.add(p.id)
                    logger.info("强制保留必去景点: %s (关键词=%s)", p.name, kw)
                    break
        # 头部插入并去重,让 itinerary_optimizer 截断时优先保留
        selected = hits + selected
        seen = set()
        deduped: List[POIInfo] = []
        for p in selected:
            key = p.name.strip() or p.id or f"{p.name}|{p.address}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        selected = deduped[:top_n]

    # 命中率告警: must_include 至少 50% 应该被命中,否则记 warn 让用户/开发者关注
    if must_include:
        hit_count = sum(1 for kw in must_include
                        if any(kw in s.name for s in selected))
        if hit_count < len(must_include) * 0.5:
            logger.warning(
                "⚠️ 必去地标命中率低: %d/%d (该城市可能在 CITY_MUST_VISIT 之外,或高德搜不到)",
                hit_count, len(must_include),
            )

    logger.info(
        "景点聚合 city=%s keywords=%d raw=%d 去重=%d typecode过滤后=%d 详情补全后=%d 最终=%d",
        city, len(keywords), raw_count, len(merged),
        len(filtered), len(enriched), len(selected),
    )
    if selected:
        sample = ", ".join(f"{p.name}({p.level or '无评级'},{p.rating or '?'}★)" for p in selected[:5])
        logger.info("景点 top5: %s", sample)
    return selected


def _rescue_must_include_pois(
    merged: Dict[str, POIInfo],
    must_include: List[str],
    already_selected: List[POIInfo],
    city: str = "",
) -> List[POIInfo]:
    """对 must_include 关键词命中但 location 缺失的 POI,用 REST geocode 救回。

    P19 重写:
    - 走 amap_rest_service.geocode_rest (HTTP 直连,比 mcp stdio 快 5x)
    - 用 ThreadPoolExecutor 并发跑多个 geocode (3 个必去 → 8s 而非 24s)
    - 必传 city 限制范围,加 is_within_city 双保险
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .amap_rest_service import geocode_rest
    from ..models.schemas import Location

    selected_ids = {p.id for p in already_selected}

    # 1. 收集需要救回的目标 POI(每关键词最多 1 个)
    targets: List[POIInfo] = []
    for kw in must_include:
        for p in merged.values():
            if p.id in selected_ids or p.id in {t.id for t in targets}:
                continue
            if kw not in p.name:
                continue
            if p.location is not None or not p.address:
                continue
            targets.append(p)
            break

    if not targets:
        return []

    # 2. 并发 geocode (5 个 worker, 每次 5s 超时)
    def _geocode_one(poi: POIInfo):
        result = geocode_rest(poi.address, city=city, timeout=5)
        return poi, result

    rescued: List[POIInfo] = []
    with ThreadPoolExecutor(max_workers=min(5, len(targets))) as pool:
        futures = [pool.submit(_geocode_one, t) for t in targets]
        try:
            for fut in as_completed(futures, timeout=15):
                try:
                    poi, coords = fut.result(timeout=6)
                except Exception as exc:
                    logger.warning("rescue geocode 异常: %s", exc)
                    continue
                if coords is None:
                    continue
                lng, lat = coords
                # 双保险: 跨城检查
                if city and not is_within_city(lng, lat, city):
                    logger.warning(
                        "🚫 救回的 %s 坐标 (%.4f,%.4f) 不在 %s 范围,丢弃",
                        poi.name, lng, lat, city,
                    )
                    continue
                poi.location = Location(longitude=lng, latitude=lat)
                rescued.append(poi)
                logger.info("🆘 救回必去地标 (REST geocode): %s @ %s", poi.name, poi.address)
        except Exception as exc:
            # as_completed 整体超时会抛 TimeoutError,主流程不能因此崩溃
            logger.warning("rescue 整体超时或异常 (%s),已救回 %d/%d", exc, len(rescued), len(targets))
    return rescued

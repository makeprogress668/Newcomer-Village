"""行程优化：把候选景点按地理聚类分天，再在每天内做 TSP 最短路径。

核心目的是解决"行程绕路"——同一天的景点应当地理临近，跨天的安排应贴近酒店。
不引入 numpy/sklearn，纯 stdlib 实现。

主要 API:
- haversine(loc_a, loc_b)             : 球面距离（公里）
- cluster_by_day(pois, n_days)        : k-means 聚类，返回每天的景点桶
- order_within_day(pois, start)       : 最近邻 + 2-opt 求短路径
- estimate_commute(loc_a, loc_b, mode): 估算通勤距离/时间
- optimize(pois, n_days, ...)         : 顶层封装，返回 List[List[POIInfo]]
"""

import math
import random
from typing import List, Optional, Tuple

from ..models.schemas import Location, POIInfo


# ============ 距离计算 ============

EARTH_RADIUS_KM = 6371.0

# 各交通方式的平均速度 (km/h)
SPEED_KM_H = {
    "walking": 5.0,
    "driving": 30.0,
    "transit": 15.0,
    "bicycling": 12.0,
}


def haversine(a: Location, b: Location) -> float:
    """两点球面距离 (km)。"""
    if a is None or b is None:
        return 0.0
    lat1 = math.radians(a.latitude)
    lat2 = math.radians(b.latitude)
    dlat = lat2 - lat1
    dlng = math.radians(b.longitude - a.longitude)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def estimate_commute(a: Location, b: Location, mode: str = "driving") -> dict:
    """估算通勤距离/时间。返回 {distance_km, duration_min, mode}。"""
    dist = haversine(a, b)
    speed = SPEED_KM_H.get(mode, SPEED_KM_H["driving"])
    duration_min = dist / speed * 60.0
    return {
        "distance_km": round(dist, 2),
        "duration_min": round(duration_min, 1),
        "mode": mode,
    }


# ============ 聚类 ============

def _centroid(pois: List[POIInfo]) -> Optional[Location]:
    if not pois:
        return None
    n = len(pois)
    avg_lng = sum(p.location.longitude for p in pois) / n
    avg_lat = sum(p.location.latitude for p in pois) / n
    return Location(longitude=avg_lng, latitude=avg_lat)


def cluster_by_day(
    pois: List[POIInfo],
    n_days: int,
    max_iter: int = 30,
    seed: int = 42,
) -> List[List[POIInfo]]:
    """简易 k-means 聚类，把景点分成 n_days 组。

    初始化策略：按经度排序后均匀取 n_days 个点作为初始质心，避免随机不稳定。
    """
    n_days = max(1, n_days)
    if not pois:
        return [[] for _ in range(n_days)]
    if len(pois) <= n_days:
        # 点比天数还少，每天最多一个
        return [[p] for p in pois] + [[] for _ in range(n_days - len(pois))]

    # 初始质心：按经度排序均匀取 n_days 个
    sorted_pois = sorted(pois, key=lambda p: (p.location.longitude, p.location.latitude))
    step = len(sorted_pois) / n_days
    centers: List[Location] = [
        sorted_pois[min(int(i * step + step / 2), len(sorted_pois) - 1)].location
        for i in range(n_days)
    ]

    rng = random.Random(seed)
    assignments = [0] * len(pois)
    for _ in range(max_iter):
        # 分配
        new_assignments = []
        for p in pois:
            best_idx = 0
            best_d = float("inf")
            for i, c in enumerate(centers):
                d = haversine(p.location, c)
                if d < best_d:
                    best_d = d
                    best_idx = i
            new_assignments.append(best_idx)

        # 重算质心
        new_centers: List[Location] = []
        for i in range(n_days):
            cluster_pts = [pois[j] for j in range(len(pois)) if new_assignments[j] == i]
            if not cluster_pts:
                # 空簇：扰动一个随机景点的位置作为新质心，避免簇被吞掉
                rand_p = rng.choice(pois)
                new_centers.append(rand_p.location)
            else:
                new_centers.append(_centroid(cluster_pts))

        if new_assignments == assignments:
            centers = new_centers
            break
        assignments = new_assignments
        centers = new_centers

    buckets: List[List[POIInfo]] = [[] for _ in range(n_days)]
    for poi, idx in zip(pois, assignments):
        buckets[idx].append(poi)
    return buckets


# ============ TSP（每日内排序） ============

def _path_length(path: List[POIInfo], start: Optional[Location]) -> float:
    if not path:
        return 0.0
    total = 0.0
    if start is not None:
        total += haversine(start, path[0].location)
    for i in range(len(path) - 1):
        total += haversine(path[i].location, path[i + 1].location)
    return total


def _nearest_neighbor(pois: List[POIInfo], start: Optional[Location]) -> List[POIInfo]:
    if not pois:
        return []
    remaining = list(pois)
    out: List[POIInfo] = []
    cursor = start
    if cursor is None:
        # 没有起点：先取列表中最靠"西南"的（lng+lat 最小）作为起点
        first = min(remaining, key=lambda p: p.location.longitude + p.location.latitude)
        out.append(first)
        remaining.remove(first)
        cursor = first.location
    while remaining:
        nxt = min(remaining, key=lambda p: haversine(cursor, p.location))
        out.append(nxt)
        remaining.remove(nxt)
        cursor = nxt.location
    return out


def _two_opt(path: List[POIInfo], start: Optional[Location], max_iter: int = 50) -> List[POIInfo]:
    """简易 2-opt：反转任意子段如果能让总路径更短。规模 ≤6 收敛极快。"""
    if len(path) < 3:
        return path
    best = list(path)
    best_len = _path_length(best, start)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                cand_len = _path_length(candidate, start)
                if cand_len + 1e-9 < best_len:
                    best = candidate
                    best_len = cand_len
                    improved = True
    return best


def order_within_day(
    pois: List[POIInfo],
    start: Optional[Location] = None,
) -> List[POIInfo]:
    """最近邻 + 2-opt 给单日景点排序。"""
    if not pois:
        return []
    nn = _nearest_neighbor(pois, start)
    return _two_opt(nn, start)


# ============ 顶层封装 ============

def optimize(
    pois: List[POIInfo],
    n_days: int,
    max_per_day: int = 3,
    initial_start: Optional[Location] = None,
) -> List[List[POIInfo]]:
    """
    把候选景点切成 n_days 天的有序行程。

    Args:
        pois: 候选景点（评分排序后的 top N，来自 poi_aggregator）
        n_days: 旅行天数
        max_per_day: 每天最多安排几个景点
        initial_start: 第 1 天起点（如出发地酒店）。后续每天起点用前一天最后一个景点

    Returns:
        长度 == n_days 的列表，每项是当日已排序的景点列表
    """
    if n_days <= 0:
        return []
    if not pois:
        return [[] for _ in range(n_days)]

    # 截断 pois 数量到 n_days * max_per_day（保证每天填满即可）
    capacity = n_days * max_per_day
    pois_use = pois[:capacity]

    buckets = cluster_by_day(pois_use, n_days)

    # 每天截断到 max_per_day（k-means 不保证每簇大小均匀）
    # 多余的点滚到下一天的桶里，最后再排序
    overflow: List[POIInfo] = []
    for i in range(n_days):
        if len(buckets[i]) > max_per_day:
            overflow.extend(buckets[i][max_per_day:])
            buckets[i] = buckets[i][:max_per_day]
    # 把溢出补到尚未填满的桶里
    for poi in overflow:
        for i in range(n_days):
            if len(buckets[i]) < max_per_day:
                buckets[i].append(poi)
                break

    # 按质心经度排序，让"行程方向"相对稳定（西→东，避免随机）
    indexed = list(enumerate(buckets))
    indexed.sort(key=lambda x: _centroid(x[1]).longitude if x[1] else 0.0)
    ordered_buckets = [b for _, b in indexed]

    # 每天排序
    ordered: List[List[POIInfo]] = []
    cursor = initial_start
    for day_pois in ordered_buckets:
        day_sorted = order_within_day(day_pois, cursor)
        ordered.append(day_sorted)
        if day_sorted:
            cursor = day_sorted[-1].location  # 第二天起点 = 前一天结束点
    return ordered


# ============ 调试辅助 ============

def total_distance_per_day(days: List[List[POIInfo]], starts: Optional[List[Location]] = None) -> List[float]:
    """计算每天的总路径长度（km），用于回归对比。"""
    out = []
    for i, day in enumerate(days):
        start = starts[i] if starts and i < len(starts) else None
        out.append(round(_path_length(day, start), 2))
    return out

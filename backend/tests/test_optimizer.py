"""itinerary_optimizer 单元测试。

覆盖：
- haversine 距离正确性
- cluster_by_day 把地理上分散的点正确分组
- order_within_day 的总路径不长于原输入顺序（2-opt 不退化）
- optimize 端到端：每天景点数 ≤ max_per_day，3 天行程同天距离合理
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.schemas import Location, POIInfo  # noqa: E402
from app.services.itinerary_optimizer import (  # noqa: E402
    _path_length,
    cluster_by_day,
    estimate_commute,
    haversine,
    optimize,
    order_within_day,
)


def _poi(id_: str, name: str, lng: float, lat: float, rating: float = 4.5) -> POIInfo:
    return POIInfo(
        id=id_,
        name=name,
        type="风景名胜;旅游景点",
        address="测试地址",
        location=Location(longitude=lng, latitude=lat),
        rating=rating,
        biz_type="风景名胜",
    )


# ============ haversine ============

def test_haversine_zero():
    a = Location(longitude=116.4, latitude=39.9)
    assert haversine(a, a) < 0.001


def test_haversine_known_distance():
    # 北京天安门 → 上海外滩，约 1067km
    beijing = Location(longitude=116.397, latitude=39.909)
    shanghai = Location(longitude=121.490, latitude=31.240)
    d = haversine(beijing, shanghai)
    assert 1000 < d < 1100


def test_haversine_short():
    # 北京内 1km 左右
    a = Location(longitude=116.397, latitude=39.909)
    b = Location(longitude=116.397, latitude=39.918)  # 同经度，纬度差 0.009
    d = haversine(a, b)
    assert 0.9 < d < 1.1


# ============ estimate_commute ============

def test_estimate_commute_driving_vs_walking():
    a = Location(longitude=116.397, latitude=39.909)
    b = Location(longitude=116.450, latitude=39.940)
    drive = estimate_commute(a, b, "driving")
    walk = estimate_commute(a, b, "walking")
    assert drive["distance_km"] == walk["distance_km"]
    assert drive["duration_min"] < walk["duration_min"]  # 开车更快


# ============ cluster_by_day ============

def test_cluster_separates_geographic_groups():
    """3 个明显的地理团：东、西、北。聚类应分开。"""
    east = [_poi(f"e{i}", f"东{i}", 116.50 + i * 0.005, 39.91) for i in range(4)]
    west = [_poi(f"w{i}", f"西{i}", 116.27 + i * 0.005, 39.99) for i in range(4)]
    north = [_poi(f"n{i}", f"北{i}", 116.40, 40.05 + i * 0.005) for i in range(4)]
    pois = east + west + north

    buckets = cluster_by_day(pois, n_days=3)
    assert len(buckets) == 3
    # 每个 bucket 都非空，每个 bucket 内部 IDs 应来自同一团
    bucket_origins = []
    for b in buckets:
        assert len(b) > 0
        prefixes = {p.id[0] for p in b}
        assert len(prefixes) == 1, f"簇内来源不纯: {[p.id for p in b]}"
        bucket_origins.append(prefixes.pop())
    assert sorted(bucket_origins) == ["e", "n", "w"]


def test_cluster_handles_fewer_pois_than_days():
    pois = [_poi("a", "A", 116.4, 39.9)]
    buckets = cluster_by_day(pois, n_days=3)
    assert len(buckets) == 3
    assert sum(len(b) for b in buckets) == 1


def test_cluster_empty_input():
    buckets = cluster_by_day([], n_days=2)
    assert buckets == [[], []]


# ============ order_within_day ============

def test_order_within_day_no_worse_than_input():
    """2-opt 不应让总路径变长。"""
    pois = [
        _poi("1", "A", 116.40, 39.90),
        _poi("2", "B", 116.45, 39.95),
        _poi("3", "C", 116.42, 39.91),
        _poi("4", "D", 116.50, 39.93),
        _poi("5", "E", 116.41, 39.94),
    ]
    start = Location(longitude=116.39, latitude=39.89)
    original_len = _path_length(pois, start)
    sorted_pois = order_within_day(pois, start)
    sorted_len = _path_length(sorted_pois, start)
    assert sorted_len <= original_len + 1e-6
    assert set(p.id for p in sorted_pois) == set(p.id for p in pois)  # 不丢点


def test_order_within_day_empty():
    assert order_within_day([], None) == []


# ============ optimize 端到端 ============

def test_optimize_three_days_beijing():
    """北京 12 个景点，3 天，每天 4 个；同天景点应地理临近。"""
    pois = [
        _poi("p01", "故宫", 116.397, 39.918, 4.8),
        _poi("p02", "天安门", 116.397, 39.909, 4.8),
        _poi("p03", "天坛", 116.412, 39.882, 4.7),
        _poi("p04", "前门", 116.397, 39.899, 4.5),
        _poi("p05", "颐和园", 116.265, 39.999, 4.7),
        _poi("p06", "圆明园", 116.302, 40.008, 4.5),
        _poi("p07", "北大", 116.310, 39.988, 4.6),
        _poi("p08", "清华", 116.326, 40.000, 4.6),
        _poi("p09", "鸟巢", 116.397, 39.992, 4.6),
        _poi("p10", "水立方", 116.391, 39.992, 4.5),
        _poi("p11", "奥林匹克公园", 116.395, 40.005, 4.5),
        _poi("p12", "雍和宫", 116.417, 39.948, 4.7),
    ]
    days = optimize(pois, n_days=3, max_per_day=4)
    assert len(days) == 3
    # 每天景点数 ≤ 4
    for day in days:
        assert len(day) <= 4
    # 总景点数 == 12
    total = sum(len(d) for d in days)
    assert total == 12
    # 同天景点两两距离合理（北京内同片区一般 < 12km）
    for day in days:
        for i in range(len(day)):
            for j in range(i + 1, len(day)):
                d = haversine(day[i].location, day[j].location)
                assert d < 15, f"同天景点过远: {day[i].name}↔{day[j].name} = {d:.1f}km"


def test_optimize_total_path_better_than_input_order():
    """聚类+TSP 后，3 天总路径应明显短于按输入顺序硬切 3 段。"""
    pois = [
        # 故意打乱：东→西→北→东→...
        _poi("p1", "故宫", 116.397, 39.918),
        _poi("p2", "颐和园", 116.265, 39.999),
        _poi("p3", "鸟巢", 116.397, 39.992),
        _poi("p4", "天坛", 116.412, 39.882),
        _poi("p5", "圆明园", 116.302, 40.008),
        _poi("p6", "雍和宫", 116.417, 39.948),
        _poi("p7", "天安门", 116.397, 39.909),
        _poi("p8", "北大", 116.310, 39.988),
        _poi("p9", "前门", 116.397, 39.899),
    ]

    # 基线：按输入顺序硬切成 3 段
    baseline_total = sum(_path_length(pois[i:i + 3], None) for i in (0, 3, 6))

    # 优化结果
    days = optimize(pois, n_days=3, max_per_day=3)
    optimized_total = sum(_path_length(d, None) for d in days)

    # 总路径应显著缩短（至少 20%）
    assert optimized_total < baseline_total * 0.8, (
        f"优化路径 {optimized_total:.2f}km 未明显短于基线 {baseline_total:.2f}km"
    )

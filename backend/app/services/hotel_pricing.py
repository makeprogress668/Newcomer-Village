"""酒店价格估算 + 跳转链接生成。

由于项目未接第三方真实订房 API，价格不能取真实值。这里改为按"档位 × 城市系数"估算
区间，并明确标注 price_source='estimated'，前端展示"参考价"角标。
预订动作通过携程跳转链接交给用户去比价。
"""

from typing import Tuple
from urllib.parse import quote

from ..data.keywords import infer_hotel_tier


# 档位 → (低位, 高位) 元/晚
TIER_PRICE: dict = {
    "经济型": (200, 400),
    "舒适型": (400, 800),
    "豪华型": (800, 2000),
    "民宿": (150, 500),
}

# 一线/新一线城市价格系数
CITY_COEF: dict = {
    "北京": 1.3,
    "上海": 1.4,
    "深圳": 1.3,
    "广州": 1.2,
    "杭州": 1.2,
    "南京": 1.1,
    "成都": 1.1,
    "苏州": 1.1,
    "三亚": 1.5,  # 旅游城市淡旺季差大，先按高位
    "丽江": 1.2,
}


# 用户输入偏好 → 标准档位 key
_PREF_ALIASES: dict = {
    "经济型酒店": "经济型",
    "经济酒店": "经济型",
    "经济": "经济型",
    "快捷酒店": "经济型",
    "舒适型酒店": "舒适型",
    "舒适": "舒适型",
    "三星": "舒适型",
    "三星级": "舒适型",
    "四星": "舒适型",
    "四星级": "舒适型",
    "豪华型酒店": "豪华型",
    "豪华": "豪华型",
    "五星": "豪华型",
    "五星级": "豪华型",
    "高端": "豪华型",
    "客栈": "民宿",
    "青旅": "民宿",
}


def normalize_tier(pref: str) -> str:
    """把用户输入的住宿偏好规范成 TIER_PRICE 的 key。"""
    if not pref:
        return "舒适型"
    if pref in TIER_PRICE:
        return pref
    return _PREF_ALIASES.get(pref, "舒适型")


def estimate_price_range(tier: str, city: str) -> Tuple[int, int]:
    """返回某档位酒店在某城市的预估价格区间（整数元）。"""
    tier = normalize_tier(tier)
    base = TIER_PRICE[tier]
    coef = CITY_COEF.get(city, 1.0)
    return (int(base[0] * coef), int(base[1] * coef))


def format_price_range(low: int, high: int) -> str:
    """格式化为前端展示文本。"""
    return f"{low}-{high}元/晚"


def build_booking_url(hotel_name: str, city: str) -> str:
    """携程酒店搜索跳转。让用户在真实平台查看实时价格 / 下单。"""
    keyword = quote(hotel_name or "酒店")
    city_q = quote(city or "")
    return f"https://hotels.ctrip.com/hotels/list?city={city_q}&keyword={keyword}"


def estimate_hotel(
    hotel_name: str,
    city: str,
    accommodation_pref: str = "舒适型",
) -> dict:
    """
    一站式：根据酒店名 + 城市 + 用户偏好，得到估算价格 + 携程链接。

    Returns:
        {
            "tier": str,
            "low": int,
            "high": int,
            "price_range": str,
            "estimated_cost": int,  # 区间中位数
            "price_source": "estimated",
            "booking_url": str,
        }
    """
    fallback = normalize_tier(accommodation_pref)
    tier = infer_hotel_tier(hotel_name, fallback_pref=fallback)
    low, high = estimate_price_range(tier, city)
    return {
        "tier": tier,
        "low": low,
        "high": high,
        "price_range": format_price_range(low, high),
        "estimated_cost": (low + high) // 2,
        "price_source": "estimated",
        "booking_url": build_booking_url(hotel_name, city),
    }

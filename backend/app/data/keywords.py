"""偏好 / 酒店关键词常量映射。

这里集中维护"用户偏好 → 高德 POI 搜索关键词"的派发规则，便于后续扩展。
偏好命中后会派发若干关键词到高德 maps_text_search，景点结果合并去重。
"""

from typing import Dict, List, Optional, Tuple


# ============ 景点关键词派发 ============

# 用户偏好（在 TripRequest.preferences 里勾选）→ 高德搜索关键词候选
PREFERENCE_KEYWORDS: Dict[str, List[str]] = {
    # P20: 每个偏好仅保留 1-2 个最具代表性的关键词,减少 API 调用
    "历史文化": ["博物馆", "古迹"],
    "美食": ["美食街"],
    "自然": ["公园"],
    "购物": ["商业街"],
    "艺术": ["美术馆"],
    "亲子": ["主题公园"],
    "夜景": ["观景台"],
    "宗教": ["寺庙"],
    "运动": ["体育公园"],
    "摄影": ["地标"],
}

# 任何偏好都附加的保底关键词,只保留最关键的 1 个 (避免重复命中)
FALLBACK_KEYWORDS: List[str] = ["5A级景区"]


# 明显非景点的高德 POI type 前缀（用于过滤搜索结果）
NON_ATTRACTION_TYPES: Tuple[str, ...] = (
    "住宿服务",
    "餐饮服务",
    "购物服务",  # 购物服务整体过滤；纯购物偏好已通过关键词派发兜底
    "公司企业",
    "金融保险服务",
    "汽车服务",
    "汽车销售",
    "汽车维修",
    "生活服务",
    "医疗保健服务",
    "政府机构",
    "科教文化服务;学校",
    "交通设施服务",
)


def expand_preferences(preferences: List[str]) -> List[str]:
    """把偏好列表展开为关键词列表，并附加保底词。返回去重后的关键词。"""
    out: List[str] = []
    seen: set = set()
    for pref in preferences or []:
        for kw in PREFERENCE_KEYWORDS.get(pref, [pref]):  # 未识别偏好直接当作关键词使用
            if kw and kw not in seen:
                out.append(kw)
                seen.add(kw)
    for kw in FALLBACK_KEYWORDS:
        if kw not in seen:
            out.append(kw)
            seen.add(kw)
    return out


# ============ 用户自由输入文本 → 额外景点关键词 ============

# 用户在 free_text_input 里提到这些触发词时，自动补充对应景点关键词
FREE_TEXT_TRIGGERS: Dict[str, List[str]] = {
    "升旗": ["天安门", "天安门广场"],
    "升旗仪式": ["天安门广场", "天安门"],
    "故宫": ["故宫博物院"],
    "长城": ["八达岭长城", "慕田峪长城"],
    "颐和园": ["颐和园"],
    "天坛": ["天坛公园"],
    "鸟巢": ["国家体育场", "鸟巢"],
    "水立方": ["国家游泳中心"],
    "什刹海": ["什刹海"],
    "南锣鼓巷": ["南锣鼓巷"],
    "雍和宫": ["雍和宫"],
    "外滩": ["外滩"],
    "东方明珠": ["东方明珠"],
    "迪士尼": ["上海迪士尼乐园"],
    "西湖": ["西湖", "断桥"],
    "兵马俑": ["秦始皇兵马俑博物馆"],
    "大雁塔": ["大雁塔"],
    "黄山": ["黄山风景区"],
    "鼓浪屿": ["鼓浪屿"],
    "博物馆": ["博物馆"],
    "夜景": ["观景台"],
    "购物": ["商业街", "步行街"],
    "美食": ["小吃街", "美食街"],
}


# ============ 城市必去保底地标 (无论用户偏好如何,每次都会强制纳入候选池) ============

# 设计哲学: 用户来某城市旅游, 哪怕选了"艺术"或"美食"偏好, 也很大概率
# 不愿错过该城市的"超级地标"。这里维护每个城市的 top 3-4 必去清单,
# 在 collect_attractions 时作为 must_include 强制纳入,确保"知名地标命中率"指标
# 始终有保底分。
CITY_MUST_VISIT: Dict[str, List[str]] = {
    "北京": ["故宫博物院", "天安门广场", "颐和园"],
    "上海": ["外滩", "东方明珠", "豫园"],
    "西安": ["秦始皇兵马俑博物馆", "大雁塔", "钟楼"],
    "杭州": ["西湖", "灵隐寺"],
    "成都": ["宽窄巷子", "锦里", "成都大熊猫繁育研究基地"],
    "南京": ["中山陵", "夫子庙", "南京博物院"],
    "苏州": ["拙政园", "留园", "虎丘"],
    "厦门": ["鼓浪屿", "厦门大学"],
    "三亚": ["亚龙湾", "天涯海角"],
    "广州": ["广州塔", "陈家祠"],
    "深圳": ["世界之窗", "深圳湾公园"],
    "重庆": ["洪崖洞", "解放碑", "磁器口"],
    "天津": ["五大道", "天津之眼"],
    "青岛": ["栈桥", "崂山"],
    "武汉": ["黄鹤楼", "东湖"],
    "长沙": ["岳麓山", "橘子洲"],
}


def get_must_visit(city: str) -> List[str]:
    """获取城市必去保底地标列表 (空字符串/未知城市返回空)。"""
    return list(CITY_MUST_VISIT.get(city or "", []))


# 热门地标扩展表: 用于排序加权,不强制每次都全部安排。
CITY_LANDMARKS: Dict[str, List[str]] = {
    "北京": [
        "故宫", "天安门", "颐和园", "长城", "八达岭", "慕田峪", "天坛", "圆明园",
        "国家博物馆", "国家自然博物馆", "景山", "北海", "什刹海", "南锣鼓巷",
        "雍和宫", "鸟巢", "水立方", "前门", "王府井", "798",
    ],
    "上海": [
        "外滩", "东方明珠", "豫园", "城隍庙", "南京路", "陆家嘴", "上海中心",
        "上海博物馆", "上海科技馆", "迪士尼", "新天地", "田子坊", "武康大楼",
        "朱家角", "浦东美术馆", "西岸美术馆",
    ],
    "西安": [
        "兵马俑", "秦始皇", "大雁塔", "小雁塔", "钟楼", "鼓楼", "城墙",
        "华清宫", "陕西历史博物馆", "大唐不夜城", "回民街",
    ],
    "杭州": ["西湖", "灵隐寺", "雷峰塔", "断桥", "苏堤", "西溪", "宋城", "河坊街"],
    "成都": ["熊猫", "宽窄巷子", "锦里", "武侯祠", "杜甫草堂", "春熙路", "太古里", "都江堰", "青城山"],
    "南京": ["中山陵", "夫子庙", "玄武湖", "明孝陵", "总统府", "南京博物院", "老门东"],
    "苏州": ["拙政园", "留园", "狮子林", "虎丘", "金鸡湖", "山塘街", "平江路", "苏州博物馆"],
    "厦门": ["鼓浪屿", "南普陀", "曾厝垵", "环岛路", "厦门大学", "中山路"],
    "三亚": ["亚龙湾", "天涯海角", "南山", "蜈支洲岛", "鹿回头", "三亚湾"],
    "广州": ["广州塔", "白云山", "陈家祠", "沙面", "北京路", "长隆", "越秀公园"],
    "深圳": ["世界之窗", "欢乐谷", "深圳湾", "莲花山", "梧桐山", "大梅沙"],
    "重庆": ["洪崖洞", "解放碑", "磁器口", "李子坝", "长江索道", "南山一棵树"],
}


def landmark_priority(city: str, poi_name: str) -> float:
    """返回热门度加权分。必去地标强加权,普通热门地标中等加权。"""
    if not poi_name:
        return 0.0
    score = 0.0
    for idx, kw in enumerate(CITY_MUST_VISIT.get(city or "", [])):
        if kw in poi_name or poi_name in kw:
            score = max(score, 6.0 - idx * 0.5)
    for kw in CITY_LANDMARKS.get(city or "", []):
        if kw in poi_name or poi_name in kw:
            score = max(score, 3.0)
    return score


# ============ 城市质心坐标 (用于跨城 POI 过滤) ============
# 防止 geocode 把"外滩"等同名地标误命中其他城市 (如香港的同名地址)。
# 维度: (longitude, latitude) — 与 amap 一致
CITY_CENTERS: Dict[str, tuple] = {
    "北京": (116.397, 39.916),
    "上海": (121.473, 31.230),
    "广州": (113.264, 23.129),
    "深圳": (114.058, 22.543),
    "杭州": (120.155, 30.275),
    "南京": (118.796, 32.060),
    "苏州": (120.620, 31.298),
    "成都": (104.066, 30.572),
    "重庆": (106.551, 29.563),
    "西安": (108.940, 34.341),
    "武汉": (114.305, 30.593),
    "天津": (117.190, 39.125),
    "青岛": (120.382, 36.066),
    "厦门": (118.089, 24.479),
    "三亚": (109.508, 18.247),
    "长沙": (112.982, 28.194),
    "丽江": (100.227, 26.872),
    "桂林": (110.290, 25.273),
    "黄山": (118.331, 29.734),
    "九江": (115.992, 29.712),
    "洛阳": (112.434, 34.663),
    "敦煌": (94.661, 40.142),
    "拉萨": (91.140, 29.645),
}


def get_city_center(city: str) -> "tuple | None":
    """返回 (lng, lat) 元组,未知城市返回 None。"""
    return CITY_CENTERS.get(city or "")


def is_within_city(longitude: float, latitude: float, city: str, max_km: float = 100.0) -> bool:
    """判断坐标是否在城市的合理范围内 (默认半径 100km)。
    未知城市直接放行 (避免误杀)。
    """
    center = CITY_CENTERS.get(city or "")
    if not center:
        return True
    # Haversine 距离 (km)
    import math
    R = 6371.0
    lat1 = math.radians(center[1])
    lat2 = math.radians(latitude)
    dlat = lat2 - lat1
    dlng = math.radians(longitude - center[0])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    distance = 2 * R * math.asin(math.sqrt(h))
    return distance <= max_km


def expand_free_text(free_text: Optional[str]) -> List[str]:
    """从用户自由输入文本中抽取额外景点搜索关键词。"""
    if not free_text:
        return []
    out: List[str] = []
    seen: set = set()
    for trigger, kws in FREE_TEXT_TRIGGERS.items():
        if trigger in free_text:
            for kw in kws:
                if kw not in seen:
                    out.append(kw)
                    seen.add(kw)
    return out


# ============ 中文景点名 → 英文搜索词（供 Unsplash 图片搜索使用）============

# 中文景点在 Unsplash 上几乎搜不到，必须用英文专名才有相关图
ATTRACTION_NAME_EN: Dict[str, str] = {
    # 北京
    "故宫": "Forbidden City Beijing",
    "故宫博物院": "Forbidden City Beijing",
    "天安门": "Tiananmen Square Beijing",
    "天安门广场": "Tiananmen Square Beijing",
    "颐和园": "Summer Palace Beijing",
    "圆明园": "Old Summer Palace Beijing",
    "天坛": "Temple of Heaven Beijing",
    "天坛公园": "Temple of Heaven Beijing",
    "长城": "Great Wall of China",
    "八达岭长城": "Badaling Great Wall",
    "慕田峪长城": "Mutianyu Great Wall",
    "司马台长城": "Simatai Great Wall",
    "金山岭长城": "Jinshanling Great Wall",
    "鸟巢": "Beijing National Stadium",
    "国家体育场": "Beijing National Stadium",
    "水立方": "Beijing Water Cube",
    "国家游泳中心": "Beijing Water Cube",
    "什刹海": "Shichahai Beijing",
    "南锣鼓巷": "Nanluoguxiang Beijing hutong",
    "雍和宫": "Yonghe Temple Beijing",
    "中国国家博物馆": "National Museum of China Beijing",
    "国家自然博物馆": "Beijing Natural History Museum",
    "北京古代建筑博物馆": "Beijing Ancient Architecture Museum",
    "景山公园": "Jingshan Park Beijing",
    "北海公园": "Beihai Park Beijing",
    # 上海
    "外滩": "The Bund Shanghai",
    "东方明珠": "Oriental Pearl Tower Shanghai",
    "豫园": "Yu Garden Shanghai",
    "上海迪士尼乐园": "Shanghai Disneyland",
    "南京路": "Nanjing Road Shanghai",
    # 西安
    "兵马俑": "Terracotta Army Xi'an",
    "秦始皇兵马俑博物馆": "Terracotta Army Xi'an",
    "大雁塔": "Big Wild Goose Pagoda Xi'an",
    "钟楼": "Bell Tower Xi'an",
    # 杭州
    "西湖": "West Lake Hangzhou",
    "雷峰塔": "Leifeng Pagoda Hangzhou",
    "断桥": "Broken Bridge West Lake",
    "灵隐寺": "Lingyin Temple Hangzhou",
    # 苏州
    "拙政园": "Humble Administrator's Garden Suzhou",
    "留园": "Lingering Garden Suzhou",
    # 黄山
    "黄山风景区": "Huangshan Mountain China",
    "黄山": "Huangshan Mountain China",
    # 厦门
    "鼓浪屿": "Gulangyu Island Xiamen",
    # 广州
    "广州塔": "Canton Tower Guangzhou",
    # 深圳
    "世界之窗": "Window of the World Shenzhen",
    # 三亚
    "亚龙湾": "Yalong Bay Sanya",
    # 桂林
    "漓江": "Li River Guilin",
}


def to_english_query(name: str, city: str = "") -> str:
    """把中文景点名映射为英文搜索词，命中映射用专名;否则去掉中文 city 用 name+landmark。"""
    if not name:
        return ""
    # 精确命中
    if name in ATTRACTION_NAME_EN:
        return ATTRACTION_NAME_EN[name]
    # 包含子串匹配（处理"故宫博物院-午门"这种带子点位的名字）
    for key, en in ATTRACTION_NAME_EN.items():
        if key in name:
            return en
    # 兜底：景点名 + 城市英文（不准确但比中文好）
    return f"{name} {city} landmark".strip()


# ============ 酒店名称黑名单（明显非正经酒店）============

HOTEL_NAME_BLACKLIST = (
    "驿站",     # 经济连锁里的廉价分支，往往是民居改造
    "招待所",
    "民居",
    "公寓出租",
    "短租",
    "日租",
    "旅店",     # 通常是单体小旅馆
    "客房",
    "农家",
    "鸡毛",
    "出租房",
)


def is_blacklisted_hotel(name: str) -> bool:
    """酒店名包含黑名单关键词则视为非正经酒店。"""
    if not name:
        return False
    for kw in HOTEL_NAME_BLACKLIST:
        if kw in name:
            return True
    return False


def is_non_attraction(poi_type: str) -> bool:
    """根据高德 type 判断该 POI 是否明显不是景点。"""
    if not poi_type:
        return False
    for prefix in NON_ATTRACTION_TYPES:
        if poi_type.startswith(prefix):
            return True
    return False


# ============ 酒店档位识别 ============

# 关键词 → 酒店档位。用于从酒店名字反推档位（覆盖用户偏好）。
HOTEL_TIER_KEYWORDS: Dict[str, List[str]] = {
    "经济型": [
        "如家", "汉庭", "7天", "锦江之星", "莫泰", "速8", "格林豪泰",
        "尚客优", "布丁", "城市便捷", "宜必思", "海友", "派酒店",
    ],
    "舒适型": [
        "亚朵", "全季", "桔子", "维也纳", "麗枫", "希岸", "怡莱", "美居",
        "智选假日", "漫心", "锦江都城", "丽柏", "丽枫", "和颐", "美程",
        "希尔顿欢朋", "万怡", "假日", "万枫", "凯悦嘉轩", "亚朵S",
    ],
    "豪华型": [
        "君悦", "丽思", "万豪", "希尔顿", "凯悦", "四季", "香格里拉", "半岛",
        "瑞吉", "文华东方", "洲际", "威斯汀", "JW", "费尔蒙", "丽晶",
        "嘉里", "宝格丽", "柏悦", "安达仕", "君悦大酒店", "豪华精选",
    ],
    "民宿": ["民宿", "客栈", "青年旅舍", "Airbnb", "B&B", "四合院"],
}


def infer_hotel_tier(hotel_name: str, fallback_pref: str = "舒适型") -> str:
    """根据酒店名字反推档位，命中关键词的档位优先；都没命中则回落到用户偏好。"""
    if not hotel_name:
        return fallback_pref
    name_lower = hotel_name.lower()
    for tier, kws in HOTEL_TIER_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in name_lower:
                return tier
    return fallback_pref or "舒适型"

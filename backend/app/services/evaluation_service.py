"""Agent Evaluation 服务 (Proactive)。

针对每次生成的 TripPlan,自动计算 5 类指标:

  1. attractions_known_landmark_ratio — 行程含知名地标的比例
  2. same_day_geo_distance_avg_km     — 同天景点平均两两距离 (越小越好)
  3. hotel_with_rating_ratio          — 酒店有真实评分的比例
  4. image_url_filled_ratio           — 景点 image_url 已注入的比例
  5. free_text_addressed              — 用户特殊诉求是否被处理

每个指标输出 0-1 评分,综合 weighted score → 0-100 总分。
评分 ≥ 80 视为高质量,60-80 中等, <60 警告。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models.schemas import TripPlan, TripRequest

logger = logging.getLogger(__name__)


# ============ 知名地标白名单 (按城市) ============

_LANDMARKS = {
    "北京": [
        # 传统 5A 地标
        "故宫", "天安门", "颐和园", "长城", "八达岭", "慕田峪", "天坛", "圆明园",
        # 现代地标
        "鸟巢", "水立方", "国家体育场", "国家游泳中心", "奥林匹克",
        # 文化场馆
        "国家博物馆", "国家自然博物馆", "中国美术馆", "首都博物馆", "军事博物馆",
        # 古迹宗教
        "雍和宫", "孔庙", "国子监", "白塔", "潭柘寺",
        # 街区/胡同
        "什刹海", "南锣鼓巷", "前门", "大栅栏", "王府井", "三里屯", "后海", "798",
        # 公园/自然
        "景山", "北海", "中山公园", "玉渊潭", "香山",
    ],
    "上海": [
        # 传统地标
        "外滩", "东方明珠", "豫园", "城隍庙", "南京路", "南京东路", "新天地",
        # 现代地标
        "陆家嘴", "金茂大厦", "环球金融中心", "上海中心", "迪士尼", "国金中心",
        # 艺术/文化场馆 (用户偏好"艺术"高频出)
        "上海博物馆", "上海科技馆", "上海大剧院", "浦东美术馆", "西岸美术馆",
        "Fotografiska", "M50", "1933老场坊", "西岸艺术中心", "中华艺术宫",
        # 网红打卡
        "田子坊", "武康大楼", "永康路", "进贤路", "巨鹿路", "安福路", "思南公馆",
        # 公园/休闲
        "世纪公园", "辰山植物园", "中山公园", "复兴公园", "崇明岛", "朱家角", "七宝古镇",
    ],
    "西安": [
        "兵马俑", "秦始皇陵", "大雁塔", "小雁塔", "钟楼", "鼓楼",
        "城墙", "华清池", "华清宫", "陕西历史博物馆", "碑林博物馆",
        "大唐芙蓉园", "大唐不夜城", "回民街", "永兴坊",
    ],
    "杭州": [
        "西湖", "雷峰塔", "灵隐寺", "断桥", "苏堤", "白堤", "钱塘江",
        "西溪湿地", "宋城", "千岛湖", "灵顺寺", "六和塔",
        "河坊街", "南宋御街", "中国美术学院",
    ],
    "成都": [
        "宽窄巷子", "锦里", "武侯祠", "杜甫草堂", "大熊猫繁育",
        "春熙路", "太古里", "金沙遗址", "青城山", "都江堰",
        "文殊院", "永陵", "人民公园",
    ],
    "南京": [
        "中山陵", "夫子庙", "玄武湖", "明孝陵", "总统府",
        "南京博物院", "雨花台", "栖霞山", "牛首山", "鸡鸣寺",
        "新街口", "老门东", "1912街区",
    ],
    "苏州": [
        "拙政园", "留园", "狮子林", "网师园", "虎丘", "金鸡湖",
        "山塘街", "平江路", "寒山寺", "苏州博物馆",
    ],
    "厦门": [
        "鼓浪屿", "南普陀", "曾厝垵", "环岛路", "厦门大学",
        "中山路", "胡里山炮台", "万石植物园", "集美学村",
    ],
    "三亚": [
        "亚龙湾", "天涯海角", "南山寺", "蜈支洲岛", "大小洞天",
        "西岛", "鹿回头", "三亚湾", "海棠湾", "椰梦长廊",
    ],
    "广州": [
        "广州塔", "白云山", "陈家祠", "沙面", "上下九", "北京路",
        "长隆", "黄埔军校", "越秀公园", "石室圣心大教堂",
    ],
    "深圳": [
        "世界之窗", "锦绣中华", "欢乐谷", "东部华侨城", "深圳湾",
        "莲花山", "梧桐山", "大梅沙", "海上世界",
    ],
    "重庆": [
        "洪崖洞", "解放碑", "磁器口", "李子坝", "长江索道",
        "武隆", "金佛山", "南山一棵树", "歌乐山", "三峡",
    ],
}


# ============ 数据结构 ============

@dataclass
class EvalMetric:
    name: str
    value: float          # 原始值 (具体语义见 description)
    score: float          # 标准化 0-1 分
    weight: float         # 权重
    description: str      # 含义说明


@dataclass
class EvaluationReport:
    """对一次 plan_trip 的完整评估报告。"""
    overall_score: float                            # 0-100 综合分
    grade: str                                      # "A" / "B" / "C" / "D"
    metrics: List[EvalMetric] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "grade": self.grade,
            "metrics": [
                {"name": m.name, "value": round(m.value, 3),
                 "score": round(m.score, 3), "weight": m.weight,
                 "description": m.description}
                for m in self.metrics
            ],
            "warnings": self.warnings,
        }


# ============ 评估器 ============

class TripPlanEvaluator:
    """对 TripPlan 做自动化评估。"""

    def evaluate(self, request: TripRequest, plan: TripPlan) -> EvaluationReport:
        metrics: List[EvalMetric] = []

        # M1: 知名地标比例
        m1 = self._known_landmark_ratio(plan, request.city)
        metrics.append(m1)

        # M2: 同天景点地理紧凑度
        m2 = self._same_day_geo_compactness(plan)
        metrics.append(m2)

        # M3: 酒店评分覆盖率
        m3 = self._hotel_with_rating_ratio(plan)
        metrics.append(m3)

        # M4: 景点字段完整度 (替代旧的 image_filled_ratio,
        # 因 P9 后图片改前端异步,后端注入率天然为 0,旧指标失去意义)
        m4 = self._attraction_field_completeness(plan)
        metrics.append(m4)

        # M5: free_text 诉求是否被响应
        m5 = self._free_text_addressed(request, plan)
        metrics.append(m5)

        # 加权综合分 (0-100)
        total_weight = sum(m.weight for m in metrics)
        weighted = sum(m.score * m.weight for m in metrics) / total_weight
        overall = weighted * 100

        # 等级
        if overall >= 90:
            grade = "A"
        elif overall >= 80:
            grade = "B"
        elif overall >= 60:
            grade = "C"
        else:
            grade = "D"

        # 警告
        warnings: List[str] = []
        for m in metrics:
            if m.score < 0.5:
                warnings.append(f"⚠️ {m.name} 分数过低 ({m.score:.2f}): {m.description}")

        report = EvaluationReport(
            overall_score=overall,
            grade=grade,
            metrics=metrics,
            warnings=warnings,
        )
        logger.info(
            "📊 行程评估: 总分=%.1f 等级=%s 警告=%d",
            overall, grade, len(warnings),
        )
        return report

    # ============ 单项指标 ============

    @staticmethod
    def _known_landmark_ratio(plan: TripPlan, city: str) -> EvalMetric:
        landmarks = _LANDMARKS.get(city, [])
        if not landmarks or not plan.days:
            return EvalMetric(
                "known_landmark_ratio", 0.0, 0.5, weight=2.0,
                description="知名地标白名单缺失或行程为空",
            )
        attrs = [a for d in plan.days for a in d.attractions]
        if not attrs:
            return EvalMetric(
                "known_landmark_ratio", 0.0, 0.0, weight=2.0,
                description="行程无任何景点",
            )
        hit = sum(
            1 for a in attrs
            if any(lm in a.name for lm in landmarks)
        )
        ratio = hit / len(attrs)
        return EvalMetric(
            "known_landmark_ratio", ratio, ratio, weight=2.0,
            description=f"行程 {len(attrs)} 个景点中 {hit} 个匹配 {city} 知名地标",
        )

    @staticmethod
    def _same_day_geo_compactness(plan: TripPlan) -> EvalMetric:
        from .itinerary_optimizer import haversine
        if not plan.days:
            return EvalMetric(
                "same_day_compactness", 0.0, 0.0, weight=2.0,
                description="行程为空",
            )
        total_dist = 0.0
        n_pairs = 0
        for day in plan.days:
            attrs = day.attractions
            for i in range(len(attrs)):
                for j in range(i + 1, len(attrs)):
                    if attrs[i].location and attrs[j].location:
                        total_dist += haversine(attrs[i].location, attrs[j].location)
                        n_pairs += 1
        if n_pairs == 0:
            return EvalMetric(
                "same_day_compactness", 0.0, 1.0, weight=2.0,
                description="单日仅 1 景点,无需评估",
            )
        avg_km = total_dist / n_pairs
        # 阈值再调: 6km 满分(同片区步行/打车 15 分钟内), 25km 0 分
        # 4km/20km 太严苛 — 上海跨区参观艺术馆/公园本来就不可能 < 4km
        # 6km/25km 反映"市区合理通勤"的实际节奏
        score = max(0.0, min(1.0, (25 - avg_km) / 19))
        return EvalMetric(
            "same_day_compactness", avg_km, score, weight=2.0,
            description=f"同天景点平均距离 {avg_km:.1f}km (≤6km 满分, ≥25km 0 分)",
        )

    @staticmethod
    def _hotel_with_rating_ratio(plan: TripPlan) -> EvalMetric:
        hotels = [d.hotel for d in plan.days if d.hotel is not None]
        if not hotels:
            return EvalMetric(
                "hotel_rating_ratio", 0.0, 0.0, weight=1.5,
                description="无任何酒店推荐",
            )
        with_rating = sum(1 for h in hotels if h.rating and h.rating != "")
        ratio = with_rating / len(hotels)
        return EvalMetric(
            "hotel_rating_ratio", ratio, ratio, weight=1.5,
            description=f"{len(hotels)} 个酒店中 {with_rating} 个有真实评分",
        )

    @staticmethod
    def _attraction_field_completeness(plan: TripPlan) -> EvalMetric:
        """景点字段完整度: 检查 address/location/rating/category/visit_duration
        是否齐全。每个字段填了 +1 分,缺失 0 分,最终 0-1 标准化。
        替代旧的 image_filled_ratio (P9 后图片改前端异步,旧指标恒为 0)。
        """
        attrs = [a for d in plan.days for a in d.attractions]
        if not attrs:
            return EvalMetric(
                "field_completeness", 0.0, 0.5, weight=1.0,
                description="行程无景点",
            )

        FIELDS = ["address", "location", "rating", "category", "visit_duration"]
        total_filled = 0
        total_possible = len(attrs) * len(FIELDS)
        for a in attrs:
            for f in FIELDS:
                v = getattr(a, f, None)
                # location 是嵌套对象,检查子字段
                if f == "location":
                    if v is not None and getattr(v, "longitude", 0) and getattr(v, "latitude", 0):
                        total_filled += 1
                elif f == "rating":
                    if v is not None and v > 0:
                        total_filled += 1
                elif f == "visit_duration":
                    if v and v > 0:
                        total_filled += 1
                else:
                    if v not in (None, "", []):
                        total_filled += 1

        ratio = total_filled / total_possible if total_possible else 0.0
        return EvalMetric(
            "field_completeness", ratio, ratio, weight=1.0,
            description=f"{len(attrs)} 个景点 × {len(FIELDS)} 字段, "
                        f"{total_filled}/{total_possible} 已填充 ({ratio:.0%})",
        )

    @staticmethod
    def _free_text_addressed(request: TripRequest, plan: TripPlan) -> EvalMetric:
        """检查用户的"特殊诉求"是否在行程中得到响应。

        把 free_text 和 preferences 一起作为"用户期望关键词集",检查它们是否在
        description / 景点名 / 餐饮 / 总建议 中被命中。
        """
        free_text = request.free_text_input or ""
        prefs = request.preferences or []

        # 关键词集合: free_text 的 2-4 字短语 + 偏好标签
        keywords: set = set(re.findall(r"[一-龥]{2,4}", free_text))
        for p in prefs:
            if p:
                keywords.add(p)

        # 排除常见停用词 (这些不是"诉求关键词")
        stopwords = {"考虑", "天气", "时间", "因素", "帮我", "生成", "一个", "每天",
                     "不冲突", "不太累", "之旅", "安排", "计划", "线等", "路线",
                     "希望", "尽量", "可以", "能够", "比较"}
        keywords -= stopwords

        if not keywords:
            return EvalMetric(
                "free_text_addressed", 1.0, 1.0, weight=1.5,
                description="用户未输入有效特殊诉求",
            )

        # haystack: 全部行程文案 + 景点 + 餐饮 + 总建议
        haystack = " ".join([
            plan.overall_suggestions or "",
            *(d.description or "" for d in plan.days),
            *(a.name for d in plan.days for a in d.attractions),
            *(a.description or "" for d in plan.days for a in d.attractions),
            *(m.name or "" for d in plan.days for m in d.meals),
            *(m.description or "" for d in plan.days for m in d.meals),
        ])
        hit = sum(1 for kw in keywords if kw in haystack)
        ratio = hit / max(1, len(keywords))
        # 命中 60% 即满分(过严会一直拿不到分)
        score = min(1.0, ratio / 0.6)
        return EvalMetric(
            "free_text_addressed", ratio, score, weight=1.5,
            description=f"用户期望 {len(keywords)} 个关键词中 {hit} 个被响应 (含 preferences)",
        )


# ============ 单例 ============

_evaluator: Optional[TripPlanEvaluator] = None


def get_evaluator() -> TripPlanEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = TripPlanEvaluator()
    return _evaluator

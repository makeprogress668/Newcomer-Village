"""热门景点兜底数据。

这些数据只在地图 API 没有返回足够可用 POI 时参与排序，用于保证项目在
课堂演示或网络不稳定时仍能生成常识上可落地的行程。线上优先使用高德真实 POI。
"""

from typing import Dict, List, Optional

from ..models.schemas import Location, POIInfo


_CURATED: Dict[str, List[dict]] = {
    "北京": [
        {"id": "curated:beijing:tiananmen", "name": "天安门广场", "address": "北京市东城区东长安街", "lng": 116.3975, "lat": 39.9087, "rating": 4.8, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:beijing:forbidden-city", "name": "故宫博物院", "address": "北京市东城区景山前街4号", "lng": 116.3970, "lat": 39.9175, "rating": 4.8, "level": "AAAAA", "type": "科教文化服务;博物馆"},
        {"id": "curated:beijing:summer-palace", "name": "颐和园", "address": "北京市海淀区新建宫门路19号", "lng": 116.2755, "lat": 39.9999, "rating": 4.7, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:beijing:temple-heaven", "name": "天坛公园", "address": "北京市东城区天坛东路甲1号", "lng": 116.4109, "lat": 39.8819, "rating": 4.7, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:beijing:national-museum", "name": "中国国家博物馆", "address": "北京市东城区东长安街16号", "lng": 116.4010, "lat": 39.9051, "rating": 4.8, "level": "AAAA", "type": "科教文化服务;博物馆"},
        {"id": "curated:beijing:badaling", "name": "八达岭长城", "address": "北京市延庆区G6京藏高速58号出口", "lng": 116.0167, "lat": 40.3560, "rating": 4.7, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:beijing:shichahai", "name": "什刹海", "address": "北京市西城区地安门西大街", "lng": 116.3860, "lat": 39.9403, "rating": 4.6, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:beijing:nanluoguxiang", "name": "南锣鼓巷", "address": "北京市东城区南锣鼓巷", "lng": 116.4031, "lat": 39.9370, "rating": 4.4, "level": "AAA", "type": "风景名胜"},
        {"id": "curated:beijing:lama-temple", "name": "雍和宫", "address": "北京市东城区雍和宫大街12号", "lng": 116.4175, "lat": 39.9471, "rating": 4.7, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:beijing:jingshan", "name": "景山公园", "address": "北京市西城区景山西街44号", "lng": 116.3967, "lat": 39.9250, "rating": 4.7, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:beijing:beihai", "name": "北海公园", "address": "北京市西城区文津街1号", "lng": 116.3896, "lat": 39.9255, "rating": 4.7, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:beijing:qianmen", "name": "前门大街", "address": "北京市东城区前门大街", "lng": 116.3977, "lat": 39.8958, "rating": 4.5, "level": "AAA", "type": "风景名胜;特色街区"},
        {"id": "curated:beijing:wangfujing", "name": "王府井步行街", "address": "北京市东城区王府井大街", "lng": 116.4116, "lat": 39.9148, "rating": 4.4, "level": "AAA", "type": "风景名胜;特色街区"},
        {"id": "curated:beijing:798", "name": "798艺术区", "address": "北京市朝阳区酒仙桥路2号", "lng": 116.5016, "lat": 39.9842, "rating": 4.5, "level": "AAA", "type": "科教文化服务;艺术区"},
        {"id": "curated:beijing:olympic", "name": "奥林匹克公园", "address": "北京市朝阳区北辰东路15号", "lng": 116.3966, "lat": 40.0080, "rating": 4.6, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:beijing:prince-gong", "name": "恭王府", "address": "北京市西城区前海西街17号", "lng": 116.3863, "lat": 39.9372, "rating": 4.6, "level": "AAAAA", "type": "风景名胜"},
    ],
    "上海": [
        {"id": "curated:shanghai:bund", "name": "外滩", "address": "上海市黄浦区中山东一路", "lng": 121.4906, "lat": 31.2397, "rating": 4.8, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:shanghai:oriental-pearl", "name": "东方明珠广播电视塔", "address": "上海市浦东新区世纪大道1号", "lng": 121.4998, "lat": 31.2397, "rating": 4.7, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:shanghai:yu-garden", "name": "豫园", "address": "上海市黄浦区福佑路168号", "lng": 121.4920, "lat": 31.2272, "rating": 4.7, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:shanghai:museum", "name": "上海博物馆", "address": "上海市黄浦区人民大道201号", "lng": 121.4752, "lat": 31.2285, "rating": 4.8, "level": "AAAA", "type": "科教文化服务;博物馆"},
        {"id": "curated:shanghai:nanjing-road", "name": "南京路步行街", "address": "上海市黄浦区南京东路", "lng": 121.4820, "lat": 31.2380, "rating": 4.6, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:shanghai:xintiandi", "name": "上海新天地", "address": "上海市黄浦区太仓路181弄", "lng": 121.4753, "lat": 31.2191, "rating": 4.6, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:shanghai:disney", "name": "上海迪士尼乐园", "address": "上海市浦东新区川沙新镇黄赵路310号", "lng": 121.6679, "lat": 31.1497, "rating": 4.7, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:shanghai:city-god", "name": "城隍庙旅游区", "address": "上海市黄浦区方浜中路249号", "lng": 121.4916, "lat": 31.2273, "rating": 4.5, "level": "AAAA", "type": "风景名胜;特色街区"},
        {"id": "curated:shanghai:wukang", "name": "武康大楼", "address": "上海市徐汇区淮海中路1842-1858号", "lng": 121.4387, "lat": 31.2146, "rating": 4.5, "level": "AAA", "type": "风景名胜;历史建筑"},
        {"id": "curated:shanghai:tianzifang", "name": "田子坊", "address": "上海市黄浦区泰康路210弄", "lng": 121.4690, "lat": 31.2097, "rating": 4.4, "level": "AAA", "type": "风景名胜;特色街区"},
        {"id": "curated:shanghai:westbund", "name": "西岸美术馆", "address": "上海市徐汇区龙腾大道2600号", "lng": 121.4590, "lat": 31.1740, "rating": 4.6, "level": "AAA", "type": "科教文化服务;美术馆"},
        {"id": "curated:shanghai:lupu", "name": "浦东美术馆", "address": "上海市浦东新区滨江大道2777号", "lng": 121.4996, "lat": 31.2356, "rating": 4.6, "level": "AAA", "type": "科教文化服务;美术馆"},
        {"id": "curated:shanghai:zhujiajiao", "name": "朱家角古镇", "address": "上海市青浦区课植园路555号", "lng": 121.0560, "lat": 31.1105, "rating": 4.5, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:shanghai:shanghai-tower", "name": "上海中心大厦", "address": "上海市浦东新区银城中路501号", "lng": 121.5011, "lat": 31.2335, "rating": 4.6, "level": "AAAA", "type": "风景名胜;观景台"},
        {"id": "curated:shanghai:luiiazui", "name": "陆家嘴中心绿地", "address": "上海市浦东新区陆家嘴环路", "lng": 121.5032, "lat": 31.2365, "rating": 4.5, "level": "AAA", "type": "风景名胜;城市公园"},
        {"id": "curated:shanghai:sipan", "name": "思南公馆", "address": "上海市黄浦区复兴中路523号", "lng": 121.4661, "lat": 31.2154, "rating": 4.5, "level": "AAA", "type": "风景名胜;历史街区"},
        {"id": "curated:shanghai:yuyuan-road", "name": "愚园路历史风貌区", "address": "上海市长宁区愚园路", "lng": 121.4385, "lat": 31.2208, "rating": 4.5, "level": "AAA", "type": "风景名胜;历史街区"},
        {"id": "curated:shanghai:waibaidu", "name": "外白渡桥", "address": "上海市黄浦区中山东一路", "lng": 121.4902, "lat": 31.2474, "rating": 4.6, "level": "AAA", "type": "风景名胜"},
        {"id": "curated:shanghai:people-square", "name": "人民广场", "address": "上海市黄浦区人民大道", "lng": 121.4757, "lat": 31.2304, "rating": 4.5, "level": "AAA", "type": "风景名胜;城市广场"},
    ],
    "西安": [
        {"id": "curated:xian:terracotta", "name": "秦始皇兵马俑博物馆", "address": "西安市临潼区秦陵北路", "lng": 109.2785, "lat": 34.3853, "rating": 4.8, "level": "AAAAA", "type": "科教文化服务;博物馆"},
        {"id": "curated:xian:city-wall", "name": "西安城墙", "address": "西安市碑林区南大街", "lng": 108.9423, "lat": 34.2541, "rating": 4.7, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:xian:big-goose", "name": "大雁塔", "address": "西安市雁塔区慈恩路1号", "lng": 108.9641, "lat": 34.2183, "rating": 4.6, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:xian:bell-tower", "name": "西安钟楼", "address": "西安市碑林区东西南北四条大街交汇处", "lng": 108.9439, "lat": 34.2610, "rating": 4.6, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:xian:shaanxi-museum", "name": "陕西历史博物馆", "address": "西安市雁塔区小寨东路91号", "lng": 108.9558, "lat": 34.2219, "rating": 4.8, "level": "AAAA", "type": "科教文化服务;博物馆"},
    ],
    "杭州": [
        {"id": "curated:hangzhou:west-lake", "name": "西湖风景名胜区", "address": "杭州市西湖区龙井路1号", "lng": 120.1410, "lat": 30.2590, "rating": 4.8, "level": "AAAAA", "type": "风景名胜"},
        {"id": "curated:hangzhou:lingyin", "name": "灵隐寺", "address": "杭州市西湖区灵隐路法云弄1号", "lng": 120.1010, "lat": 30.2409, "rating": 4.7, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:hangzhou:leifeng", "name": "雷峰塔景区", "address": "杭州市西湖区南山路15号", "lng": 120.1488, "lat": 30.2336, "rating": 4.6, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:hangzhou:xixi", "name": "西溪国家湿地公园", "address": "杭州市西湖区天目山路518号", "lng": 120.0648, "lat": 30.2697, "rating": 4.6, "level": "AAAAA", "type": "风景名胜"},
    ],
    "成都": [
        {"id": "curated:chengdu:panda", "name": "成都大熊猫繁育研究基地", "address": "成都市成华区熊猫大道1375号", "lng": 104.1458, "lat": 30.7397, "rating": 4.7, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:chengdu:kuanzhai", "name": "宽窄巷子", "address": "成都市青羊区金河路口宽窄巷子", "lng": 104.0562, "lat": 30.6730, "rating": 4.5, "level": "AA", "type": "风景名胜"},
        {"id": "curated:chengdu:jinli", "name": "锦里古街", "address": "成都市武侯区武侯祠大街231号", "lng": 104.0491, "lat": 30.6444, "rating": 4.5, "level": "AAAA", "type": "风景名胜"},
        {"id": "curated:chengdu:wuhou", "name": "武侯祠", "address": "成都市武侯区武侯祠大街231号", "lng": 104.0479, "lat": 30.6461, "rating": 4.6, "level": "AAAA", "type": "风景名胜"},
    ],
}


def get_curated_pois(city: str, keywords: Optional[List[str]] = None) -> List[POIInfo]:
    rows = _CURATED.get(city or "", [])
    if not rows:
        return []

    kws = [kw for kw in (keywords or []) if kw]
    selected = []
    for row in rows:
        if not kws or any(kw in row["name"] or kw in row["type"] for kw in kws):
            selected.append(row)
    if not selected:
        selected = rows
    else:
        selected_ids = {row["id"] for row in selected}
        selected.extend(row for row in rows if row["id"] not in selected_ids)

    out: List[POIInfo] = []
    for row in selected:
        out.append(POIInfo(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            address=row["address"],
            location=Location(longitude=row["lng"], latitude=row["lat"]),
            rating=row["rating"],
            biz_type=row["type"].split(";")[0],
            typecode="110000",
            level=row["level"],
        ))
    return out

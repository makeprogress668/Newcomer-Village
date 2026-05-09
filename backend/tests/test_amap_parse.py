"""amap_service 解析层单元测试。

只测纯函数，不依赖真实 MCP 调用。覆盖：
- _strip_mcp_envelope: HelloAgents MCPTool 包裹前缀
- _extract_json: 含前缀文本/markdown 围栏/原始 JSON
- _parse_location: 'lng,lat' 字符串 + dict 两种格式
- _parse_pois: 高德 V3 标准结构 + 字段缺失
- _parse_weather: forecasts.casts 嵌套结构
- _parse_route: route.paths[0]
"""

import json
import sys
from pathlib import Path

# 把 backend 目录加入 sys.path 以便 import app.*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.amap_service import (  # noqa: E402
    _extract_json,
    _parse_location,
    _parse_pois,
    _parse_poi_item,
    _parse_route,
    _parse_weather,
    _strip_mcp_envelope,
)


# ============ _strip_mcp_envelope ============

def test_strip_envelope_removes_prefix():
    raw = "工具 'maps_text_search' 执行结果:\n{\"pois\":[]}"
    assert _strip_mcp_envelope(raw) == "{\"pois\":[]}"


def test_strip_envelope_keeps_clean_json():
    raw = "{\"pois\":[]}"
    assert _strip_mcp_envelope(raw) == raw


def test_strip_envelope_handles_chinese_colon():
    raw = "工具 'maps_weather' 执行结果：\n{\"forecasts\":[]}"
    assert _strip_mcp_envelope(raw) == "{\"forecasts\":[]}"


# ============ _extract_json ============

def test_extract_json_plain_object():
    assert _extract_json('{"a":1}') == {"a": 1}


def test_extract_json_with_envelope():
    raw = "工具 'x' 执行结果:\n{\"a\":1, \"b\":[2,3]}"
    assert _extract_json(raw) == {"a": 1, "b": [2, 3]}


def test_extract_json_markdown_fence():
    raw = "Some text\n```json\n{\"k\": \"v\"}\n```"
    assert _extract_json(raw) == {"k": "v"}


def test_extract_json_with_leading_text():
    raw = 'Found result:\n{"pois":[{"id":"X1"}]}'
    assert _extract_json(raw) == {"pois": [{"id": "X1"}]}


def test_extract_json_array():
    raw = '前缀文本 [{"a":1},{"b":2}] 末尾'
    assert _extract_json(raw) == [{"a": 1}, {"b": 2}]


def test_extract_json_returns_none_when_invalid():
    assert _extract_json("纯文本，没有 JSON") is None
    assert _extract_json("") is None
    assert _extract_json(None) is None  # type: ignore


def test_extract_json_handles_string_with_braces():
    """字符串里的 { } 不应破坏括号计数。"""
    raw = '{"name":"测试 {} 名称","value":1}'
    assert _extract_json(raw) == {"name": "测试 {} 名称", "value": 1}


# ============ _parse_location ============

def test_parse_location_string_format():
    loc = _parse_location("116.397128,39.916527")
    assert loc is not None
    assert abs(loc.longitude - 116.397128) < 1e-6
    assert abs(loc.latitude - 39.916527) < 1e-6


def test_parse_location_dict_format():
    loc = _parse_location({"longitude": 121.5, "latitude": 31.2})
    assert loc is not None
    assert loc.longitude == 121.5


def test_parse_location_dict_with_lng_lat():
    loc = _parse_location({"lng": 113.0, "lat": 23.0})
    assert loc is not None
    assert loc.longitude == 113.0


def test_parse_location_invalid_returns_none():
    assert _parse_location(None) is None
    assert _parse_location("") is None
    assert _parse_location("116.4") is None  # 单值无逗号
    assert _parse_location("abc,def") is None


# ============ _parse_poi_item ============

def test_parse_poi_item_full():
    item = {
        "id": "B0FFFAB123",
        "name": "故宫博物院",
        "type": "风景名胜;风景名胜相关;旅游景点",
        "address": "景山前街4号",
        "location": "116.397025,39.918058",
        "tel": "010-85007421",
        "biz_ext": {"rating": "4.8", "cost": ""},
        "photos": [
            {"title": "外景", "url": "https://amap.com/p1.jpg"},
            {"title": "内景", "url": "https://amap.com/p2.jpg"},
        ],
    }
    poi = _parse_poi_item(item)
    assert poi is not None
    assert poi.id == "B0FFFAB123"
    assert poi.name == "故宫博物院"
    assert poi.rating == 4.8
    assert poi.cost is None  # 空字符串归一化为 None
    assert poi.photos == ["https://amap.com/p1.jpg", "https://amap.com/p2.jpg"]
    assert poi.biz_type == "风景名胜"
    assert poi.location.longitude > 116


def test_parse_poi_item_missing_location_keeps_poi():
    """精简版 amap-mcp-server 不返回 location,需要保留 POI 等待 detail 补全。"""
    item = {"id": "X", "name": "故宫博物院", "address": "景山前街4号", "typecode": "110201|140100"}
    poi = _parse_poi_item(item)
    assert poi is not None
    assert poi.location is None
    assert poi.typecode == "110201|140100"


def test_parse_poi_item_rating_at_top_level():
    """精简版 detail 把 rating/level 放在顶层而非 biz_ext。"""
    item = {
        "id": "B000A8UIN8",
        "name": "故宫博物院",
        "location": "116.397,39.917",
        "address": "景山前街4号",
        "type": "风景名胜;世界遗产",
        "rating": "4.9",
        "level": "AAAAA",
        "cost": [],  # 精简版用 [] 表示空
    }
    poi = _parse_poi_item(item)
    assert poi is not None
    assert poi.rating == 4.9
    assert poi.level == "AAAAA"
    assert poi.cost is None  # [] 应被规范为 None


def test_parse_poi_item_missing_name_returns_none():
    item = {"id": "X", "name": "", "location": "1.0,2.0"}
    assert _parse_poi_item(item) is None


def test_parse_poi_item_biz_ext_as_list():
    """高德有时把空 biz_ext 返回成 [] 而不是 {}。"""
    item = {
        "id": "X1",
        "name": "测试",
        "type": "餐饮",
        "address": "某街",
        "location": "100.0,30.0",
        "biz_ext": [],
    }
    poi = _parse_poi_item(item)
    assert poi is not None
    assert poi.rating is None
    assert poi.cost is None


# ============ _parse_pois ============

def test_parse_pois_from_v3_response():
    data = {
        "status": "1",
        "info": "OK",
        "count": "2",
        "pois": [
            {
                "id": "P1",
                "name": "天安门",
                "type": "景点",
                "address": "东城区",
                "location": "116.397,39.909",
                "biz_ext": {"rating": "4.9"},
            },
            {
                "id": "P2",
                "name": "颐和园",
                "type": "景点",
                "address": "海淀区",
                "location": "116.265,39.999",
                "biz_ext": {"rating": "4.7"},
            },
        ],
    }
    pois = _parse_pois(data)
    assert len(pois) == 2
    assert pois[0].name == "天安门"
    assert pois[1].rating == 4.7


def test_parse_pois_keeps_missing_location():
    """精简版返回常缺 location,本层不再过滤,留给 poi_aggregator 调 detail 补全。"""
    data = {"pois": [
        {"id": "ok", "name": "好的", "type": "", "address": "", "location": "1,2"},
        {"id": "no-loc", "name": "故宫博物院", "address": "景山前街4号", "typecode": "110201"},
    ]}
    pois = _parse_pois(data)
    assert len(pois) == 2
    by_id = {p.id: p for p in pois}
    assert by_id["ok"].location is not None
    assert by_id["no-loc"].location is None
    assert by_id["no-loc"].typecode == "110201"


def test_parse_pois_skips_unnamed_items():
    """没有 name 的项目仍然会被丢弃。"""
    data = {"pois": [
        {"id": "ok", "name": "好的", "location": "1,2"},
        {"id": "noname", "name": "", "location": "1,2"},
        {"id": "noname2", "location": "1,2"},
    ]}
    pois = _parse_pois(data)
    assert len(pois) == 1
    assert pois[0].id == "ok"


def test_parse_pois_from_list_root():
    """有些 MCP 实现直接返回 list。"""
    data = [
        {"id": "1", "name": "A", "type": "", "address": "", "location": "1,2"},
    ]
    pois = _parse_pois(data)
    assert len(pois) == 1


def test_parse_pois_empty_input():
    assert _parse_pois(None) == []
    assert _parse_pois({}) == []
    assert _parse_pois({"pois": []}) == []


# ============ _parse_weather ============

def test_parse_weather_forecasts_structure():
    data = {
        "forecasts": [{
            "city": "北京",
            "casts": [
                {
                    "date": "2026-04-27",
                    "dayweather": "晴",
                    "nightweather": "多云",
                    "daytemp": "25",
                    "nighttemp": "12",
                    "daywind": "南",
                    "daypower": "1-3",
                },
                {
                    "date": "2026-04-28",
                    "dayweather": "多云",
                    "nightweather": "阴",
                    "daytemp": "23",
                    "nighttemp": "10",
                    "daywind": "北",
                    "daypower": "3-4",
                },
            ],
        }]
    }
    weather = _parse_weather(data)
    assert len(weather) == 2
    assert weather[0].date == "2026-04-27"
    assert weather[0].day_weather == "晴"
    assert weather[0].day_temp == 25  # 通过 schema validator 转 int
    assert weather[1].night_weather == "阴"


def test_parse_weather_empty():
    assert _parse_weather(None) == []
    assert _parse_weather({}) == []


# ============ _parse_route ============

def test_parse_route_with_paths():
    data = {
        "route": {
            "paths": [
                {"distance": "5234", "duration": "1820", "steps": [{"instruction": "向北"}]}
            ]
        }
    }
    r = _parse_route(data)
    assert r["distance_m"] == 5234.0
    assert r["duration_s"] == 1820.0
    assert len(r["steps"]) == 1


def test_parse_route_flat():
    """有些响应没有外层 route 包裹。"""
    data = {"paths": [{"distance": "100", "duration": "60"}]}
    r = _parse_route(data)
    assert r["distance_m"] == 100.0
    assert r["duration_s"] == 60.0


def test_parse_route_empty():
    assert _parse_route({}) == {}
    assert _parse_route(None) == {}


# ============ end-to-end 字符串解析（模拟 MCP 完整返回） ============

def test_full_pipeline_search_poi_response():
    """模拟 MCPTool.run 的完整返回，验证 _extract_json + _parse_pois 串联。"""
    payload = {
        "status": "1",
        "info": "OK",
        "pois": [{
            "id": "AMAP001",
            "name": "故宫",
            "type": "风景名胜;旅游景点",
            "address": "北京市东城区",
            "location": "116.397,39.918",
            "biz_ext": {"rating": "4.8"},
            "photos": [{"url": "https://example.com/gugong.jpg"}],
        }],
    }
    raw = "工具 'maps_text_search' 执行结果:\n" + json.dumps(payload, ensure_ascii=False)
    data = _extract_json(raw)
    pois = _parse_pois(data)
    assert len(pois) == 1
    assert pois[0].name == "故宫"
    assert pois[0].rating == 4.8
    assert pois[0].photos == ["https://example.com/gugong.jpg"]
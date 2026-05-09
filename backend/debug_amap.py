"""高德 MCP 调试脚本：打印原始返回，快速定位 search_poi 解析问题。

用法:
    cd backend
    .venv\\Scripts\\python.exe debug_amap.py
"""

from app.services.logging_setup import setup_logging
setup_logging()

from app.services.amap_service import get_amap_service

amap = get_amap_service()

print("\n" + "=" * 70)
print("[1] 列出 MCP server 暴露的所有工具名（确认工具名拼写）")
print("=" * 70)
tools = amap.mcp_tool._available_tools or []
for t in tools[:20]:
    print(f"  - {t.get('name', '?')}")
print(f"共 {len(tools)} 个工具")

print("\n" + "=" * 70)
print("[2] 直接调用 maps_text_search 看原始返回")
print("=" * 70)
raw = amap._call("maps_text_search", {
    "keywords": "故宫",
    "city": "北京",
    "citylimit": "true",
})
print(f"RAW (前 2000 字):\n{raw[:2000]}")

print("\n" + "=" * 70)
print("[3] 跑完整 search_poi 看解析后的结果")
print("=" * 70)
pois = amap.search_poi("故宫", "北京")
print(f"解析后 POI 数量: {len(pois)}")
for p in pois[:5]:
    print(f"  - {p.name} | location={p.location} | rating={p.rating}")

print("\n" + "=" * 70)
print("[4] 调 maps_search_detail 看详情接口的返回（关键！）")
print("=" * 70)
# 用第 [2] 步抓到的故宫博物院 ID
detail_raw = amap._call("maps_search_detail", {"id": "B000A8UIN8"})
print(f"DETAIL RAW (前 3000 字):\n{detail_raw[:3000]}")

print("\n" + "=" * 70)
print("[5] 调 maps_geo 用地址换坐标（备选方案）")
print("=" * 70)
geo_raw = amap._call("maps_geo", {"address": "景山前街4号", "city": "北京"})
print(f"GEO RAW (前 1500 字):\n{geo_raw[:1500]}")

print("\n" + "=" * 70)
print("[6] 端到端验证: collect_attractions 看真实景点候选 (含 free_text 升旗)")
print("=" * 70)
from app.services.poi_aggregator import collect_attractions

candidates = collect_attractions(
    city="北京",
    preferences=["历史文化"],
    free_text="想看升旗仪式,顺便去爬长城",
    top_n=10,
)
print(f"\n最终景点候选数: {len(candidates)}")
for i, p in enumerate(candidates, 1):
    loc = f"{p.location.longitude:.4f},{p.location.latitude:.4f}" if p.location else "无坐标"
    print(f"  {i:2d}. {p.name} | rating={p.rating} | level={p.level} | type={p.biz_type} | {loc}")

print("\n" + "=" * 70)
print("[7] 天气接口完整诊断")
print("=" * 70)
weather_raw = amap._call("maps_weather", {"city": "北京"})
print(f"WEATHER RAW (前 2000 字):\n{weather_raw[:2000]}")
weather_parsed = amap.get_weather("北京")
print(f"\n解析得 {len(weather_parsed)} 天:")
for w in weather_parsed[:5]:
    print(f"  - {w.date} | {w.day_weather}/{w.night_weather} | {w.day_temp}°/{w.night_temp}° | {w.wind_direction} {w.wind_power}")

print("\n" + "=" * 70)
print("[8] around_search 酒店候选 + 黑名单过滤")
print("=" * 70)
from app.models.schemas import Location
from app.data.keywords import is_blacklisted_hotel
# 故宫附近找酒店
center = Location(longitude=116.397, latitude=39.918)
hotels_raw = amap.around_search(center, keywords="酒店", radius=2000)
print(f"around_search 召回数: {len(hotels_raw)}")
print("前 10 条 + 黑名单状态:")
for p in hotels_raw[:10]:
    flag = "❌黑名单" if is_blacklisted_hotel(p.name) else "✓"
    print(f"  {flag} {p.name} | rating={p.rating} | {p.address}")

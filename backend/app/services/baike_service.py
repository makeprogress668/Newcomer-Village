"""百度百科图片服务。

利用百度百科开放接口（appid=379020,被广泛使用,无需鉴权）获取景点首图。
对中文景点名命中率极高,且国内访问稳定 —— 解决 Unsplash 在国内访问困难
和中文搜索命中率低的双重问题。
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


_BAIKE_API = "https://baike.baidu.com/api/openapi/BaikeLemmaCardApi"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def get_baike_photo(name: str, city: str = "") -> Optional[str]:
    """从百度百科获取景点首图。优先用 "城市+景点" 词条减少歧义。"""
    if not name:
        return None

    # 候选查询词，从精确到泛化
    queries = []
    if city and city not in name:
        queries.append(f"{city}{name}")  # 如"北京故宫"消歧义
    queries.append(name)
    # 去掉子点位（"故宫博物院-午门" → "故宫博物院"）
    if "-" in name:
        queries.append(name.split("-")[0])
    if "(" in name:
        queries.append(name.split("(")[0])

    seen = set()
    for q in queries:
        if not q or q in seen:
            continue
        seen.add(q)
        url = _fetch(q)
        if url:
            logger.info("百度百科命中 [%s] query=%s", name, q)
            return url
    return None


def _fetch(query: str, max_retries: int = 2) -> Optional[str]:
    """单次百度百科查询。返回首图 URL 或 None。

    返回值含义:
      - URL 字符串: 命中
      - None: 词条不存在 / 词条存在但无主图 / 节流（已 retry 仍失败）
    errno 字段表示"词条不存在",直接 return 不重试,节省查询次数。
    """
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(
                _BAIKE_API,
                params={
                    "scope": "103",
                    "format": "json",
                    "appid": "379020",
                    "bk_key": query,
                    "bk_length": "600",
                },
                timeout=8,
                headers={"User-Agent": _USER_AGENT},
            )
            if resp.status_code == 200:
                data = resp.json() if resp.content else {}
                # errno → 词条本身不存在,换查询词
                if "errno" in data:
                    return None
                image = data.get("image")
                if isinstance(image, str) and image.startswith("http"):
                    return image
                # image=None 但有其它字段 → 词条存在但无主图,也不 retry
                if data.get("title") or data.get("id"):
                    return None
                # 走到这里说明返回结构异常（节流空响应等）→ retry
        except Exception as exc:
            logger.warning("百度百科取图异常 query=%s attempt=%d: %s", query, attempt, exc)
        if attempt < max_retries:
            time.sleep(0.5 * (attempt + 1))  # 指数退避: 0.5s → 1.0s
    return None

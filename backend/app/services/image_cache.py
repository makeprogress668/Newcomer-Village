"""景点图片 SQLite 缓存。

设计目标：
- 减少高德 / Unsplash 重复请求，省 API 配额
- 命中率统计可观测
- 简单依赖（仅 stdlib sqlite3 + threading.Lock，不引入 aiosqlite/redis）

key = normalize(name) + '|' + normalize(city)
"""

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


_DEFAULT_TTL_SECONDS = 30 * 24 * 3600  # 30 天

# 简单的命中率计数器（进程内）
_stats_lock = threading.Lock()
_stats = {"hit": 0, "miss": 0, "set": 0}


def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # 去除空白和常见标点（保留中文、字母、数字）
    return re.sub(r"[\s　,，./()（）·\-_]+", "", s)


_KEY_VERSION = "v3"  # 改版本号即可让全部旧缓存失效（无需删数据库）
# v3 = 接入高德 RESTful API (主源换成实景照,旧的百度百科 logo 缓存全部失效)


def make_key(name: str, city: str) -> str:
    return f"{_KEY_VERSION}|{_normalize(name)}|{_normalize(city)}"


class ImageCache:
    """SQLite 实现的图片缓存。线程安全。"""

    def __init__(self, db_path: str, ttl: int = _DEFAULT_TTL_SECONDS):
        self.db_path = db_path
        self.ttl = ttl
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_schema(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_cache (
                    key TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    source TEXT NOT NULL,
                    fetched_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, name: str, city: str) -> Optional[dict]:
        """返回 {url, source} 或 None（未命中或已过期）。"""
        key = make_key(name, city)
        if not key.replace("|", ""):
            return None
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT url, source, fetched_at FROM image_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            with _stats_lock:
                _stats["miss"] += 1
            return None
        url, source, fetched_at = row
        if int(time.time()) - int(fetched_at) > self.ttl:
            # 过期。简单策略：删除并视作未命中，让上层重新拉
            self._delete(key)
            with _stats_lock:
                _stats["miss"] += 1
            return None
        with _stats_lock:
            _stats["hit"] += 1
        return {"url": url, "source": source}

    def set(self, name: str, city: str, url: str, source: str) -> None:
        if not url:
            return
        key = make_key(name, city)
        if not key.replace("|", ""):
            return
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO image_cache(key, url, source, fetched_at) "
                "VALUES (?, ?, ?, ?)",
                (key, url, source, int(time.time())),
            )
            conn.commit()
        with _stats_lock:
            _stats["set"] += 1

    def _delete(self, key: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM image_cache WHERE key = ?", (key,))
            conn.commit()


# ============ 单例 ============

_cache: Optional[ImageCache] = None


def get_image_cache() -> ImageCache:
    global _cache
    if _cache is None:
        from ..config import get_settings
        settings = get_settings()
        _cache = ImageCache(db_path=settings.cache_db_path)
    return _cache


def get_cache_stats() -> dict:
    """返回当前进程的命中统计（用于日志/调试）。"""
    with _stats_lock:
        total = _stats["hit"] + _stats["miss"]
        rate = (_stats["hit"] / total) if total else 0.0
        return {**_stats, "hit_rate": round(rate, 3)}

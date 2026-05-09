"""统一日志配置：stdlib logging + RotatingFileHandler。

使用方式（在 run.py / 应用启动时调用一次）：

    from app.services.logging_setup import setup_logging
    setup_logging()

各模块：

    import logging
    logger = logging.getLogger(__name__)
    logger.info("...")
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_file: str = "app.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5MB / file
    backup_count: int = 5,
) -> None:
    """初始化全局 logging。重复调用安全（仅首次生效）。"""
    global _initialized
    if _initialized:
        return

    from ..config import get_settings
    settings = get_settings()
    level = (level or settings.log_level or "INFO").upper()
    log_dir = log_dir or getattr(settings, "log_dir", None) or "./logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # 清空已有 handler，避免重复输出
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    # 文件（带 rotate）
    file_path = Path(log_dir) / log_file
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    # 降低过吵的第三方日志
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True
    root.info(
        "logging initialized: level=%s, file=%s (rotate %dMB x %d)",
        level, file_path, max_bytes // (1024 * 1024), backup_count,
    )

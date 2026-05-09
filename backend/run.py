"""启动脚本"""

import uvicorn

from app.config import get_settings
from app.services.logging_setup import setup_logging

if __name__ == "__main__":
    setup_logging()
    settings = get_settings()

    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )


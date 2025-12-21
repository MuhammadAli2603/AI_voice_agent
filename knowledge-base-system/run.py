"""
Application runner script.
Simple entry point to start the FastAPI server.
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Server: http://{settings.host}:{settings.port}")
    print(f"Documentation: http://{settings.host}:{settings.port}/docs")
    print(f"Debug mode: {settings.debug}")
    print()

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )

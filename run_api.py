"""Quick start script for KisanCredit API.

Usage:
    python run_api.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn
    from src.utils.config import settings

    print("=" * 80)
    print("Starting KisanCredit API Server")
    print("=" * 80)
    print(f"Environment: {settings.environment}")
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print(f"Debug: {settings.debug}")
    print(f"Model Path: {settings.model_path}")
    print("-" * 80)
    print(f"API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"ReDoc: http://{settings.api_host}:{settings.api_port}/redoc")
    print(f"Health: http://{settings.api_host}:{settings.api_port}/api/v1/health")
    print("=" * 80)

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )

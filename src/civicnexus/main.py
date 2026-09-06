from fastapi import FastAPI

from civicnexus.api.router import api_router
from civicnexus.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()

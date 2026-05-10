import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import router as v1_router
from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger().setLevel(level)
    logging.getLogger("app").setLevel(level)


def create_app() -> FastAPI:
    settings = Settings()
    configure_logging(settings)
    app = FastAPI(
        title="Veraser",
        description="Video object removal API.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_origin_regex=settings.cors_allow_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(v1_router)
    return app


app = create_app()

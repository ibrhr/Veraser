from fastapi import FastAPI

from app.api.v1.routes import router as v1_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Veraser",
        description="Video object removal API.",
        version="0.1.0",
    )
    app.include_router(v1_router)
    return app


app = create_app()

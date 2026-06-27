from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.datasets import router as datasets_router
from backend.api.keypoint_observations import router as keypoint_observations_router
from backend.api.training import router as training_router


def create_app() -> FastAPI:
    app = FastAPI(title="URDF Ops Backend", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(training_router)
    app.include_router(datasets_router)
    app.include_router(keypoint_observations_router)
    return app


app = create_app()

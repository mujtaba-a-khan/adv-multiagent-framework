"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adversarial_framework.config.settings import get_settings
from adversarial_framework.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup / shutdown hooks."""
    settings = get_settings()
    setup_logging(level=settings.log_level, json_output=settings.log_json)
    logger.info("startup", environment=settings.environment)

    yield

    # Shutdown: dispose DB engine
    from adversarial_framework.db.engine import dispose_engine

    await dispose_engine()
    logger.info("shutdown")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Adversarial Framework API",
        description="Multi-agent adversarial red-teaming framework for AI safety",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from adversarial_framework.api.v1.router import v1_router

    app.include_router(v1_router, prefix="/api/v1")

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app

"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

try:
    import structlog
except Exception:  # pragma: no cover
    structlog = None
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.router import api_router
from app.config import get_settings
from app.database import Base, engine

settings = get_settings()

if structlog is not None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )
    logger = structlog.get_logger()
else:
    logger = logging.getLogger("signflow")

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_runtime_schema() -> None:
    """Apply small runtime-safe schema patches for local SQLite compatibility."""
    inspector = inspect(engine)
    if "model_versions" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("model_versions")}
    if "class_labels" in columns:
        return

    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE model_versions ADD COLUMN class_labels JSON NOT NULL DEFAULT '[]'")
        )


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database schema and ML pipeline for local development."""
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()
    logger.info("app.startup", env=settings.env)

    # Pre-load inference pipeline with active model
    try:
        from app.api.translate import get_or_create_pipeline

        pipeline = get_or_create_pipeline()
        logger.info("inference_pipeline_preloaded")
    except Exception as e:
        logger.warning("failed_to_preload_pipeline", error=str(e))


@app.get("/healthz", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Simple health endpoint for probes."""
    return {"status": "ok"}


app.include_router(api_router, prefix=settings.api_v1_prefix)

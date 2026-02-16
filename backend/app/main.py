"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

try:
    import structlog
except Exception:  # pragma: no cover
    structlog = None
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import inspect, text

from app.api.router import api_router
from app.config import get_settings
from app.database import Base, engine
from app.ml.metrics import get_metrics_collector

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

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_host_list)


@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    """Reject oversized request bodies before expensive processing."""
    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                max_bytes = settings.max_request_mb * 1024 * 1024
                if int(content_length) > max_bytes:
                    return JSONResponse(status_code=413, content={"detail": "Request body too large"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
    return await call_next(request)


def ensure_runtime_schema() -> None:
    """Apply small runtime-safe schema patches for local SQLite compatibility."""
    with engine.begin() as connection:
        inspector = inspect(connection)
        table_names = inspector.get_table_names()
        if "model_versions" in table_names:
            model_columns = {column["name"] for column in inspector.get_columns("model_versions")}
            if "class_labels" not in model_columns:
                connection.execute(
                    text("ALTER TABLE model_versions ADD COLUMN class_labels JSON NOT NULL DEFAULT '[]'")
                )
            if "metadata" not in model_columns:
                connection.execute(
                    text("ALTER TABLE model_versions ADD COLUMN metadata JSON NOT NULL DEFAULT '{}'")
                )

        if "videos" in table_names:
            video_columns = {column["name"] for column in inspector.get_columns("videos")}
            if "detection_rate" not in video_columns:
                connection.execute(
                    text("ALTER TABLE videos ADD COLUMN detection_rate FLOAT NOT NULL DEFAULT 0")
                )
            if "quality_score" not in video_columns:
                connection.execute(
                    text("ALTER TABLE videos ADD COLUMN quality_score FLOAT NOT NULL DEFAULT 0")
                )
            if "is_trainable" not in video_columns:
                connection.execute(
                    text("ALTER TABLE videos ADD COLUMN is_trainable BOOLEAN NOT NULL DEFAULT TRUE")
                )
            if "landmark_feature_dim" not in video_columns:
                connection.execute(
                    text("ALTER TABLE videos ADD COLUMN landmark_feature_dim INTEGER NOT NULL DEFAULT 225")
                )


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database schema and ML pipeline for local development."""
    if settings.env.lower() == "production":
        if "*" in settings.cors_origin_list:
            raise RuntimeError("CORS_ORIGINS cannot contain '*' in production")
        if settings.docs_enabled:
            logger.warning("docs_enabled_in_production")

    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()
    logger.info("app.startup", env=settings.env)

    # Pre-load inference pipeline with active model
    try:
        from app.api.translate import get_or_create_pipeline

        _ = get_or_create_pipeline()
        logger.info("inference_pipeline_preloaded")
    except Exception as e:
        logger.warning("failed_to_preload_pipeline", error=str(e))


@app.get("/healthz", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Simple health endpoint for probes."""
    return {"status": "ok"}


@app.get("/metrics", tags=["monitoring"])
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    payload, content_type = get_metrics_collector().render_latest()
    return Response(content=payload, media_type=content_type)


app.include_router(api_router, prefix=settings.api_v1_prefix)

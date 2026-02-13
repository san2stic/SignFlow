"""Aggregate API router for all v1 endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import dictionary, media, models, signs, stats, training, translate

api_router = APIRouter()
api_router.include_router(translate.router, prefix="/translate", tags=["translate"])
api_router.include_router(signs.router, prefix="/signs", tags=["signs"])
api_router.include_router(media.router, prefix="/media", tags=["media"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(dictionary.router, prefix="/dictionary", tags=["dictionary"])
api_router.include_router(stats.router, prefix="/stats", tags=["stats"])

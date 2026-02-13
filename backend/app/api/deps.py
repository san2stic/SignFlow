"""Reusable dependency providers for API routes."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Generator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import SessionLocal


class SimpleRateLimiter:
    """In-memory fixed-window rate limiter keyed by IP and endpoint."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, key: str, limit_per_minute: int) -> None:
        """Raise if requests exceed configured per-minute threshold."""
        now = time.time()
        cutoff = now - 60
        with self._lock:
            window = [ts for ts in self._store[key] if ts >= cutoff]
            if len(window) >= limit_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                )
            window.append(now)
            self._store[key] = window


rate_limiter = SimpleRateLimiter()


def get_db() -> Generator[Session, None, None]:
    """Yield database session and close safely."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_app_settings() -> Settings:
    """Expose settings dependency for routes/services."""
    return get_settings()


def enforce_rate_limit(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> None:
    """Apply generic rate limiting by client host and endpoint path."""
    host = request.client.host if request.client else "unknown"
    key = f"{host}:{request.url.path}"
    rate_limiter.check(key=key, limit_per_minute=settings.rate_limit_per_minute)

"""Reusable dependency providers for API routes."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Generator

from fastapi import Depends, HTTPException, Request, WebSocket, status
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


class WebSocketConnectionLimiter:
    """Track active websocket connections by client host and endpoint."""

    def __init__(self) -> None:
        self._active_connections: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def acquire(self, key: str, limit: int) -> bool:
        """Reserve one websocket slot if under limit."""
        with self._lock:
            current = self._active_connections.get(key, 0)
            if current >= limit:
                return False
            self._active_connections[key] = current + 1
            return True

    def release(self, key: str) -> None:
        """Release one websocket slot."""
        with self._lock:
            current = self._active_connections.get(key, 0)
            if current <= 1:
                self._active_connections.pop(key, None)
                return
            self._active_connections[key] = current - 1


rate_limiter = SimpleRateLimiter()
ws_connection_limiter = WebSocketConnectionLimiter()


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


def _request_host(request: Request) -> str:
    """Resolve best-effort request host for per-client controls."""
    return request.client.host if request.client else "unknown"


def _websocket_host(websocket: WebSocket) -> str:
    """Resolve best-effort websocket host for per-client controls."""
    return websocket.client.host if websocket.client else "unknown"


def enforce_rate_limit(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> None:
    """Apply generic rate limiting by client host and endpoint path."""
    host = _request_host(request)
    key = f"{host}:{request.url.path}"
    rate_limiter.check(key=key, limit_per_minute=settings.rate_limit_per_minute)


def enforce_write_rate_limit(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> None:
    """Apply stricter per-IP limits on mutating endpoints."""
    host = _request_host(request)
    key = f"{host}:{request.url.path}:write"
    rate_limiter.check(key=key, limit_per_minute=settings.write_rate_limit_per_minute)


def acquire_ws_slot(websocket: WebSocket, settings: Settings, *, endpoint: str) -> str:
    """Acquire websocket slot for client or raise 429."""
    host = _websocket_host(websocket)
    key = f"{host}:{endpoint}"
    if not ws_connection_limiter.acquire(key=key, limit=settings.ws_max_connections_per_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many concurrent websocket connections",
        )
    return key


def release_ws_slot(slot_key: str | None) -> None:
    """Release previously-acquired websocket slot."""
    if slot_key:
        ws_connection_limiter.release(slot_key)


def enforce_ws_message_rate(host: str, *, endpoint: str, settings: Settings) -> None:
    """Rate limit websocket message throughput per client and endpoint."""
    key = f"{host}:{endpoint}:ws-msg"
    rate_limiter.check(key=key, limit_per_minute=settings.ws_messages_per_minute)

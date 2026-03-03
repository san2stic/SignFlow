"""REST and WebSocket endpoints for the Git→Docker auto-update system."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status

from app.api.deps import (
    acquire_ws_slot,
    enforce_rate_limit,
    enforce_write_rate_limit,
    release_ws_slot,
)
from app.auth.dependencies import get_current_active_user
from app.config import get_settings
from app.models.user import User
from app.schemas.deployment import DeploymentHistoryRead, TriggerRequest, TriggerResponse, UpdaterStatus

router = APIRouter()


def _get_updater_service():
    """Lazy accessor for the updater singleton (avoids circular import at module level)."""
    from app.services.updater_service import updater_service

    if updater_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Updater service is not initialised",
        )
    return updater_service


# ─────────────────────────── REST endpoints ───────────────────────────────────


@router.get(
    "/status",
    response_model=UpdaterStatus,
    tags=["updater"],
    dependencies=[Depends(enforce_rate_limit)],
)
async def get_updater_status() -> UpdaterStatus:
    """Return the current state of the updater service and last deployment."""
    svc = _get_updater_service()
    data = await svc.get_status()
    return UpdaterStatus(
        state=data["state"],
        current_deployment_id=data.get("current_deployment_id"),
        last_deployment=data.get("last_deployment"),
        git_remote_url=data.get("git_remote_url", ""),
        git_branch=data.get("git_branch", "main"),
        local_commit=data.get("local_commit"),
        remote_commit=data.get("remote_commit"),
        last_check_at=(
            datetime.fromisoformat(data["last_check_at"])
            if data.get("last_check_at")
            else None
        ),
        poll_interval_s=data.get("poll_interval_s", 60),
        auto_update_enabled=data.get("auto_update_enabled", False),
    )


@router.get(
    "/history",
    response_model=list[DeploymentHistoryRead],
    tags=["updater"],
    dependencies=[Depends(enforce_rate_limit)],
)
async def get_deployment_history(limit: int = 20) -> list[dict]:
    """Return the last N deployment records (most recent first)."""
    svc = _get_updater_service()
    history = await svc.get_history(limit=min(limit, 100))
    return history


@router.post(
    "/trigger",
    response_model=TriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["updater"],
    dependencies=[Depends(enforce_write_rate_limit)],
)
async def trigger_deployment(
    payload: TriggerRequest = TriggerRequest(),
    current_user: User = Depends(get_current_active_user),
) -> TriggerResponse:
    """Trigger a manual deployment immediately.

    Protected by JWT authentication (POST mutation).
    Returns 409 if a deployment is already in progress.
    """
    svc = _get_updater_service()
    try:
        deployment_id = await svc.trigger_manual()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    return TriggerResponse(
        deployment_id=deployment_id,
        message="Deployment triggered",
    )


@router.post(
    "/rollback/{deployment_id}",
    response_model=TriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["updater"],
    dependencies=[Depends(enforce_write_rate_limit)],
)
async def trigger_rollback(
    deployment_id: int,
    current_user: User = Depends(get_current_active_user),
) -> TriggerResponse:
    """Trigger a rollback to a specific previous successful deployment.

    Protected by JWT authentication (POST mutation).
    Returns 404 if the target deployment does not exist.
    Returns 409 if a deployment is already in progress.
    """
    svc = _get_updater_service()
    try:
        new_id = await svc.rollback_to(deployment_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    return TriggerResponse(
        deployment_id=new_id,
        message=f"Rollback to deployment {deployment_id} triggered",
        rollback_of_id=deployment_id,
    )


# ─────────────────────────── WebSocket endpoint ───────────────────────────────


@router.websocket("/live")
async def updater_live(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time deployment status streaming.

    Message types emitted by the server:
      - heartbeat     — idle state ping every 5s
      - status_update — state transition during a deployment
      - build_log     — one line of docker build output
      - completed     — deployment succeeded
      - error         — deployment failed
      - rollback      — rollback initiated

    Message types accepted from the client:
      - ping          — keepalive, server replies with pong
    """
    settings = get_settings()
    slot_key: Optional[str] = None

    try:
        slot_key = acquire_ws_slot(websocket, settings=settings, endpoint="updater-live")
    except HTTPException as exc:
        await websocket.close(code=1013, reason=str(exc.detail))
        return

    await websocket.accept()

    svc = _get_updater_service()
    await svc.register_ws(websocket)

    try:
        # Listen for messages (ping/pong keepalive) while keeping the connection open.
        # The service independently broadcasts to this client via _broadcast().
        while True:
            try:
                # Non-blocking receive with a generous timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                if isinstance(data, dict) and data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
            except asyncio.TimeoutError:
                # No client message received — send our own heartbeat
                try:
                    state_data = await svc.get_status()
                    await websocket.send_json({
                        "type": "heartbeat",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "state": state_data["state"],
                    })
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        await svc.unregister_ws(websocket)
        release_ws_slot(slot_key)

"""REST and WebSocket endpoints for model training orchestration."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from app.api.deps import (
    acquire_ws_slot,
    enforce_rate_limit,
    enforce_write_rate_limit,
    get_db,
    release_ws_slot,
)
from app.api.translate import reload_pipeline
from app.config import get_settings
from app.database import SessionLocal
from app.ml.utils import estimate_remaining_time
from app.schemas.training import TrainingMetrics, TrainingSession, TrainingSessionCreate
from app.services.model_service import ModelService
from app.services.training_service import normalize_training_config, training_service

router = APIRouter()
model_service = ModelService()


def _deployment_state(session) -> dict[str, object]:
    """Compute deployment readiness state from session metrics/config."""
    metrics = session.metrics or {}
    config = normalize_training_config(session.config or {})
    threshold = float(metrics.get("deploy_threshold", config.min_deploy_accuracy))
    final_val = metrics.get("final_val_accuracy")
    if final_val is None and session.status == "completed":
        final_val = metrics.get("val_accuracy")

    final_val_float = float(final_val) if final_val is not None else None
    deployment_ready = bool(
        metrics.get(
            "deployment_gate_passed",
            metrics.get(
                "deployment_ready",
                final_val_float is not None and final_val_float >= threshold,
            ),
        )
    )

    if session.status == "failed":
        next_action = "review_error"
    elif session.status != "completed":
        next_action = "wait"
    elif deployment_ready:
        next_action = "deploy"
    else:
        next_action = "collect_more_examples"

    candidate_action = str(metrics.get("recommended_next_action", next_action))
    if candidate_action not in {"deploy", "collect_more_examples", "wait", "review_error"}:
        candidate_action = next_action

    return {
        "deployment_ready": deployment_ready,
        "deploy_threshold": threshold,
        "final_val_accuracy": final_val_float,
        "recommended_next_action": candidate_action,
    }


def _serialize_session(session) -> TrainingSession:
    """Convert ORM training session into Pydantic response schema."""
    metrics = session.metrics or {}
    deploy_state = _deployment_state(session)
    return TrainingSession(
        id=session.id,
        sign_id=session.sign_id,
        mode=session.mode,
        status=session.status,
        progress=session.progress,
        config=normalize_training_config(session.config or {}),
        metrics=TrainingMetrics(
            loss=float(metrics.get("loss", 0.0)),
            accuracy=float(metrics.get("accuracy", 0.0)),
            val_accuracy=float(metrics.get("val_accuracy", 0.0)),
            macro_f1=float(metrics.get("macro_f1", 0.0)),
            target_sign_f1=(
                float(metrics["target_sign_f1"])
                if metrics.get("target_sign_f1") is not None
                else None
            ),
            open_set_fpr=(
                float(metrics["open_set_fpr"])
                if metrics.get("open_set_fpr") is not None
                else None
            ),
            latency_p95_ms=(
                float(metrics["latency_p95_ms"])
                if metrics.get("latency_p95_ms") is not None
                else None
            ),
            calibration_temperature=(
                float(metrics["calibration_temperature"])
                if metrics.get("calibration_temperature") is not None
                else None
            ),
            deployment_gate_passed=bool(metrics.get("deployment_gate_passed", False)),
        )
        if metrics
        else None,
        model_version_produced=session.model_version_produced,
        deployment_ready=bool(deploy_state["deployment_ready"]),
        deploy_threshold=float(deploy_state["deploy_threshold"]),
        final_val_accuracy=(
            float(deploy_state["final_val_accuracy"])
            if deploy_state["final_val_accuracy"] is not None
            else None
        ),
        recommended_next_action=str(deploy_state["recommended_next_action"]),
        started_at=session.started_at,
        completed_at=session.completed_at,
        error_message=session.error_message,
    )


@router.post(
    "/sessions",
    response_model=TrainingSession,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(enforce_rate_limit), Depends(enforce_write_rate_limit)],
)
def create_training_session(payload: TrainingSessionCreate, db: Session = Depends(get_db)) -> TrainingSession:
    """Queue a training session and start background execution."""
    if payload.mode == "few-shot" and payload.sign_id is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="sign_id is required for few-shot")
    session = training_service.create_session(db, payload)
    return _serialize_session(session)


@router.get("/sessions/{session_id}")
def get_training_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    """Fetch training session status and current metrics."""
    session = training_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    serialized = _serialize_session(session).model_dump()
    serialized["current_epoch"] = int((session.metrics or {}).get("current_epoch", 0))
    serialized["estimated_remaining"] = estimate_remaining_time(session.progress)
    return serialized


@router.get("/sessions", response_model=list[TrainingSession])
def list_training_sessions(db: Session = Depends(get_db)) -> list[TrainingSession]:
    """List training sessions history."""
    sessions = training_service.list_sessions(db)
    return [_serialize_session(item) for item in sessions]


@router.post("/sessions/{session_id}/stop", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(enforce_write_rate_limit)])
def stop_training_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    """Stop a running training session."""
    session = training_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    training_service.stop_session(session_id)
    return {"status": "stopping"}


@router.post("/sessions/{session_id}/deploy", dependencies=[Depends(enforce_write_rate_limit)])
def deploy_training_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    """Activate model produced by a completed training session."""
    session = training_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if session.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training session must be completed before deployment",
        )

    state = _deployment_state(session)
    if not bool(state["deployment_ready"]):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Model does not meet deployment threshold",
        )

    model_version = training_service.get_session_model(db, session)
    if not model_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version for session not found",
        )

    activated = model_service.activate(db, model_id=model_version.id)
    reload_pipeline()

    metrics = dict(session.metrics or {})
    metrics["recommended_next_action"] = "wait"
    session.metrics = metrics
    db.commit()

    return {
        "status": "deployed",
        "session_id": session_id,
        "active_model_id": activated.active_model_id,
        "version": activated.version,
    }


@router.websocket("/sessions/{session_id}/live")
async def training_session_live(websocket: WebSocket, session_id: str) -> None:
    """Stream session metrics every 500ms over WebSocket."""
    settings = get_settings()
    slot_key = None
    try:
        slot_key = acquire_ws_slot(websocket, settings=settings, endpoint="training-live")
    except HTTPException as exc:
        await websocket.close(code=1013, reason=str(exc.detail))
        return

    await websocket.accept()

    try:
        while True:
            db = SessionLocal()
            try:
                session = training_service.get_session(db, session_id)
                if not session:
                    await websocket.send_json({"error": "Session not found"})
                    break

                payload = {
                    "status": session.status,
                    "progress": session.progress,
                    "metrics": session.metrics or {},
                    "estimated_remaining": estimate_remaining_time(session.progress),
                    "error_message": session.error_message,
                    **_deployment_state(session),
                }
                await websocket.send_json(payload)
                if session.status in {"completed", "failed"}:
                    break
            finally:
                db.close()

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    finally:
        release_ws_slot(slot_key)

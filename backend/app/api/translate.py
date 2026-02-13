"""WebSocket endpoint for real-time sign translation streaming."""

from __future__ import annotations

from collections import Counter
import re
import time
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketState

from app.api.deps import acquire_ws_slot, enforce_ws_message_rate, release_ws_slot
from app.config import get_settings
from app.database import SessionLocal
from app.ml.pipeline import SignFlowInferencePipeline
from app.models.model_version import ModelVersion
from app.models.sign import Sign

router = APIRouter()
logger = structlog.get_logger(__name__)

# Global pipeline instance (loaded once at startup)
_global_pipeline: SignFlowInferencePipeline | None = None
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_-]+")


def _is_disconnect_runtime_error(error: RuntimeError) -> bool:
    """Detect Starlette runtime errors emitted after a websocket disconnect."""
    message = str(error)
    return (
        'disconnect message has been received' in message
        or 'close message has been sent' in message
    )


def _resolve_pipeline_labels(db: Session, active_model: ModelVersion | None) -> list[str]:
    """Resolve labels for inference using model metadata first, DB slugs as fallback."""
    if active_model and active_model.class_labels:
        labels = [label for label in active_model.class_labels if label and label != "NONE"]
        if labels:
            return labels

    return [item.slug for item in db.scalars(select(Sign).order_by(Sign.name.asc())).all()]


def _extract_sentence_tokens(sentence_buffer: str) -> list[str]:
    """Extract normalized sign tokens from sentence buffer."""
    if not sentence_buffer:
        return []
    return [token.lower() for token in _TOKEN_RE.findall(sentence_buffer)]


def _increment_usage_counts(tokens: list[str]) -> None:
    """Increment per-sign usage counts for newly emitted sentence tokens."""
    if not tokens:
        return

    counts = Counter(tokens)
    db = SessionLocal()
    try:
        signs = db.scalars(select(Sign).where(Sign.slug.in_(counts.keys()))).all()
        for sign in signs:
            sign.usage_count = int(sign.usage_count or 0) + int(counts.get(sign.slug, 0))
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("usage_count_increment_failed", error=str(exc))
    finally:
        db.close()


def get_or_create_pipeline() -> SignFlowInferencePipeline:
    """
    Get or create the global inference pipeline with active model.

    Returns:
        Initialized SignFlowInferencePipeline
    """
    global _global_pipeline

    if _global_pipeline is not None:
        return _global_pipeline

    logger.info("initializing_global_pipeline")

    # Get active model from database
    db = SessionLocal()
    try:
        active_model = db.scalar(
            select(ModelVersion)
            .where(ModelVersion.is_active.is_(True))
            .order_by(ModelVersion.created_at.desc())
        )
        labels = _resolve_pipeline_labels(db, active_model)

        if active_model and Path(active_model.file_path).exists():
            logger.info(
                "loading_active_model",
                version=active_model.version,
                path=active_model.file_path,
            )
            pipeline = SignFlowInferencePipeline(
                model_path=active_model.file_path,
                seq_len=30,
                confidence_threshold=0.7,
                device="cpu",
            )
            pipeline.set_labels(labels)

            logger.info("pipeline_initialized", num_labels=len(labels))
        else:
            logger.warning("no_active_model_found_using_fallback")
            # Create pipeline without model (will return NONE predictions)
            pipeline = SignFlowInferencePipeline(
                model_path=None,
                seq_len=30,
                confidence_threshold=0.7,
            )
            pipeline.set_labels(labels)

        _global_pipeline = pipeline
        return pipeline

    except Exception as e:
        logger.error("failed_to_initialize_pipeline", error=str(e), exc_info=True)
        # Fallback: create pipeline without model
        pipeline = SignFlowInferencePipeline(model_path=None, seq_len=30, confidence_threshold=0.7)
        try:
            labels = [item.slug for item in db.scalars(select(Sign).order_by(Sign.name.asc())).all()]
            pipeline.set_labels(labels)
        except Exception:
            pass
        _global_pipeline = pipeline
        return pipeline
    finally:
        db.close()


def reload_pipeline() -> None:
    """Force reload of the global pipeline (called after training completes)."""
    global _global_pipeline
    _global_pipeline = None
    logger.info("pipeline_reload_scheduled")


@router.websocket("/stream")
async def translate_stream(websocket: WebSocket) -> None:
    """
    Receive landmarks frames and return rolling predictions in real time.

    Protocol:
    - Client sends JSON: {"timestamp": float, "frame_idx": int, "hands": {...}, "pose": [...]}
    - Server responds: {"prediction": str, "confidence": float, "alternatives": [...], ...}
    """
    settings = get_settings()
    slot_key = None
    client_host = websocket.client.host if websocket.client else "unknown"
    try:
        slot_key = acquire_ws_slot(websocket, settings=settings, endpoint="translate-stream")
    except HTTPException as exc:
        await websocket.close(code=1013, reason=str(exc.detail))
        return

    await websocket.accept()

    # Get global pipeline (with active model loaded)
    try:
        pipeline = get_or_create_pipeline()
    except Exception as e:
        logger.error("failed_to_get_pipeline", error=str(e))
        await websocket.close(code=1011, reason="Failed to initialize inference pipeline")
        return

    # Reset pipeline state for this session
    pipeline.reset()

    frame_count = 0
    start_time = time.time()
    counted_token_total = 0

    logger.info("websocket_connection_established")

    try:
        while True:
            # Receive landmarks from client
            try:
                enforce_ws_message_rate(client_host, endpoint="translate-stream", settings=settings)
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except HTTPException:
                await websocket.close(code=1008, reason="WebSocket rate limit exceeded")
                break
            except RuntimeError as e:
                if _is_disconnect_runtime_error(e):
                    raise WebSocketDisconnect(code=1000) from e
                logger.warning("websocket_receive_failed", error=str(e))
                break
            except ValueError as e:
                logger.warning("invalid_json_received", error=str(e))
                continue
            except Exception as e:
                logger.warning("websocket_receive_failed", error=str(e))
                break

            # Validate payload structure
            if not isinstance(payload, dict):
                logger.warning("invalid_payload_format", type=type(payload).__name__)
                continue

            if "hands" not in payload and "pose" not in payload:
                logger.warning("missing_landmarks_in_payload")
                continue

            # Process frame
            frame_start = time.time()
            try:
                prediction = pipeline.process_frame(payload)

                # Calculate latency
                latency_ms = (time.time() - frame_start) * 1000
                frame_count += 1

                # Log performance metrics periodically
                if frame_count % 30 == 0:  # Every ~1 second at 30fps
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.debug(
                        "websocket_performance",
                        frame_count=frame_count,
                        fps=round(fps, 1),
                        avg_latency_ms=round(latency_ms, 1),
                    )

                # Update usage counts only when a new token is appended to sentence buffer.
                tokens = _extract_sentence_tokens(prediction.sentence_buffer)
                token_count = len(tokens)
                if token_count < counted_token_total:
                    counted_token_total = token_count
                elif token_count > counted_token_total:
                    new_tokens = tokens[counted_token_total:token_count]
                    _increment_usage_counts(new_tokens)
                    counted_token_total = token_count

                # Send prediction to client
                try:
                    await websocket.send_json(
                        {
                            "prediction": prediction.prediction,
                            "confidence": prediction.confidence,
                            "alternatives": prediction.alternatives,
                            "sentence_buffer": prediction.sentence_buffer,
                            "is_sentence_complete": prediction.is_sentence_complete,
                            "latency_ms": round(latency_ms, 1),
                        }
                    )
                except WebSocketDisconnect:
                    raise
                except RuntimeError as e:
                    if _is_disconnect_runtime_error(e):
                        raise WebSocketDisconnect(code=1000) from e
                    raise

                # Log warning if latency is too high
                if latency_ms > 100:
                    logger.warning(
                        "high_inference_latency",
                        latency_ms=latency_ms,
                        frame_count=frame_count,
                    )

            except WebSocketDisconnect:
                raise
            except RuntimeError as e:
                if _is_disconnect_runtime_error(e):
                    raise WebSocketDisconnect(code=1000) from e
                logger.error("frame_processing_failed", error=str(e), exc_info=True)
                # Send error response but keep connection alive
                await websocket.send_json(
                    {
                        "prediction": "NONE",
                        "confidence": 0.0,
                        "alternatives": [],
                        "sentence_buffer": "",
                        "is_sentence_complete": False,
                        "error": "Frame processing failed",
                    }
                )
            except Exception as e:
                logger.error("frame_processing_failed", error=str(e), exc_info=True)
                # Send error response but keep connection alive
                await websocket.send_json(
                    {
                        "prediction": "NONE",
                        "confidence": 0.0,
                        "alternatives": [],
                        "sentence_buffer": "",
                        "is_sentence_complete": False,
                        "error": "Frame processing failed",
                    }
                )

    except WebSocketDisconnect:
        logger.info(
            "websocket_disconnected",
            frame_count=frame_count,
            duration_sec=round(time.time() - start_time, 1),
        )
        return
    except Exception as e:
        logger.error("websocket_error", error=str(e), exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Internal server error")
        return
    finally:
        release_ws_slot(slot_key)

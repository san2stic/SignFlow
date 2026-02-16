"""WebSocket endpoint for real-time sign translation streaming."""

from __future__ import annotations

from collections import Counter
import re
import time
from pathlib import Path

import numpy as np
import structlog
import torch
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketState

from app.api.deps import acquire_ws_slot, enforce_ws_message_rate, release_ws_slot
from app.api.router_v2 import ModelRouter
from app.api.shadow_mode import ShadowModeEvaluator
from app.config import get_settings
from app.database import SessionLocal
from app.ml.active_learning import ActiveLearningQueue
from app.ml.metrics import get_metrics_collector
from app.ml.monitoring import DriftDetector
from app.ml.pipeline import SignFlowInferencePipeline
from app.ml.torchserve_client import TorchServeClient
from app.models.model_version import ModelVersion
from app.models.sign import Sign

router = APIRouter()
logger = structlog.get_logger(__name__)

# Global pipeline template (loaded once at startup; per-connection state is isolated)
_global_pipeline: SignFlowInferencePipeline | None = None
_global_torchserve_client: TorchServeClient | None = None
_global_drift_detector: DriftDetector | None = None
_global_model_router: ModelRouter | None = None
_global_model_router_config: tuple[float, str | None, bool, str | None] | None = None
_global_shadow_evaluator: ShadowModeEvaluator | None = None
_global_active_learning_queue: ActiveLearningQueue | None = None
_global_active_learning_config: tuple[str, float, int, int, float] | None = None
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
        labels = [label for label in active_model.class_labels if label]
        if labels:
            return labels

    return [item.slug for item in db.scalars(select(Sign).order_by(Sign.name.asc())).all()]


def _load_checkpoint_runtime_metadata(model_path: str) -> dict[str, object]:
    """Read lightweight runtime metadata from checkpoint."""
    try:
        checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        return {}
    config = checkpoint.get("config") or {}
    metadata = checkpoint.get("metadata") or {}
    return {
        "sequence_length": config.get("sequence_length"),
        "calibration_temperature": metadata.get("calibration_temperature"),
    }


def _extract_sentence_tokens(sentence_buffer: str) -> list[str]:
    """Extract normalized sign tokens from sentence buffer."""
    if not sentence_buffer:
        return []
    return [token.lower() for token in _TOKEN_RE.findall(sentence_buffer)]


def _to_optional_int(value: object) -> int | None:
    """Best-effort integer parsing for frame metadata."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_optional_float(value: object) -> float | None:
    """Best-effort float parsing for timestamp metadata."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    Get or create the global inference pipeline template with active model.

    Returns:
        Initialized SignFlowInferencePipeline
    """
    global _global_pipeline

    if _global_pipeline is not None:
        return _global_pipeline

    logger.info("initializing_global_pipeline_template")
    settings = get_settings()

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
            try:
                model_metadata = active_model.artifact_metadata or {}
            except Exception:  # noqa: BLE001
                model_metadata = {}
            calibration = model_metadata.get("calibration", {}) if isinstance(model_metadata, dict) else {}
            class_thresholds = (
                model_metadata.get("class_thresholds", {})
                if isinstance(model_metadata, dict)
                else {}
            )
            checkpoint_runtime = _load_checkpoint_runtime_metadata(active_model.file_path)
            sequence_length = int(
                checkpoint_runtime.get("sequence_length")
                or settings.translate_seq_len
            )
            calibration_temperature = calibration.get("temperature")
            if calibration_temperature is None:
                calibration_temperature = checkpoint_runtime.get("calibration_temperature")

            logger.info(
                "loading_active_model",
                version=active_model.version,
                path=active_model.file_path,
            )
            pipeline = SignFlowInferencePipeline(
                model_path=active_model.file_path,
                model_version=active_model.version,
                seq_len=max(8, min(256, sequence_length)),
                confidence_threshold=settings.translate_confidence_threshold,
                inference_num_views=settings.translate_inference_num_views,
                inference_temperature=settings.translate_inference_temperature,
                max_view_disagreement=settings.translate_max_view_disagreement,
                tta_enable_mirror=settings.translate_tta_enable_mirror,
                tta_enable_temporal_jitter=settings.translate_tta_enable_temporal_jitter,
                tta_enable_spatial_noise=settings.translate_tta_enable_spatial_noise,
                tta_temporal_jitter_ratio=settings.translate_tta_temporal_jitter_ratio,
                tta_spatial_noise_std=settings.translate_tta_spatial_noise_std,
                calibration_temperature=(
                    float(calibration_temperature)
                    if calibration_temperature is not None
                    else None
                ),
                class_thresholds=(
                    class_thresholds if isinstance(class_thresholds, dict) else {}
                ),
                device="cpu",
            )
            pipeline.set_labels(labels)

            logger.info("pipeline_template_initialized", num_labels=len(labels))
        else:
            logger.warning("no_active_model_found_using_fallback")
            # Create pipeline without model (will return NONE predictions)
            pipeline = SignFlowInferencePipeline(
                model_path=None,
                model_version="none",
                seq_len=settings.translate_seq_len,
                confidence_threshold=settings.translate_confidence_threshold,
                inference_num_views=settings.translate_inference_num_views,
                inference_temperature=settings.translate_inference_temperature,
                max_view_disagreement=settings.translate_max_view_disagreement,
                tta_enable_mirror=settings.translate_tta_enable_mirror,
                tta_enable_temporal_jitter=settings.translate_tta_enable_temporal_jitter,
                tta_enable_spatial_noise=settings.translate_tta_enable_spatial_noise,
                tta_temporal_jitter_ratio=settings.translate_tta_temporal_jitter_ratio,
                tta_spatial_noise_std=settings.translate_tta_spatial_noise_std,
            )
            pipeline.set_labels(labels)

        _global_pipeline = pipeline
        return pipeline

    except Exception as e:
        logger.error("failed_to_initialize_pipeline", error=str(e), exc_info=True)
        # Fallback: create pipeline without model
        pipeline = SignFlowInferencePipeline(
            model_path=None,
            model_version="none",
            seq_len=settings.translate_seq_len,
            confidence_threshold=settings.translate_confidence_threshold,
            inference_num_views=settings.translate_inference_num_views,
            inference_temperature=settings.translate_inference_temperature,
            max_view_disagreement=settings.translate_max_view_disagreement,
            tta_enable_mirror=settings.translate_tta_enable_mirror,
            tta_enable_temporal_jitter=settings.translate_tta_enable_temporal_jitter,
            tta_enable_spatial_noise=settings.translate_tta_enable_spatial_noise,
            tta_temporal_jitter_ratio=settings.translate_tta_temporal_jitter_ratio,
            tta_spatial_noise_std=settings.translate_tta_spatial_noise_std,
        )
        try:
            labels = [item.slug for item in db.scalars(select(Sign).order_by(Sign.name.asc())).all()]
            pipeline.set_labels(labels)
        except Exception:
            pass
        _global_pipeline = pipeline
        return pipeline
    finally:
        db.close()


def load_pipeline_for_model(model_id: str) -> SignFlowInferencePipeline | None:
    """Load a pipeline template for a specific model version id."""
    settings = get_settings()
    db = SessionLocal()
    try:
        model = db.get(ModelVersion, model_id)
        if model is None or not model.file_path or not Path(model.file_path).exists():
            return None

        labels = _resolve_pipeline_labels(db, model)
        try:
            model_metadata = model.artifact_metadata or {}
        except Exception:  # noqa: BLE001
            model_metadata = {}
        calibration = model_metadata.get("calibration", {}) if isinstance(model_metadata, dict) else {}
        class_thresholds = (
            model_metadata.get("class_thresholds", {})
            if isinstance(model_metadata, dict)
            else {}
        )
        checkpoint_runtime = _load_checkpoint_runtime_metadata(model.file_path)
        sequence_length = int(
            checkpoint_runtime.get("sequence_length")
            or settings.translate_seq_len
        )
        calibration_temperature = calibration.get("temperature")
        if calibration_temperature is None:
            calibration_temperature = checkpoint_runtime.get("calibration_temperature")

        pipeline = SignFlowInferencePipeline(
            model_path=model.file_path,
            model_version=model.version,
            seq_len=max(8, min(256, sequence_length)),
            confidence_threshold=settings.translate_confidence_threshold,
            inference_num_views=settings.translate_inference_num_views,
            inference_temperature=settings.translate_inference_temperature,
            max_view_disagreement=settings.translate_max_view_disagreement,
            tta_enable_mirror=settings.translate_tta_enable_mirror,
            tta_enable_temporal_jitter=settings.translate_tta_enable_temporal_jitter,
            tta_enable_spatial_noise=settings.translate_tta_enable_spatial_noise,
            tta_temporal_jitter_ratio=settings.translate_tta_temporal_jitter_ratio,
            tta_spatial_noise_std=settings.translate_tta_spatial_noise_std,
            calibration_temperature=(
                float(calibration_temperature)
                if calibration_temperature is not None
                else None
            ),
            class_thresholds=(
                class_thresholds if isinstance(class_thresholds, dict) else {}
            ),
            device="cpu",
        )
        pipeline.set_labels(labels)
        return pipeline
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failed_to_load_specific_model_pipeline",
            model_id=model_id,
            error=str(exc),
        )
        return None
    finally:
        db.close()


def get_or_create_model_router() -> ModelRouter:
    """Get or create canary/shadow-aware model router."""
    global _global_model_router, _global_model_router_config

    settings = get_settings()
    config_signature = (
        float(settings.canary_percentage),
        settings.canary_model_id,
        bool(settings.shadow_mode_enabled),
        settings.shadow_model_id,
    )

    if _global_model_router is not None and _global_model_router_config == config_signature:
        return _global_model_router

    _global_model_router = ModelRouter(
        production_provider=lambda: get_or_create_pipeline(),
        model_loader=lambda model_id: load_pipeline_for_model(model_id),
        canary_percentage=settings.canary_percentage,
        canary_model_id=settings.canary_model_id,
        shadow_mode_enabled=settings.shadow_mode_enabled,
        shadow_model_id=settings.shadow_model_id,
    )
    _global_model_router_config = config_signature
    return _global_model_router


def get_or_create_shadow_evaluator() -> ShadowModeEvaluator:
    """Get or create shared shadow-mode evaluator."""
    global _global_shadow_evaluator

    if _global_shadow_evaluator is not None:
        return _global_shadow_evaluator

    settings = get_settings()
    _global_shadow_evaluator = ShadowModeEvaluator(
        min_confidence=settings.shadow_min_confidence,
    )
    return _global_shadow_evaluator


def get_or_create_torchserve_client() -> TorchServeClient:
    """Get or create a shared TorchServe HTTP client."""
    global _global_torchserve_client

    if _global_torchserve_client is not None:
        return _global_torchserve_client

    settings = get_settings()
    timeout_seconds = max(0.1, float(settings.torchserve_timeout_ms) / 1000.0)
    _global_torchserve_client = TorchServeClient(
        base_url=settings.torchserve_url,
        timeout_seconds=timeout_seconds,
    )
    return _global_torchserve_client


def get_or_create_drift_detector() -> DriftDetector:
    """Get or create shared confidence drift detector."""
    global _global_drift_detector

    if _global_drift_detector is not None:
        return _global_drift_detector

    settings = get_settings()
    _global_drift_detector = DriftDetector(
        window_size=settings.drift_window_size,
        check_every=settings.drift_check_every,
        min_samples=settings.drift_min_samples,
        p_value_threshold=settings.drift_p_value_threshold,
        mean_shift_threshold=settings.drift_mean_shift_threshold,
    )
    return _global_drift_detector


def get_or_create_active_learning_queue() -> ActiveLearningQueue:
    """Get or create shared active-learning uncertainty queue."""
    global _global_active_learning_queue, _global_active_learning_config

    settings = get_settings()
    config_signature = (
        settings.active_learning_strategy,
        float(settings.active_learning_min_uncertainty),
        int(settings.active_learning_max_queue),
        int(settings.active_learning_top_n),
        float(settings.active_learning_cooldown_seconds),
    )

    if (
        _global_active_learning_queue is not None
        and _global_active_learning_config == config_signature
    ):
        return _global_active_learning_queue

    _global_active_learning_queue = ActiveLearningQueue(
        strategy=settings.active_learning_strategy,
        min_uncertainty=settings.active_learning_min_uncertainty,
        max_size=settings.active_learning_max_queue,
        top_n=settings.active_learning_top_n,
        cooldown_seconds=settings.active_learning_cooldown_seconds,
    )
    _global_active_learning_config = config_signature
    return _global_active_learning_queue


def reload_pipeline() -> None:
    """Force reload of the global pipeline (called after training completes)."""
    global _global_pipeline, _global_model_router, _global_model_router_config
    _global_pipeline = None
    _global_model_router = None
    _global_model_router_config = None
    logger.info("pipeline_reload_scheduled")


@router.get("/active-learning/queue")
def list_active_learning_queue(
    limit: int = Query(default=20, ge=1, le=500),
) -> dict[str, object]:
    """List top uncertain samples queued for active-learning annotation."""
    settings = get_settings()
    if not settings.active_learning_enabled:
        return {
            "enabled": False,
            "queue_size": 0,
            "strategy": settings.active_learning_strategy,
            "items": [],
        }

    queue = get_or_create_active_learning_queue()
    items = [sample.to_dict() for sample in queue.top_uncertain(limit=limit)]
    snapshot = queue.snapshot()
    return {
        "enabled": True,
        "queue_size": int(snapshot["queue_size"]),
        "strategy": snapshot["strategy"],
        "items": items,
    }


@router.post("/active-learning/queue/{sample_id}/resolve")
def resolve_active_learning_sample(sample_id: str) -> dict[str, object]:
    """Resolve/remove one active-learning candidate after annotation."""
    settings = get_settings()
    if not settings.active_learning_enabled:
        raise HTTPException(status_code=400, detail="Active learning is disabled")

    queue = get_or_create_active_learning_queue()
    sample = queue.resolve(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")

    return {"resolved": True, "sample": sample.to_dict()}


@router.websocket("/stream")
async def translate_stream(websocket: WebSocket) -> None:
    """
    Receive landmarks frames and return rolling predictions in real time.

    Protocol:
    - Client sends JSON: {"timestamp": float, "frame_idx": int, "hands": {...}, "pose": [...]}
    - Server responds: {"prediction": str, "confidence": float, "alternatives": [...], ...}
    """
    settings = get_settings()
    use_torchserve = bool(settings.use_torchserve)
    metrics_collector = get_metrics_collector()
    drift_detector = get_or_create_drift_detector() if settings.drift_detection_enabled else None
    shadow_evaluator = get_or_create_shadow_evaluator() if settings.shadow_mode_enabled else None
    active_learning_queue = (
        get_or_create_active_learning_queue()
        if settings.active_learning_enabled
        else None
    )
    torchserve_client: TorchServeClient | None = None
    shadow_pipeline: SignFlowInferencePipeline | None = None
    route = "production"
    slot_key = None
    client_host = websocket.client.host if websocket.client else "unknown"
    try:
        slot_key = acquire_ws_slot(websocket, settings=settings, endpoint="translate-stream")
    except HTTPException as exc:
        await websocket.close(code=1013, reason=str(exc.detail))
        return

    await websocket.accept()

    # Build an isolated per-connection pipeline from the shared template.
    try:
        model_router = get_or_create_model_router()
        routing_session = model_router.spawn_sessions()
        pipeline = routing_session.primary
        shadow_pipeline = routing_session.shadow
        route = routing_session.route
    except Exception as e:
        logger.error("failed_to_get_pipeline", error=str(e))
        await websocket.close(code=1011, reason="Failed to initialize inference pipeline")
        return

    if settings.inference_metrics_enabled:
        metrics_collector.record_routing_decision(route=route)

    if route == "canary" and use_torchserve:
        # TorchServe path currently serves one active model; keep canary honest by local inference.
        logger.info("torchserve_disabled_for_canary_route")
        use_torchserve = False

    if use_torchserve:
        try:
            torchserve_client = get_or_create_torchserve_client()
            logger.info("torchserve_enabled_for_translate_stream", url=settings.torchserve_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "torchserve_client_init_failed_fallback_local",
                error=str(exc),
            )
            metrics_collector.record_torchserve_error(reason="client_init_failed")
            use_torchserve = False

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
                if use_torchserve and torchserve_client is not None and hasattr(pipeline, "process_frame_async"):
                    async def _infer_window_async(
                        window: np.ndarray,
                    ) -> tuple[str, float, list[dict[str, float]]]:
                        try:
                            return await torchserve_client.predict(window)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "torchserve_inference_failed_fallback_local",
                                error=str(exc),
                            )
                            metrics_collector.record_torchserve_error(reason="inference_error")
                            return pipeline._infer_window(window)

                    prediction = await pipeline.process_frame_async(
                        payload,
                        infer_window_async=_infer_window_async,
                    )
                else:
                    prediction = pipeline.process_frame(payload)

                # Calculate latency
                latency_ms = (time.time() - frame_start) * 1000
                frame_count += 1

                if settings.inference_metrics_enabled and prediction.decision_diagnostics is not None:
                    metrics_collector.record_inference(
                        model_version=getattr(pipeline, "model_version", "unknown"),
                        sign=prediction.prediction,
                        confidence=prediction.confidence,
                        latency_seconds=latency_ms / 1000.0,
                    )

                    if drift_detector is not None and prediction.prediction != "RECORDING":
                        drift_result = drift_detector.record(prediction.confidence)
                        if drift_result.checked and drift_result.drift_detected:
                            metrics_collector.record_drift_alert(kind="confidence")
                            logger.warning(
                                "confidence_drift_detected",
                                samples=drift_result.samples,
                                reason=drift_result.reason,
                                p_value=drift_result.p_value,
                                statistic=drift_result.statistic,
                                reference_mean=drift_result.reference_mean,
                                current_mean=drift_result.current_mean,
                            )

                if shadow_pipeline is not None:
                    try:
                        shadow_prediction = shadow_pipeline.process_frame(payload)
                        if shadow_evaluator is not None:
                            comparison = shadow_evaluator.compare(
                                primary=prediction,
                                shadow=shadow_prediction,
                            )
                            if settings.inference_metrics_enabled:
                                metrics_collector.record_shadow_comparison(
                                    route=route,
                                    disagreed=comparison.disagreed,
                                )
                            if comparison.high_confidence_disagreement:
                                logger.warning(
                                    "shadow_high_confidence_disagreement",
                                    route=route,
                                    primary=comparison.primary_prediction,
                                    shadow=comparison.shadow_prediction,
                                    confidence_gap=round(comparison.confidence_gap, 4),
                                )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("shadow_mode_inference_failed", error=str(exc))

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

                active_learning_hint: dict[str, object] | None = None
                if active_learning_queue is not None:
                    sample = active_learning_queue.consider(
                        prediction=prediction.prediction,
                        confidence=prediction.confidence,
                        alternatives=prediction.alternatives,
                        sentence_buffer=prediction.sentence_buffer,
                        frame_idx=_to_optional_int(payload.get("frame_idx")),
                        timestamp=_to_optional_float(payload.get("timestamp")),
                        route=route,
                        model_version=str(getattr(pipeline, "model_version", "unknown")),
                    )
                    if sample is not None:
                        active_learning_hint = {
                            "queued": True,
                            "sample_id": sample.id,
                            "uncertainty": round(sample.uncertainty, 4),
                            "strategy": sample.strategy,
                        }

                # Send prediction to client
                try:
                    await websocket.send_json(
                        {
                            "prediction": prediction.prediction,
                            "confidence": prediction.confidence,
                            "alternatives": prediction.alternatives,
                            "sentence_buffer": prediction.sentence_buffer,
                            "is_sentence_complete": prediction.is_sentence_complete,
                            "decision_diagnostics": prediction.decision_diagnostics,
                            "active_learning": active_learning_hint,
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

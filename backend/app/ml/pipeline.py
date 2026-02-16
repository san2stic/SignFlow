"""Inference pipeline for real-time sign translation from landmarks."""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import structlog
import torch

from app.ml.features import FrameLandmarks, normalize_landmarks
from app.ml.model import SignTransformer
from app.ml.tta import TTAConfig, TTAGenerator
from app.ml.trainer import load_model_checkpoint

logger = structlog.get_logger(__name__)

# Optional ONNX Runtime import
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.debug("onnxruntime_not_available", msg="ONNXRuntime not installed")


class InferenceState(Enum):
    """States for the inference state machine."""

    IDLE = "idle"
    RECORDING = "recording"
    INFERRING = "inferring"


@dataclass
class Prediction:
    """Prediction payload returned by translation pipeline."""

    prediction: str
    confidence: float
    alternatives: list[dict[str, float]]
    sentence_buffer: str
    is_sentence_complete: bool
    decision_diagnostics: dict[str, object] | None = None


class SignFlowInferencePipeline:
    """Sliding-window inference with confidence threshold and temporal smoothing."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        model_version: str | None = None,
        seq_len: int = 64,
        confidence_threshold: float = 0.7,
        smoothing_window: int = 5,
        min_hand_visibility: float = 0.2,
        min_prediction_margin: float = 0.1,
        min_motion_energy: float = 0.003,
        device: str = "cpu",
        max_buffer_frames: int = 180,
        rest_frames_threshold: int = 10,
        motion_start_threshold: float = 0.005,
        min_recording_frames: int = 15,
        pre_roll_frames: int = 4,
        frontend_confidence_floor: float = 0.35,
        inference_num_views: int = 1,
        inference_temperature: float = 1.0,
        max_view_disagreement: float = 0.35,
        tta_enable_mirror: bool = True,
        tta_enable_temporal_jitter: bool = True,
        tta_enable_spatial_noise: bool = True,
        tta_temporal_jitter_ratio: float = 0.05,
        tta_spatial_noise_std: float = 0.005,
        calibration_temperature: float | None = None,
        class_thresholds: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to model checkpoint. If None, pipeline runs without model.
            seq_len: Target sequence length for model input (default: 64 frames)
            confidence_threshold: Minimum confidence to accept prediction (default: 0.7)
            smoothing_window: Temporal smoothing window size in frames.
            min_hand_visibility: Minimum rolling hand visibility required to infer.
            min_prediction_margin: Minimum top1-top2 probability margin before accepting.
            min_motion_energy: Minimum hand motion level for stable confidence.
            device: Device to run inference on ("cpu" or "cuda")
            max_buffer_frames: Maximum frames to buffer before forcing inference.
            rest_frames_threshold: Consecutive low-motion frames to detect sign end.
            motion_start_threshold: Motion energy threshold to start recording.
            min_recording_frames: Minimum frames recorded before allowing sign end.
            pre_roll_frames: Number of frames to keep before motion onset.
            frontend_confidence_floor: Confidence floor for frontend landmarks quality.
            inference_num_views: Number of temporal views for test-time ensembling.
            inference_temperature: Softmax temperature used for calibrated probabilities.
            max_view_disagreement: Max tolerated view disagreement before confidence penalty.
            tta_enable_mirror: Whether to include left/right mirrored views.
            tta_enable_temporal_jitter: Whether to include temporal speed perturbations.
            tta_enable_spatial_noise: Whether to include spatial landmark noise.
            tta_temporal_jitter_ratio: Max relative speed perturbation for temporal jitter.
            tta_spatial_noise_std: Gaussian std for spatial perturbation.
        """
        self.seq_len = seq_len
        self.model_version = str(model_version or "unknown")
        self.confidence_threshold = confidence_threshold
        self.min_hand_visibility = min_hand_visibility
        self.min_prediction_margin = min_prediction_margin
        self.min_motion_energy = min_motion_energy
        self.pre_roll_frames = max(0, pre_roll_frames)
        self.frontend_confidence_floor = float(np.clip(frontend_confidence_floor, 0.0, 1.0))
        self.inference_num_views = max(1, int(inference_num_views))
        self.inference_temperature = float(max(0.1, inference_temperature))
        self.max_view_disagreement = float(max(1e-6, max_view_disagreement))
        self.tta_enable_mirror = bool(tta_enable_mirror)
        self.tta_enable_temporal_jitter = bool(tta_enable_temporal_jitter)
        self.tta_enable_spatial_noise = bool(tta_enable_spatial_noise)
        self.tta_temporal_jitter_ratio = float(np.clip(tta_temporal_jitter_ratio, 0.0, 0.3))
        self.tta_spatial_noise_std = float(max(0.0, tta_spatial_noise_std))
        self._tta_generator = TTAGenerator(
            TTAConfig(
                num_views=self.inference_num_views,
                enable_mirror=self.tta_enable_mirror,
                enable_temporal_jitter=self.tta_enable_temporal_jitter,
                enable_spatial_noise=self.tta_enable_spatial_noise,
                temporal_jitter_ratio=self.tta_temporal_jitter_ratio,
                spatial_noise_std=self.tta_spatial_noise_std,
            )
        )
        self.calibration_temperature = (
            float(max(0.1, calibration_temperature))
            if calibration_temperature is not None
            else None
        )
        self.class_thresholds = {
            str(key): float(np.clip(value, 0.0, 1.0))
            for key, value in (class_thresholds or {}).items()
        }
        self.device = torch.device(device)
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=max_buffer_frames)
        self.hand_visibility_history: deque[float] = deque(maxlen=seq_len)
        self.motion_history: deque[float] = deque(maxlen=seq_len)
        self.prediction_history: deque[tuple[str, float]] = deque(maxlen=max(3, smoothing_window))
        self.labels: list[str] = []
        self.sentence_tokens: list[str] = []
        self._current_motion_energy = 0.0
        self._decision_counters: dict[str, int] = {
            "total_inferences": 0,
            "accepted": 0,
            "rejected_total": 0,
            "rejected_by_confidence_threshold": 0,
            "rejected_by_adaptive_threshold": 0,
            "rejected_by_class_threshold": 0,
            "rejected_by_label_none": 0,
            "rejected_by_calibration": 0,
            "rejected_by_margin": 0,
            "rejected_by_motion": 0,
            "rejected_by_tta_disagreement": 0,
        }
        self._last_decision_trace: dict[str, object] = {
            "status": "idle",
            "reason": "not_started",
        }

        # State machine attributes
        self.max_buffer_frames = max_buffer_frames
        self.rest_frames_threshold = rest_frames_threshold
        self.motion_start_threshold = motion_start_threshold
        self.min_recording_frames = min_recording_frames
        self.state = InferenceState.IDLE
        self._rest_frame_count = 0
        self._recording_frame_count = 0
        self._latest_frontend_confidence = 1.0

        # Load model if path provided
        self.model: SignTransformer | None = None
        self.onnx_session: ort.InferenceSession | None = None
        self.use_onnx = False
        if model_path:
            self.load_model(model_path)

    def set_calibration(
        self,
        *,
        calibration_temperature: float | None = None,
        class_thresholds: dict[str, float] | None = None,
    ) -> None:
        """Configure optional calibration temperature and per-class thresholds."""
        if calibration_temperature is not None:
            self.calibration_temperature = float(max(0.1, calibration_temperature))
        if class_thresholds is not None:
            self.class_thresholds = {
                str(key): float(np.clip(value, 0.0, 1.0))
                for key, value in class_thresholds.items()
            }

    def load_model(self, model_path: str | Path) -> None:
        """
        Load model from checkpoint (supports both PyTorch .pt and ONNX .onnx).

        Args:
            model_path: Path to .pt or .onnx checkpoint file
        """
        model_path = Path(model_path)

        # Check if ONNX model
        if model_path.suffix == ".onnx":
            self._load_onnx_model(model_path)
        else:
            self._load_pytorch_model(model_path)

    def _load_pytorch_model(self, model_path: Path) -> None:
        """Load PyTorch model from checkpoint."""
        try:
            logger.info("loading_pytorch_model", path=str(model_path))
            self.model = load_model_checkpoint(str(model_path), device=str(self.device))
            self.model.to(self.device)
            self.model.set_inference_mode()
            self.use_onnx = False
            self.onnx_session = None
            logger.info(
                "pytorch_model_loaded_successfully",
                num_classes=self.model.num_classes,
                device=str(self.device),
            )
        except Exception as e:
            logger.error("failed_to_load_pytorch_model", path=str(model_path), error=str(e))
            self.model = None
            raise

    def _load_onnx_model(self, model_path: Path) -> None:
        """Load ONNX model for inference."""
        if not ONNX_AVAILABLE:
            logger.error(
                "onnx_load_failed",
                reason="onnxruntime_not_installed",
                path=str(model_path)
            )
            raise ImportError("onnxruntime is required for ONNX model inference")

        try:
            logger.info("loading_onnx_model", path=str(model_path))

            # Determine execution providers based on device
            if self.device.type == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            # Create ONNX Runtime session
            self.onnx_session = ort.InferenceSession(
                str(model_path),
                providers=providers
            )

            # Clear PyTorch model
            self.model = None
            self.use_onnx = True

            # Get model info
            input_shape = self.onnx_session.get_inputs()[0].shape
            output_shape = self.onnx_session.get_outputs()[0].shape

            logger.info(
                "onnx_model_loaded_successfully",
                path=str(model_path),
                input_shape=input_shape,
                output_shape=output_shape,
                providers=self.onnx_session.get_providers(),
            )

        except Exception as e:
            logger.error("failed_to_load_onnx_model", path=str(model_path), error=str(e))
            self.onnx_session = None
            raise

    def set_labels(self, labels: list[str]) -> None:
        """
        Configure active vocabulary labels for model outputs.

        Args:
            labels: List of sign labels (class names)
        """
        cleaned = [str(label) for label in labels if label]

        if self.model is None:
            self.labels = cleaned
            logger.debug("labels_set", num_labels=len(self.labels))
            return

        expected = int(self.model.num_classes)
        aligned = cleaned
        if len(aligned) != expected:
            logger.warning(
                "label_count_mismatch",
                expected=expected,
                provided=len(aligned),
            )
            aligned = aligned[:expected]
            while len(aligned) < expected:
                aligned.append(f"class_{len(aligned)}")

        self.labels = aligned
        logger.debug("labels_set", num_labels=len(self.labels))

    def spawn_session(self) -> SignFlowInferencePipeline:
        """Create an isolated per-session pipeline sharing only immutable runtime artifacts."""
        session_pipeline = SignFlowInferencePipeline(
            model_path=None,
            model_version=self.model_version,
            seq_len=self.seq_len,
            confidence_threshold=self.confidence_threshold,
            smoothing_window=int(self.prediction_history.maxlen or 5),
            min_hand_visibility=self.min_hand_visibility,
            min_prediction_margin=self.min_prediction_margin,
            min_motion_energy=self.min_motion_energy,
            device=str(self.device),
            max_buffer_frames=self.max_buffer_frames,
            rest_frames_threshold=self.rest_frames_threshold,
            motion_start_threshold=self.motion_start_threshold,
            min_recording_frames=self.min_recording_frames,
            pre_roll_frames=self.pre_roll_frames,
            frontend_confidence_floor=self.frontend_confidence_floor,
            inference_num_views=self.inference_num_views,
            inference_temperature=self.inference_temperature,
            max_view_disagreement=self.max_view_disagreement,
            tta_enable_mirror=self.tta_enable_mirror,
            tta_enable_temporal_jitter=self.tta_enable_temporal_jitter,
            tta_enable_spatial_noise=self.tta_enable_spatial_noise,
            tta_temporal_jitter_ratio=self.tta_temporal_jitter_ratio,
            tta_spatial_noise_std=self.tta_spatial_noise_std,
            calibration_temperature=self.calibration_temperature,
            class_thresholds=dict(self.class_thresholds),
        )
        session_pipeline.model = self.model
        session_pipeline.onnx_session = self.onnx_session
        session_pipeline.use_onnx = self.use_onnx
        session_pipeline.set_labels(list(self.labels))
        return session_pipeline

    def process_frame(self, payload: dict) -> Prediction:
        """Process a landmarks frame through the state machine."""
        frame = FrameLandmarks(
            left_hand=payload.get("hands", {}).get("left", []) or [],
            right_hand=payload.get("hands", {}).get("right", []) or [],
            pose=payload.get("pose", []) or [],
            face=payload.get("face", []) or [],
        )

        # Use metadata if available (from improved frontend detection)
        metadata = payload.get("metadata", {})
        frontend_confidence = self._resolve_frontend_confidence(metadata)
        self._latest_frontend_confidence = frontend_confidence

        # Log frontend confidence for monitoring (optional)
        if frontend_confidence < 0.3:
            logger.debug(
                "low_frontend_confidence",
                confidence=round(frontend_confidence, 3),
                left_visible=metadata.get("leftHandVisible", False),
                right_visible=metadata.get("rightHandVisible", False),
            )

        raw_hand_visibility = self._compute_hand_visibility(frame)
        hand_visibility = self._blend_hand_visibility(raw_hand_visibility, frontend_confidence)
        self.hand_visibility_history.append(hand_visibility)

        features = normalize_landmarks(frame, include_face=False)
        self.frame_buffer.append(features)
        self._current_motion_energy = self._compute_motion_energy()
        self.motion_history.append(self._current_motion_energy)

        # State machine
        if self.state == InferenceState.IDLE:
            if (self._current_motion_energy > self.motion_start_threshold
                    and hand_visibility > self.min_hand_visibility):
                if self.pre_roll_frames > 0:
                    context_frames = list(self.frame_buffer)[-self.pre_roll_frames:]
                    self.frame_buffer.clear()
                    self.frame_buffer.extend(context_frames)
                else:
                    self.frame_buffer.clear()
                    self.frame_buffer.append(features)
                self.state = InferenceState.RECORDING
                self._recording_frame_count = 1
                self._rest_frame_count = 0
            return self._idle_prediction()

        if self.state == InferenceState.RECORDING:
            self._recording_frame_count += 1

            is_resting = self._current_motion_energy < self.motion_start_threshold
            if is_resting:
                self._rest_frame_count += 1
            else:
                self._rest_frame_count = 0

            sign_ended = (
                (self._rest_frame_count >= self.rest_frames_threshold
                 and self._recording_frame_count >= self.min_recording_frames)
                or len(self.frame_buffer) >= self.max_buffer_frames
            )

            if sign_ended:
                return self._infer_complete_sign()

            return self._recording_prediction()

        # INFERRING state fallback
        self.state = InferenceState.IDLE
        return self._idle_prediction()

    async def process_frame_async(
        self,
        payload: dict,
        *,
        infer_window_async: Callable[
            [np.ndarray], Awaitable[tuple[str, float, list[dict[str, float]]]]
        ] | None = None,
    ) -> Prediction:
        """
        Async variant of process_frame.

        Args:
            payload: Frame landmarks payload
            infer_window_async: Optional async backend used at sign-end inference.
                If omitted, falls back to local model inference.
        """
        frame = FrameLandmarks(
            left_hand=payload.get("hands", {}).get("left", []) or [],
            right_hand=payload.get("hands", {}).get("right", []) or [],
            pose=payload.get("pose", []) or [],
            face=payload.get("face", []) or [],
        )

        metadata = payload.get("metadata", {})
        frontend_confidence = self._resolve_frontend_confidence(metadata)
        self._latest_frontend_confidence = frontend_confidence

        if frontend_confidence < 0.3:
            logger.debug(
                "low_frontend_confidence",
                confidence=round(frontend_confidence, 3),
                left_visible=metadata.get("leftHandVisible", False),
                right_visible=metadata.get("rightHandVisible", False),
            )

        raw_hand_visibility = self._compute_hand_visibility(frame)
        hand_visibility = self._blend_hand_visibility(raw_hand_visibility, frontend_confidence)
        self.hand_visibility_history.append(hand_visibility)

        features = normalize_landmarks(frame, include_face=False)
        self.frame_buffer.append(features)
        self._current_motion_energy = self._compute_motion_energy()
        self.motion_history.append(self._current_motion_energy)

        if self.state == InferenceState.IDLE:
            if (
                self._current_motion_energy > self.motion_start_threshold
                and hand_visibility > self.min_hand_visibility
            ):
                if self.pre_roll_frames > 0:
                    context_frames = list(self.frame_buffer)[-self.pre_roll_frames:]
                    self.frame_buffer.clear()
                    self.frame_buffer.extend(context_frames)
                else:
                    self.frame_buffer.clear()
                    self.frame_buffer.append(features)
                self.state = InferenceState.RECORDING
                self._recording_frame_count = 1
                self._rest_frame_count = 0
            return self._idle_prediction()

        if self.state == InferenceState.RECORDING:
            self._recording_frame_count += 1

            is_resting = self._current_motion_energy < self.motion_start_threshold
            if is_resting:
                self._rest_frame_count += 1
            else:
                self._rest_frame_count = 0

            sign_ended = (
                (
                    self._rest_frame_count >= self.rest_frames_threshold
                    and self._recording_frame_count >= self.min_recording_frames
                )
                or len(self.frame_buffer) >= self.max_buffer_frames
            )

            if sign_ended:
                return await self._infer_complete_sign_async(
                    infer_window_async=infer_window_async
                )

            return self._recording_prediction()

        self.state = InferenceState.IDLE
        return self._idle_prediction()

    def _idle_prediction(self) -> Prediction:
        """Return a neutral prediction for the IDLE state."""
        return Prediction(
            prediction="NONE",
            confidence=0.0,
            alternatives=[],
            sentence_buffer=" ".join(self.sentence_tokens),
            is_sentence_complete=False,
        )

    def _recording_prediction(self) -> Prediction:
        """Return a recording-in-progress prediction."""
        return Prediction(
            prediction="RECORDING",
            confidence=0.0,
            alternatives=[],
            sentence_buffer=" ".join(self.sentence_tokens),
            is_sentence_complete=False,
        )

    def _infer_complete_sign(self) -> Prediction:
        """Run inference on the complete recorded sign."""
        enriched = self._prepare_enriched_window()
        if enriched is None:
            self.state = InferenceState.IDLE
            return self._idle_prediction()

        predicted_label, predicted_confidence, alternatives = self._infer_window(enriched)
        return self._build_prediction_response(
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
            alternatives=alternatives,
        )

    async def _infer_complete_sign_async(
        self,
        *,
        infer_window_async: Callable[
            [np.ndarray], Awaitable[tuple[str, float, list[dict[str, float]]]]
        ] | None = None,
    ) -> Prediction:
        """Run inference on the complete recorded sign using an optional async backend."""
        enriched = self._prepare_enriched_window()
        if enriched is None:
            self.state = InferenceState.IDLE
            return self._idle_prediction()

        if infer_window_async is None:
            predicted_label, predicted_confidence, alternatives = self._infer_window(enriched)
        else:
            predicted_label, predicted_confidence, alternatives = await infer_window_async(enriched)

        return self._build_prediction_response(
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
            alternatives=alternatives,
        )

    def _prepare_enriched_window(self) -> np.ndarray | None:
        """Build an enriched feature window from the buffered raw landmarks."""
        from app.ml.dataset import temporal_resample
        from app.ml.feature_engineering import compute_enriched_features

        if not self.frame_buffer:
            return None

        buffer_array = np.stack(list(self.frame_buffer), axis=0)
        trimmed = self._trim_recording_window(buffer_array, trailing_rest=self._rest_frame_count)
        resampled = temporal_resample(trimmed, target_len=self.seq_len)
        return compute_enriched_features(resampled)

    def _normalize_alternatives(self, alternatives: list[dict[str, float]] | None) -> list[dict[str, float]]:
        """Normalize alternatives payload shape and numeric confidence."""
        normalized: list[dict[str, float]] = []
        for alternative in alternatives or []:
            if not isinstance(alternative, dict):
                continue
            sign = alternative.get("sign") or alternative.get("prediction")
            if not sign:
                continue
            try:
                confidence = float(alternative.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            normalized.append({"sign": str(sign), "confidence": round(confidence, 3)})
        return normalized

    def _build_prediction_response(
        self,
        *,
        predicted_label: str,
        predicted_confidence: float,
        alternatives: list[dict[str, float]] | None,
    ) -> Prediction:
        """Apply post-filters, reset recording state, and produce final response payload."""
        normalized_label = str(predicted_label or "NONE")
        normalized_confidence = float(predicted_confidence or 0.0)
        normalized_alternatives = self._normalize_alternatives(alternatives)

        adaptive_threshold = self._adaptive_threshold()
        label_threshold = self._threshold_for_label(normalized_label)
        effective_threshold = max(adaptive_threshold, label_threshold)
        filtered_label, filtered_confidence, filter_trace = self._apply_prediction_filters(
            prediction=normalized_label,
            confidence=normalized_confidence,
            threshold=effective_threshold,
        )

        decision_trace = {
            "prediction_before_filters": self._last_decision_trace,
            "adaptive_threshold": round(float(adaptive_threshold), 4),
            "class_threshold": round(float(label_threshold), 4),
            "effective_threshold": round(float(effective_threshold), 4),
            "filter_trace": filter_trace,
            "counters": dict(self._decision_counters),
        }

        self.frame_buffer.clear()
        self._rest_frame_count = 0
        self._recording_frame_count = 0
        self.state = InferenceState.IDLE

        if filtered_label != "NONE" and filtered_confidence >= effective_threshold:
            if not self.sentence_tokens or self.sentence_tokens[-1] != filtered_label:
                self.sentence_tokens.append(filtered_label)

        sentence = " ".join(self.sentence_tokens)
        return Prediction(
            prediction=filtered_label,
            confidence=filtered_confidence,
            alternatives=normalized_alternatives,
            sentence_buffer=sentence,
            is_sentence_complete=False,
            decision_diagnostics=decision_trace,
        )

    def reset(self) -> None:
        """Reset temporal state for a new translation session."""
        self.frame_buffer.clear()
        self.hand_visibility_history.clear()
        self.motion_history.clear()
        self.prediction_history.clear()
        self.sentence_tokens.clear()
        self._current_motion_energy = 0.0
        self.state = InferenceState.IDLE
        self._rest_frame_count = 0
        self._recording_frame_count = 0
        self._latest_frontend_confidence = 1.0
        for key in list(self._decision_counters.keys()):
            self._decision_counters[key] = 0
        self._last_decision_trace = {
            "status": "idle",
            "reason": "reset",
        }

    def snapshot_decision_diagnostics(self) -> dict[str, object]:
        """Expose serializable decision diagnostics for audit and tests."""
        return {
            "counters": dict(self._decision_counters),
            "last_decision_trace": dict(self._last_decision_trace),
            "state": self.state.value,
            "buffer_frames": int(len(self.frame_buffer)),
        }

    def _infer_window(self, window: np.ndarray) -> tuple[str, float, list[dict[str, float]]]:
        """
        Run PyTorch model inference over one sequence window.

        Args:
            window: Numpy array of shape [seq_len, num_features]

        Returns:
            Tuple of (prediction_label, confidence, alternatives)
        """
        # If no model loaded, return NONE
        if self.model is None or len(self.labels) == 0:
            return "NONE", 0.0, []

        try:
            probs, view_disagreement = self._infer_probabilities(window)
            self._decision_counters["total_inferences"] += 1

            # Get top predictions
            top_k = min(4, len(probs))  # Get top 4 predictions
            top_indices = np.argsort(probs)[-top_k:][::-1]

            if len(self.labels) != len(probs):
                logger.warning(
                    "runtime_label_mismatch",
                    model_classes=len(probs),
                    labels=len(self.labels),
                )
                runtime_labels = self.labels[: len(probs)]
                while len(runtime_labels) < len(probs):
                    runtime_labels.append(f"class_{len(runtime_labels)}")
            else:
                runtime_labels = self.labels

            # Main prediction
            pred_idx = int(top_indices[0])
            raw_confidence = float(probs[pred_idx])

            # Map index to label
            if pred_idx < len(runtime_labels):
                prediction = runtime_labels[pred_idx]
            else:
                prediction = "NONE"
                raw_confidence = 0.0

            # Alternatives (top 3 excluding main prediction)
            alternatives = []
            for idx in top_indices[1:top_k]:
                if idx < len(runtime_labels):
                    alt_label = runtime_labels[idx]
                    if alt_label in {"NONE", "[NONE]"}:
                        continue
                    alt_conf = float(probs[idx])
                    alternatives.append({
                        "sign": alt_label,
                        "confidence": round(alt_conf, 3)
                    })

            calibrated_confidence = self._calibrate_confidence(
                probs=probs,
                top_indices=top_indices,
                raw_confidence=raw_confidence,
                disagreement=view_disagreement,
            )

            # If confidence margin is too low, treat prediction as NONE.
            if calibrated_confidence <= 0.0:
                self._decision_counters["rejected_by_calibration"] += 1
                self._decision_counters["rejected_total"] += 1
                self._last_decision_trace = {
                    "status": "rejected",
                    "stage": "calibration",
                    "reason": "calibrated_confidence_non_positive",
                    "raw_confidence": round(raw_confidence, 4),
                    "calibrated_confidence": round(calibrated_confidence, 4),
                    "view_disagreement": round(float(view_disagreement), 6),
                }
                return "NONE", 0.0, alternatives

            if prediction in {"NONE", "[NONE]"}:
                self._decision_counters["rejected_by_label_none"] += 1
                self._decision_counters["rejected_total"] += 1
                self._last_decision_trace = {
                    "status": "rejected",
                    "stage": "labeling",
                    "reason": "model_predicted_none_class",
                    "calibrated_confidence": round(calibrated_confidence, 4),
                }
                return "NONE", round(calibrated_confidence, 3), alternatives

            self._last_decision_trace = {
                "status": "candidate",
                "stage": "post_inference",
                "prediction": prediction,
                "raw_confidence": round(raw_confidence, 4),
                "calibrated_confidence": round(calibrated_confidence, 4),
                "view_disagreement": round(float(view_disagreement), 6),
                "top2_margin": round(
                    float(raw_confidence - (float(probs[top_indices[1]]) if len(top_indices) > 1 else 0.0)),
                    4,
                ),
            }

            return prediction, round(calibrated_confidence, 3), alternatives

        except Exception as e:
            logger.error("inference_failed", error=str(e), exc_info=True)
            return "NONE", 0.0, []

    def _infer_probabilities(self, window: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference on one or multiple temporal views and return mean probabilities."""
        if self.model is None and self.onnx_session is None:
            return np.array([], dtype=np.float32), 0.0

        views = self._build_inference_views(window)

        # ONNX inference path
        if self.use_onnx and self.onnx_session is not None:
            return self._infer_probabilities_onnx(views)

        # PyTorch inference path
        return self._infer_probabilities_pytorch(views)

    def _infer_probabilities_pytorch(self, views: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Run PyTorch inference on multiple views."""
        stacked_probs: list[torch.Tensor] = []

        with torch.no_grad():
            for view in views:
                tensor = torch.from_numpy(view).float().unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                softmax_temp = self.calibration_temperature or self.inference_temperature
                probs = torch.softmax(logits / softmax_temp, dim=1)
                stacked_probs.append(probs.squeeze(0))

        if not stacked_probs:
            return np.array([], dtype=np.float32), 0.0

        probs_stack = torch.stack(stacked_probs, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        mean_probs = mean_probs / mean_probs.sum().clamp_min(1e-8)
        disagreement = float(probs_stack.std(dim=0, unbiased=False).mean().item())
        return mean_probs.cpu().numpy(), disagreement

    def _infer_probabilities_onnx(self, views: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Run ONNX Runtime inference on multiple views."""
        stacked_probs: list[np.ndarray] = []

        for view in views:
            # Prepare input: [1, seq_len, features]
            onnx_input = view.astype(np.float32).reshape(1, view.shape[0], view.shape[1])

            # Run ONNX inference
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            logits = self.onnx_session.run([output_name], {input_name: onnx_input})[0]

            # Apply softmax with temperature
            softmax_temp = self.calibration_temperature or self.inference_temperature
            logits_scaled = logits / softmax_temp

            # Compute softmax manually
            exp_logits = np.exp(logits_scaled - logits_scaled.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            stacked_probs.append(probs.squeeze(0))

        if not stacked_probs:
            return np.array([], dtype=np.float32), 0.0

        # Stack and average probabilities
        probs_array = np.stack(stacked_probs, axis=0)
        mean_probs = probs_array.mean(axis=0)
        mean_probs = mean_probs / max(mean_probs.sum(), 1e-8)
        disagreement = float(probs_array.std(axis=0, ddof=0).mean())

        return mean_probs, disagreement

    def _build_inference_views(self, window: np.ndarray) -> list[np.ndarray]:
        """Create lightweight temporal views for test-time ensembling."""
        if window.ndim != 2:
            return [window.astype(np.float32, copy=False)]
        return self._tta_generator.generate(window)

    def _smooth(self) -> tuple[str, float]:
        """Apply majority-vote smoothing with confidence stabilization."""
        if not self.prediction_history:
            return "NONE", 0.0

        score_by_label: dict[str, list[float]] = {}
        for label, score in self.prediction_history:
            score_by_label.setdefault(label, []).append(score)

        best_label = max(
            score_by_label,
            key=lambda key: (len(score_by_label[key]), np.mean(score_by_label[key])),
        )
        best_scores = score_by_label[best_label]
        vote_ratio = len(best_scores) / len(self.prediction_history)
        mean_confidence = float(np.mean(best_scores))

        # Keep confidence interpretable while rewarding temporal agreement.
        stabilized_confidence = mean_confidence * (0.6 + 0.4 * vote_ratio)
        return best_label, round(stabilized_confidence, 3)

    def _mean_hand_visibility(self) -> float:
        """Return rolling mean hand visibility over the current frame window."""
        if not self.hand_visibility_history:
            return 0.0
        return float(np.mean(self.hand_visibility_history))

    def _mean_motion_energy(self) -> float:
        """Return rolling hand motion magnitude over the active frame window."""
        if not self.motion_history:
            return 0.0
        return float(np.mean(self.motion_history))

    def _compute_hand_visibility(self, frame: FrameLandmarks) -> float:
        """Estimate hand landmark visibility score in [0, 1] for one frame."""
        expected_points_per_hand = 21

        def visible_points(hand: list[list[float]]) -> int:
            if not hand:
                return 0
            count = 0
            for point in hand[:expected_points_per_hand]:
                if len(point) >= 3 and any(abs(float(coord)) > 1e-6 for coord in point[:3]):
                    count += 1
            return count

        total_visible = visible_points(frame.left_hand) + visible_points(frame.right_hand)
        total_expected = expected_points_per_hand * 2
        return min(1.0, total_visible / total_expected)

    def _compute_motion_energy(self) -> float:
        """Estimate hand motion energy between recent frames."""
        if len(self.frame_buffer) < 2:
            return 0.0

        previous = self.frame_buffer[-2]
        current = self.frame_buffer[-1]
        hand_previous = previous[:126]  # left + right hands
        hand_current = current[:126]
        motion = np.mean(np.abs(hand_current - hand_previous))
        return float(motion)

    def _adaptive_threshold(self) -> float:
        """
        Raise threshold when tracking quality drops.

        This suppresses false positives for low-visibility / low-motion windows.
        """
        threshold = self.confidence_threshold
        if self._mean_hand_visibility() < max(0.3, self.min_hand_visibility + 0.05):
            threshold += 0.05
        if self._mean_motion_energy() < self.min_motion_energy:
            threshold += 0.06
        if self._latest_frontend_confidence < self.frontend_confidence_floor:
            threshold += 0.08
        elif self._latest_frontend_confidence < 0.55:
            threshold += 0.04
        return float(min(0.95, threshold))

    def _threshold_for_label(self, label: str) -> float:
        """Return class-specific confidence threshold when available."""
        if not label:
            return 0.0
        if label in self.class_thresholds:
            return float(self.class_thresholds[label])
        if label == "NONE" and "[NONE]" in self.class_thresholds:
            return float(self.class_thresholds["[NONE]"])
        if label == "[NONE]" and "NONE" in self.class_thresholds:
            return float(self.class_thresholds["NONE"])
        return 0.0

    def _apply_prediction_filters(
        self,
        *,
        prediction: str,
        confidence: float,
        threshold: float,
    ) -> tuple[str, float, dict[str, object]]:
        """Filter raw predictions with temporal consensus + adaptive threshold."""
        if prediction == "NONE" or confidence <= 0.0:
            self._decision_counters["rejected_by_calibration"] += 1
            self._decision_counters["rejected_total"] += 1
            trace = {
                "status": "rejected",
                "stage": "filters",
                "reason": "prediction_none_or_non_positive_confidence",
                "confidence": round(float(confidence), 4),
                "threshold": round(float(threshold), 4),
            }
            self._last_decision_trace = trace
            return "NONE", 0.0, trace

        if self.prediction_history and self.prediction_history[-1][0] != prediction:
            self.prediction_history.clear()

        self.prediction_history.append((prediction, confidence))
        smoothed_label, smoothed_confidence = self._smooth()

        # Keep fast response when a stable label repeats.
        if smoothed_label == prediction:
            confidence = max(confidence, smoothed_confidence)

        if confidence < threshold:
            self._decision_counters["rejected_by_confidence_threshold"] += 1
            self._decision_counters["rejected_total"] += 1
            if threshold > self.confidence_threshold:
                self._decision_counters["rejected_by_adaptive_threshold"] += 1
            class_threshold = self._threshold_for_label(prediction)
            if class_threshold > 0 and threshold >= class_threshold:
                self._decision_counters["rejected_by_class_threshold"] += 1
            trace = {
                "status": "rejected",
                "stage": "filters",
                "reason": "below_effective_threshold",
                "prediction": prediction,
                "confidence": round(float(confidence), 4),
                "smoothed_confidence": round(float(smoothed_confidence), 4),
                "threshold": round(float(threshold), 4),
            }
            self._last_decision_trace = trace
            return "NONE", 0.0, trace

        self._decision_counters["accepted"] += 1
        trace = {
            "status": "accepted",
            "stage": "filters",
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "threshold": round(float(threshold), 4),
            "smoothed_label": smoothed_label,
            "smoothed_confidence": round(float(smoothed_confidence), 4),
        }
        self._last_decision_trace = trace

        return prediction, round(confidence, 3), trace

    def _trim_recording_window(self, window: np.ndarray, trailing_rest: int) -> np.ndarray:
        """Trim pre/post idle frames before resampling."""
        if window.ndim != 2 or window.shape[0] == 0:
            return window

        trimmed = window
        if trailing_rest > 0:
            min_keep = max(2, self.min_recording_frames // 2)
            if trimmed.shape[0] > (trailing_rest + min_keep):
                trimmed = trimmed[:-trailing_rest]

        hand_energy = np.mean(np.abs(trimmed[:, :126]), axis=1)
        active_indices = np.flatnonzero(hand_energy > 1e-4)
        if active_indices.size == 0:
            return trimmed

        start = int(max(0, active_indices[0] - 1))
        end = int(min(trimmed.shape[0], active_indices[-1] + 2))
        if end <= start:
            return trimmed
        return trimmed[start:end]

    @staticmethod
    def _resolve_frontend_confidence(metadata: dict) -> float:
        """Parse frontend confidence metadata into a stable [0, 1] range."""
        if not isinstance(metadata, dict):
            return 1.0
        value = metadata.get("averageConfidence")
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 1.0
        if np.isnan(confidence):
            return 1.0
        return float(np.clip(confidence, 0.0, 1.0))

    def _blend_hand_visibility(self, hand_visibility: float, frontend_confidence: float) -> float:
        """Blend backend-computed hand visibility with frontend tracking quality."""
        if frontend_confidence >= self.frontend_confidence_floor:
            return hand_visibility
        penalty = (self.frontend_confidence_floor - frontend_confidence) * 0.8
        return float(max(0.0, hand_visibility - penalty))

    def _calibrate_confidence(
        self,
        *,
        probs: np.ndarray,
        top_indices: np.ndarray,
        raw_confidence: float,
        disagreement: float = 0.0,
    ) -> float:
        """Calibrate confidence with margin and distribution certainty."""
        if raw_confidence <= 0.0:
            return 0.0

        top2_confidence = float(probs[top_indices[1]]) if len(top_indices) > 1 else 0.0
        margin = max(0.0, raw_confidence - top2_confidence)

        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = float(np.log(max(len(probs), 2)))
        certainty = 1.0 - min(1.0, entropy / max_entropy)

        # Motion-aware calibration for open-set rejection in static/noisy windows.
        motion_factor = np.clip(
            self._current_motion_energy / max(self.min_motion_energy, 1e-6), 0.0, 1.0
        )
        calibrated = (
            (0.58 * raw_confidence)
            + (0.24 * margin)
            + (0.10 * certainty)
            + (0.08 * motion_factor)
        )
        if margin < self.min_prediction_margin:
            self._decision_counters["rejected_by_margin"] += 1
            calibrated *= margin / self.min_prediction_margin
        if self._current_motion_energy < self.min_motion_energy:
            self._decision_counters["rejected_by_motion"] += 1
            calibrated *= 0.85

        if self.inference_num_views > 1:
            disagreement_ratio = np.clip(disagreement / self.max_view_disagreement, 0.0, 1.0)
            calibrated *= 1.0 - (0.35 * disagreement_ratio)
            if disagreement > self.max_view_disagreement:
                self._decision_counters["rejected_by_tta_disagreement"] += 1
                calibrated *= 0.8

        return float(np.clip(calibrated, 0.0, 1.0))

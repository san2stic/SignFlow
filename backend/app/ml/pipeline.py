"""Inference pipeline for real-time sign translation from landmarks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import structlog
import torch

from app.ml.features import FrameLandmarks, normalize_landmarks
from app.ml.model import SignTransformer
from app.ml.trainer import load_model_checkpoint

logger = structlog.get_logger(__name__)


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


class SignFlowInferencePipeline:
    """Sliding-window inference with confidence threshold and temporal smoothing."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
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
        """
        self.seq_len = seq_len
        self.confidence_threshold = confidence_threshold
        self.min_hand_visibility = min_hand_visibility
        self.min_prediction_margin = min_prediction_margin
        self.min_motion_energy = min_motion_energy
        self.device = torch.device(device)
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=max_buffer_frames)
        self.hand_visibility_history: deque[float] = deque(maxlen=seq_len)
        self.motion_history: deque[float] = deque(maxlen=seq_len)
        self.prediction_history: deque[tuple[str, float]] = deque(maxlen=max(3, smoothing_window))
        self.labels: list[str] = ["NONE"]
        self.sentence_tokens: list[str] = []
        self._current_motion_energy = 0.0

        # State machine attributes
        self.max_buffer_frames = max_buffer_frames
        self.rest_frames_threshold = rest_frames_threshold
        self.motion_start_threshold = motion_start_threshold
        self.min_recording_frames = min_recording_frames
        self.state = InferenceState.IDLE
        self._rest_frame_count = 0
        self._recording_frame_count = 0

        # Load model if path provided
        self.model: SignTransformer | None = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str | Path) -> None:
        """
        Load PyTorch model from checkpoint.

        Args:
            model_path: Path to .pt checkpoint file
        """
        try:
            logger.info("loading_model", path=str(model_path))
            self.model = load_model_checkpoint(model_path, device=str(self.device))
            self.model.to(self.device)
            self.model.set_inference_mode()
            logger.info(
                "model_loaded_successfully",
                num_classes=self.model.num_classes,
                device=str(self.device),
            )
        except Exception as e:
            logger.error("failed_to_load_model", path=str(model_path), error=str(e))
            self.model = None
            raise

    def set_labels(self, labels: list[str]) -> None:
        """
        Configure active vocabulary labels for model outputs.

        Args:
            labels: List of sign labels (class names)
        """
        self.labels = ["NONE", *[label for label in labels if label != "NONE"]]
        logger.debug("labels_set", num_labels=len(self.labels))

    def process_frame(self, payload: dict) -> Prediction:
        """Process a landmarks frame through the state machine."""
        frame = FrameLandmarks(
            left_hand=payload.get("hands", {}).get("left", []) or [],
            right_hand=payload.get("hands", {}).get("right", []) or [],
            pose=payload.get("pose", []) or [],
            face=payload.get("face", []) or [],
        )

        hand_visibility = self._compute_hand_visibility(frame)
        self.hand_visibility_history.append(hand_visibility)

        features = normalize_landmarks(frame, include_face=False)
        self.frame_buffer.append(features)
        self._current_motion_energy = self._compute_motion_energy()
        self.motion_history.append(self._current_motion_energy)

        # State machine
        if self.state == InferenceState.IDLE:
            if (self._current_motion_energy > self.motion_start_threshold
                    and hand_visibility > self.min_hand_visibility):
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
        from app.ml.dataset import temporal_resample
        from app.ml.feature_engineering import compute_enriched_features

        buffer_array = np.stack(list(self.frame_buffer), axis=0)
        resampled = temporal_resample(buffer_array, target_len=self.seq_len)
        enriched = compute_enriched_features(resampled)

        predicted_label, predicted_confidence, alternatives = self._infer_window(enriched)

        # Reset for next sign
        self.frame_buffer.clear()
        self._rest_frame_count = 0
        self._recording_frame_count = 0
        self.state = InferenceState.IDLE

        if predicted_label != "NONE" and predicted_confidence >= self.confidence_threshold:
            if not self.sentence_tokens or self.sentence_tokens[-1] != predicted_label:
                self.sentence_tokens.append(predicted_label)

        sentence = " ".join(self.sentence_tokens)
        return Prediction(
            prediction=predicted_label,
            confidence=predicted_confidence,
            alternatives=alternatives,
            sentence_buffer=sentence,
            is_sentence_complete=False,
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

    def _infer_window(self, window: np.ndarray) -> tuple[str, float, list[dict[str, float]]]:
        """
        Run PyTorch model inference over one sequence window.

        Args:
            window: Numpy array of shape [seq_len, num_features]

        Returns:
            Tuple of (prediction_label, confidence, alternatives)
        """
        # If no model loaded, return NONE
        if self.model is None or len(self.labels) <= 1:
            return "NONE", 0.0, []

        try:
            # Convert to PyTorch tensor
            tensor = torch.from_numpy(window).float().unsqueeze(0)  # [1, seq_len, features]
            tensor = tensor.to(self.device)

            # Forward pass
            with torch.no_grad():
                logits = self.model(tensor)  # [1, num_classes]
                probabilities = torch.softmax(logits, dim=1)  # [1, num_classes]

            # Get top predictions
            probs = probabilities[0].cpu().numpy()
            top_k = min(4, len(probs))  # Get top 4 predictions
            top_indices = np.argsort(probs)[-top_k:][::-1]

            # Main prediction
            pred_idx = int(top_indices[0])
            raw_confidence = float(probs[pred_idx])

            # Map index to label
            if pred_idx < len(self.labels):
                prediction = self.labels[pred_idx]
            else:
                prediction = "NONE"
                raw_confidence = 0.0

            # Alternatives (top 3 excluding main prediction)
            alternatives = []
            for idx in top_indices[1:top_k]:
                if idx < len(self.labels):
                    alt_label = self.labels[idx]
                    alt_conf = float(probs[idx])
                    alternatives.append({
                        "sign": alt_label,
                        "confidence": round(alt_conf, 3)
                    })

            calibrated_confidence = self._calibrate_confidence(
                probs=probs,
                top_indices=top_indices,
                raw_confidence=raw_confidence,
            )

            # If confidence margin is too low, treat prediction as NONE.
            if calibrated_confidence <= 0.0:
                return "NONE", 0.0, alternatives

            return prediction, round(calibrated_confidence, 3), alternatives

        except Exception as e:
            logger.error("inference_failed", error=str(e), exc_info=True)
            return "NONE", 0.0, []

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
        return float(min(0.95, threshold))

    def _calibrate_confidence(
        self,
        *,
        probs: np.ndarray,
        top_indices: np.ndarray,
        raw_confidence: float,
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
            calibrated *= margin / self.min_prediction_margin
        if self._current_motion_energy < self.min_motion_energy:
            calibrated *= 0.85

        return float(np.clip(calibrated, 0.0, 1.0))

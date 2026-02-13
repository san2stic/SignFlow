"""Inference pipeline unit test."""

from __future__ import annotations

import numpy as np
import torch

from app.ml.pipeline import SignFlowInferencePipeline


def test_pipeline_warmup_prediction() -> None:
    """Pipeline should output NONE until sequence window is full."""
    pipeline = SignFlowInferencePipeline(seq_len=3)
    payload = {
        "timestamp": 0,
        "frame_idx": 1,
        "hands": {"left": [[0.1, 0.1, 0.0]] * 21, "right": [[0.1, 0.1, 0.0]] * 21},
        "pose": [[0.1, 0.1, 0.0]] * 33
    }

    first = pipeline.process_frame(payload)
    assert first.prediction == "NONE"
    assert first.confidence == 0.0


def test_pipeline_smoothing_prefers_temporal_consensus() -> None:
    """Smoothing should prefer repeated labels over a single high-confidence outlier."""
    pipeline = SignFlowInferencePipeline(seq_len=1, smoothing_window=3)
    pipeline.prediction_history.extend(
        [
            ("bonjour", 0.95),
            ("merci", 0.82),
            ("merci", 0.80),
        ]
    )

    smoothed_label, smoothed_confidence = pipeline._smooth()

    assert smoothed_label == "merci"
    assert smoothed_confidence > 0.65


def test_pipeline_rejects_predictions_when_hands_not_visible() -> None:
    """Inference should be muted when hand visibility stays below threshold."""
    pipeline = SignFlowInferencePipeline(seq_len=3, min_hand_visibility=0.25)
    pipeline._infer_window = lambda _window: ("bonjour", 0.95, [])  # type: ignore[method-assign]

    payload = {
        "timestamp": 0,
        "frame_idx": 1,
        "hands": {"left": [], "right": []},
        "pose": [[0.1, 0.1, 0.0]] * 33,
    }

    pipeline.process_frame(payload)
    pipeline.process_frame(payload)
    output = pipeline.process_frame(payload)

    assert output.prediction == "NONE"
    assert output.confidence == 0.0
    assert output.alternatives == []


def test_pipeline_single_class_label_mapping_has_no_none_offset() -> None:
    """Single-class models should map index 0 to the provided sign label."""

    class _SingleClassModel(torch.nn.Module):
        num_classes = 1

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[8.0]], device=x.device)

    pipeline = SignFlowInferencePipeline(seq_len=30)
    pipeline.model = _SingleClassModel()
    pipeline.set_labels(["lsfb_bonjour"])

    window = np.zeros((30, 469), dtype=np.float32)
    prediction, confidence, alternatives = pipeline._infer_window(window)

    assert prediction == "lsfb_bonjour"
    assert confidence > 0.7
    assert alternatives == []


def test_pipeline_adaptive_threshold_increases_on_low_frontend_confidence() -> None:
    """Adaptive threshold should become stricter when frontend tracking confidence is low."""
    pipeline = SignFlowInferencePipeline(
        confidence_threshold=0.7,
        min_motion_energy=0.003,
        frontend_confidence_floor=0.4,
    )
    pipeline.hand_visibility_history.extend([0.9, 0.95, 0.92])
    pipeline.motion_history.extend([0.02, 0.019, 0.021])

    pipeline._latest_frontend_confidence = 0.9
    high_quality_threshold = pipeline._adaptive_threshold()

    pipeline._latest_frontend_confidence = 0.2
    low_quality_threshold = pipeline._adaptive_threshold()

    assert high_quality_threshold >= pipeline.confidence_threshold
    assert low_quality_threshold > high_quality_threshold


def test_pipeline_trim_recording_window_removes_idle_edges() -> None:
    """Trimming should remove obvious idle frames before/after active sign motion."""
    pipeline = SignFlowInferencePipeline(min_recording_frames=6)

    idle_before = np.zeros((4, 225), dtype=np.float32)
    active = np.ones((8, 225), dtype=np.float32) * 0.2
    idle_after = np.zeros((3, 225), dtype=np.float32)
    window = np.concatenate([idle_before, active, idle_after], axis=0)

    trimmed = pipeline._trim_recording_window(window, trailing_rest=3)

    assert trimmed.shape[0] < window.shape[0]
    assert np.any(np.abs(trimmed[:, :126]) > 1e-6)


def test_pipeline_prediction_filters_reject_below_threshold() -> None:
    """Post-processing should reject predictions that do not meet adaptive threshold."""
    pipeline = SignFlowInferencePipeline()

    label, confidence = pipeline._apply_prediction_filters(
        prediction="bonjour",
        confidence=0.66,
        threshold=0.7,
    )
    assert label == "NONE"
    assert confidence == 0.0

    accepted_label, accepted_confidence = pipeline._apply_prediction_filters(
        prediction="bonjour",
        confidence=0.82,
        threshold=0.7,
    )
    assert accepted_label == "bonjour"
    assert accepted_confidence >= 0.82


def test_pipeline_builds_multiple_inference_views() -> None:
    """TTA helper should generate the configured number of temporal views."""
    pipeline = SignFlowInferencePipeline(seq_len=64, inference_num_views=4)
    window = np.random.default_rng(42).standard_normal((64, 469)).astype(np.float32)

    views = pipeline._build_inference_views(window)

    assert len(views) == 4
    assert all(view.shape == window.shape for view in views)


def test_pipeline_confidence_penalizes_high_view_disagreement() -> None:
    """Calibration should lower confidence when TTA views disagree strongly."""
    pipeline = SignFlowInferencePipeline(inference_num_views=3, max_view_disagreement=0.2)
    probs = np.array([0.85, 0.1, 0.05], dtype=np.float32)
    top_indices = np.array([0, 1, 2], dtype=np.int64)

    low_disagreement = pipeline._calibrate_confidence(
        probs=probs,
        top_indices=top_indices,
        raw_confidence=0.85,
        disagreement=0.02,
    )
    high_disagreement = pipeline._calibrate_confidence(
        probs=probs,
        top_indices=top_indices,
        raw_confidence=0.85,
        disagreement=0.4,
    )

    assert low_disagreement > high_disagreement

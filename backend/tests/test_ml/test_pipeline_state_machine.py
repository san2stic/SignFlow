"""Tests for inference pipeline state machine."""
from __future__ import annotations

import numpy as np

from app.ml.pipeline import SignFlowInferencePipeline, InferenceState


def _hand_payload(value: float = 0.5) -> dict:
    """Create a payload with visible hand landmarks."""
    return {
        "hands": {
            "left": [[value, value, 0.0]] * 21,
            "right": [[value + 0.1, value, 0.0]] * 21,
        },
        "pose": [[0.5, 0.5, 0.0]] * 33,
    }


def _idle_payload() -> dict:
    """Create a payload with hands at rest (zero)."""
    return {
        "hands": {"left": [[0.0, 0.0, 0.0]] * 21, "right": [[0.0, 0.0, 0.0]] * 21},
        "pose": [[0.5, 0.5, 0.0]] * 33,
    }


def test_pipeline_starts_in_idle():
    pipeline = SignFlowInferencePipeline()
    assert pipeline.state == InferenceState.IDLE


def test_pipeline_transitions_to_recording_on_motion():
    pipeline = SignFlowInferencePipeline()
    for i in range(5):
        pipeline.process_frame(_hand_payload(0.1 * (i + 1)))
    assert pipeline.state == InferenceState.RECORDING


def test_pipeline_returns_to_idle_after_rest():
    pipeline = SignFlowInferencePipeline(
        rest_frames_threshold=3, min_recording_frames=2
    )
    for i in range(5):
        pipeline.process_frame(_hand_payload(0.1 * (i + 1)))
    for _ in range(5):
        pipeline.process_frame(_idle_payload())
    assert pipeline.state == InferenceState.IDLE


def test_pipeline_prediction_on_sign_complete():
    pipeline = SignFlowInferencePipeline(
        rest_frames_threshold=3, min_recording_frames=2
    )
    for i in range(10):
        pipeline.process_frame(_hand_payload(0.1 * (i + 1)))
    for _ in range(5):
        result = pipeline.process_frame(_idle_payload())
    assert result.prediction == "NONE"

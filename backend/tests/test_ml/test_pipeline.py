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

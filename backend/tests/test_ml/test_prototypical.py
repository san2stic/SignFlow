"""Prototype fallback behavior tests."""

from __future__ import annotations

import numpy as np
import torch

from app.ml.dataset import LandmarkDataset, SignSample
from app.ml.model import SignTransformer
from app.ml.prototypical import run_prototypical_fallback


def _make_sequence(center: float, noise: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(1234 + int(center * 100))
    base = np.full((12, 225), center, dtype=np.float32)
    jitter = rng.normal(0.0, noise, size=base.shape).astype(np.float32)
    return base + jitter


def test_prototypical_fallback_returns_metrics_and_prototypes() -> None:
    """Fallback should produce valid metrics/prototypes on low-shot data."""
    train_samples = [
        SignSample(landmarks=_make_sequence(0.1), label=0),
        SignSample(landmarks=_make_sequence(0.11), label=0),
        SignSample(landmarks=_make_sequence(0.9), label=1),
        SignSample(landmarks=_make_sequence(0.89), label=1),
    ]
    val_samples = [
        SignSample(landmarks=_make_sequence(0.12), label=0),
        SignSample(landmarks=_make_sequence(0.88), label=1),
    ]

    train_dataset = LandmarkDataset(
        train_samples,
        sequence_length=12,
        stride=12,
        apply_sliding_window=False,
        use_enriched_features=False,
    )
    val_dataset = LandmarkDataset(
        val_samples,
        sequence_length=12,
        stride=12,
        apply_sliding_window=False,
        use_enriched_features=False,
    )

    model = SignTransformer(num_features=225, num_classes=2, d_model=32)

    metric, prototypes = run_prototypical_fallback(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device="cpu",
        batch_size=8,
    )

    assert metric.epoch == 1
    assert 0.0 <= metric.train_accuracy <= 1.0
    assert 0.0 <= metric.val_accuracy <= 1.0
    assert set(prototypes.keys()) == {0, 1}
    assert all(vector.ndim == 1 for vector in prototypes.values())

    with torch.no_grad():
        weight = model.classifier.weight
        assert weight.shape[0] == 2

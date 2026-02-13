"""End-to-end integration test for the improved ML pipeline."""
from __future__ import annotations

import numpy as np
import torch

from app.ml.dataset import LandmarkDataset, SignSample, temporal_resample
from app.ml.feature_engineering import compute_enriched_features, ENRICHED_FEATURE_DIM
from app.ml.model import SignTransformer
from app.ml.trainer import SignTrainer, TrainingConfig


def test_full_training_pipeline_no_overfit():
    """Train on synthetic data and verify model doesn't achieve 0.0 loss in 1 epoch."""
    rng = np.random.default_rng(42)
    num_classes = 3
    samples_per_class = 10

    samples = []
    for class_idx in range(num_classes):
        for _ in range(samples_per_class):
            num_frames = rng.integers(30, 120)
            base = rng.standard_normal((1, 225)).astype(np.float32) * (class_idx + 1)
            noise = rng.standard_normal((num_frames, 225)).astype(np.float32) * 0.3
            landmarks = base + noise
            samples.append(SignSample(landmarks=landmarks, label=class_idx))

    rng.shuffle(samples)
    train_samples = samples[:24]
    val_samples = samples[24:]

    train_ds = LandmarkDataset(train_samples, sequence_length=64, apply_sliding_window=False, use_enriched_features=True)
    val_ds = LandmarkDataset(val_samples, sequence_length=64, apply_sliding_window=False, use_enriched_features=True)

    model = SignTransformer(num_classes=num_classes)

    config = TrainingConfig(
        num_epochs=3,
        learning_rate=1e-4,
        batch_size=8,
        num_workers=0,
        device="cpu",
        early_stopping_patience=15,
    )

    trainer = SignTrainer(model=model, config=config)
    metrics = trainer.fit(train_ds, val_ds)

    # Key assertion: model should NOT have 0.0 loss after epoch 1
    assert metrics[0].train_loss > 0.01, f"Train loss too low at epoch 1: {metrics[0].train_loss}"
    assert len(metrics) == 3


def test_temporal_resample_then_enriched_features():
    """Verify the data pipeline: raw -> resample -> enrich."""
    raw = np.random.default_rng(42).standard_normal((100, 225)).astype(np.float32)
    resampled = temporal_resample(raw, target_len=64)
    enriched = compute_enriched_features(resampled)
    assert enriched.shape == (64, ENRICHED_FEATURE_DIM)
    assert not np.any(np.isnan(enriched))

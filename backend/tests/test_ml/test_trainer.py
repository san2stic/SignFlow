"""Trainer-level tests for advanced optimization features."""

from __future__ import annotations

import torch

from app.ml.feature_engineering import ENRICHED_FEATURE_DIM
from app.ml.model import SignTransformer
from app.ml.trainer import SignTrainer, TrainingConfig


def test_temporal_mask_regularization_preserves_shape_and_masks_tokens() -> None:
    """Temporal masking should keep tensor shape and drop at least one span."""
    model = SignTransformer(num_features=ENRICHED_FEATURE_DIM, num_classes=3)
    config = TrainingConfig(
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        temporal_mask_prob=1.0,
        temporal_mask_span_ratio=0.3,
    )
    trainer = SignTrainer(model=model, config=config)
    trainer.model.train()

    landmarks = torch.ones((2, 24, ENRICHED_FEATURE_DIM), dtype=torch.float32)
    masked = trainer._apply_temporal_mask(landmarks)

    assert masked.shape == landmarks.shape
    assert torch.any(masked == 0.0)


def test_trainer_cpu_disables_amp_even_if_requested() -> None:
    """AMP should only activate on CUDA devices."""
    model = SignTransformer(num_features=ENRICHED_FEATURE_DIM, num_classes=2)
    config = TrainingConfig(
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        use_amp=True,
    )

    trainer = SignTrainer(model=model, config=config)
    assert trainer.autocast_enabled is False


def test_stable_inverse_frequency_weights_are_clipped_and_normalized() -> None:
    """Stable class-imbalance weights should remain bounded and finite."""
    counts = torch.tensor([50, 5, 1], dtype=torch.float32).numpy()
    weights = SignTrainer._stable_inverse_frequency_weights(
        counts,
        power=0.75,
        min_weight=0.3,
        max_weight=3.0,
    )

    assert weights.shape == (3,)
    assert float(weights.min()) >= 0.3
    assert float(weights.max()) <= 3.0
    assert weights[0] <= weights[1] <= weights[2]

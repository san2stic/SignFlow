"""Unit tests for SignTransformer pooling behavior."""

from __future__ import annotations

import torch

from app.ml.model import SignTransformer


def test_masked_temporal_pool_ignores_inactive_frames() -> None:
    """Pooling should average only active frames and skip zero-padded ones."""
    encoded = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]]],
        dtype=torch.float32,
    )
    raw_input = torch.tensor(
        [[[1.0, 1.0], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )

    pooled = SignTransformer.masked_temporal_pool(encoded, raw_input)
    expected = torch.tensor([[2.0, 3.0]], dtype=torch.float32)

    assert torch.allclose(pooled, expected)


def test_masked_temporal_pool_falls_back_to_mean_when_all_frames_inactive() -> None:
    """If no frame is active, pooling should fall back to standard mean pooling."""
    encoded = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]]],
        dtype=torch.float32,
    )
    raw_input = torch.zeros_like(encoded)

    pooled = SignTransformer.masked_temporal_pool(encoded, raw_input)
    expected = encoded.mean(dim=1)

    assert torch.allclose(pooled, expected)


from app.ml.feature_engineering import ENRICHED_FEATURE_DIM


def test_model_reduced_defaults():
    """New defaults should expose the V2 architecture settings."""
    model = SignTransformer(num_classes=5)
    assert model.d_model == 192
    assert model.nhead == 6
    assert model.num_layers == 4
    assert model.num_features == ENRICHED_FEATURE_DIM
    assert model.use_multiscale_stem is True
    assert model.use_cosine_head is True


def test_model_forward_with_enriched_features():
    """Model should handle enriched feature dimension."""
    model = SignTransformer(num_classes=5)
    x = torch.randn(2, 64, ENRICHED_FEATURE_DIM)
    logits = model(x)
    assert logits.shape == (2, 5)


def test_model_old_config_still_works():
    """Explicit old-style config should still work."""
    model = SignTransformer(
        num_features=225, num_classes=3, d_model=256, nhead=8, num_layers=4
    )
    x = torch.randn(1, 30, 225)
    logits = model(x)
    assert logits.shape == (1, 3)

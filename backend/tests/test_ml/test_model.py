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

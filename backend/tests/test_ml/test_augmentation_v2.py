"""Tests for new augmentation: temporal_crop."""
from __future__ import annotations

import numpy as np

from app.ml.augmentation import temporal_crop


def test_temporal_crop_preserves_features():
    """Output should have same shape as input."""
    seq = np.random.default_rng(42).standard_normal((64, 469)).astype(np.float32)
    cropped = temporal_crop(seq, min_ratio=0.7)
    assert cropped.shape == (64, 469)


def test_temporal_crop_is_different():
    """Cropped sequence should differ from original (stochastic)."""
    seq = np.random.default_rng(42).standard_normal((64, 469)).astype(np.float32)
    np.random.seed(123)
    cropped = temporal_crop(seq, min_ratio=0.7)
    assert not np.allclose(cropped, seq)


def test_temporal_crop_short_sequence():
    """Very short sequence should be returned as-is."""
    seq = np.random.default_rng(42).standard_normal((3, 10)).astype(np.float32)
    cropped = temporal_crop(seq, min_ratio=0.7)
    assert cropped.shape == (3, 10)

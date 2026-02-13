"""Tests for temporal resampling in dataset."""
from __future__ import annotations

import numpy as np

from app.ml.dataset import temporal_resample


def test_resample_shorter_sequence_pads():
    seq = np.arange(20).reshape(4, 5).astype(np.float32)
    result = temporal_resample(seq, target_len=8)
    assert result.shape == (8, 5)


def test_resample_longer_sequence_downsamples():
    seq = np.arange(100).reshape(20, 5).astype(np.float32)
    result = temporal_resample(seq, target_len=8)
    assert result.shape == (8, 5)


def test_resample_exact_length_unchanged():
    seq = np.arange(40).reshape(8, 5).astype(np.float32)
    result = temporal_resample(seq, target_len=8)
    assert result.shape == (8, 5)
    assert np.allclose(result, seq)


def test_resample_preserves_endpoints():
    seq = np.random.default_rng(42).standard_normal((30, 225)).astype(np.float32)
    result = temporal_resample(seq, target_len=64)
    assert np.allclose(result[0], seq[0], atol=1e-5)
    assert np.allclose(result[-1], seq[-1], atol=1e-5)


def test_resample_single_frame():
    seq = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result = temporal_resample(seq, target_len=8)
    assert result.shape == (8, 3)
    for i in range(8):
        assert np.allclose(result[i], seq[0])

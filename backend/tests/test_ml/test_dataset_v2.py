"""Tests for updated LandmarkDataset with resampling and enriched features."""
from __future__ import annotations

import numpy as np
import torch

from app.ml.dataset import LandmarkDataset, SignSample
from app.ml.feature_engineering import ENRICHED_FEATURE_DIM


def _make_sample(num_frames: int, label: int = 0) -> SignSample:
    rng = np.random.default_rng(42 + label)
    landmarks = rng.standard_normal((num_frames, 225)).astype(np.float32)
    return SignSample(landmarks=landmarks, label=label)


def test_dataset_resample_mode_output_shape():
    samples = [_make_sample(50, 0), _make_sample(100, 1)]
    ds = LandmarkDataset(samples, sequence_length=64, use_enriched_features=True, apply_sliding_window=False)
    tensor, label = ds[0]
    assert tensor.shape == (64, ENRICHED_FEATURE_DIM)


def test_dataset_resample_mode_short_video():
    samples = [_make_sample(10, 0)]
    ds = LandmarkDataset(samples, sequence_length=64, use_enriched_features=True, apply_sliding_window=False)
    tensor, _ = ds[0]
    assert tensor.shape == (64, ENRICHED_FEATURE_DIM)


def test_dataset_resample_mode_long_video():
    samples = [_make_sample(200, 0)]
    ds = LandmarkDataset(samples, sequence_length=64, use_enriched_features=True, apply_sliding_window=False)
    tensor, _ = ds[0]
    assert tensor.shape == (64, ENRICHED_FEATURE_DIM)


def test_dataset_one_sample_per_video():
    samples = [_make_sample(50, 0), _make_sample(100, 1), _make_sample(200, 2)]
    ds = LandmarkDataset(samples, sequence_length=64, use_enriched_features=True, apply_sliding_window=False)
    assert len(ds) == 3


def test_dataset_backward_compat_sliding_window():
    samples = [_make_sample(50, 0)]
    ds = LandmarkDataset(samples, sequence_length=30, stride=10, apply_sliding_window=True, use_enriched_features=False)
    tensor, _ = ds[0]
    assert tensor.shape[1] == 225

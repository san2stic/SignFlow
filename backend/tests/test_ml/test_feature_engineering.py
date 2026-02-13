"""Tests for enriched feature computation."""
from __future__ import annotations

import numpy as np

from app.ml.feature_engineering import (
    compute_enriched_features,
    compute_hand_distances,
    compute_hand_shape_features,
    compute_joint_angles,
    compute_velocities,
    normalize_body_frame,
    ENRICHED_FEATURE_DIM,
)


def _make_sequence(num_frames: int = 10, num_features: int = 225) -> np.ndarray:
    """Create a synthetic landmark sequence."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((num_frames, num_features)).astype(np.float32)


def test_compute_velocities_shape():
    seq = _make_sequence(10, 225)
    vel = compute_velocities(seq)
    assert vel.shape == (10, 225)


def test_compute_velocities_first_frame_is_zero():
    seq = _make_sequence(10, 225)
    vel = compute_velocities(seq)
    assert np.allclose(vel[0], 0.0)


def test_compute_velocities_values():
    seq = np.array([[1.0, 2.0], [4.0, 6.0], [5.0, 5.0]], dtype=np.float32)
    vel = compute_velocities(seq)
    assert np.allclose(vel[1], [3.0, 4.0])
    assert np.allclose(vel[2], [1.0, -1.0])


def test_compute_hand_distances_shape():
    seq = _make_sequence(10, 225)
    dist = compute_hand_distances(seq)
    assert dist.shape == (10, 5)


def test_compute_joint_angles_shape():
    seq = _make_sequence(10, 225)
    angles = compute_joint_angles(seq)
    assert angles.shape == (10, 4)


def test_compute_hand_shape_features_shape():
    seq = _make_sequence(10, 225)
    shape_feats = compute_hand_shape_features(seq)
    assert shape_feats.shape == (10, 10)


def test_compute_enriched_features_output_dim():
    seq = _make_sequence(10, 225)
    enriched = compute_enriched_features(seq)
    assert enriched.shape == (10, ENRICHED_FEATURE_DIM)


def test_compute_enriched_features_no_nans():
    seq = _make_sequence(10, 225)
    seq[3, :] = 0.0  # simulate missing frame
    enriched = compute_enriched_features(seq)
    assert not np.any(np.isnan(enriched))


def test_compute_enriched_features_single_frame():
    seq = _make_sequence(1, 225)
    enriched = compute_enriched_features(seq)
    assert enriched.shape[0] == 1
    assert enriched.shape[1] == ENRICHED_FEATURE_DIM


def test_normalize_body_frame_is_translation_invariant():
    seq = _make_sequence(6, 225)
    translated = seq.copy()
    translated[:, 0::3] += 0.35
    translated[:, 1::3] -= 0.22
    translated[:, 2::3] += 0.18

    normalized = normalize_body_frame(seq)
    normalized_translated = normalize_body_frame(translated)

    assert np.allclose(normalized, normalized_translated, atol=1e-4)

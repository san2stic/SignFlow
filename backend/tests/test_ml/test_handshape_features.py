"""Unit tests for backend/app/ml/handshape_features.py."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.handshape_features import (
    HANDSHAPE_FEATURE_DIM,
    compute_abduction_angles,
    compute_finger_angles,
    compute_fingertip_distances,
    compute_global_curvature,
    compute_palm_normal,
    extract_both_hands_handshape,
    extract_handshape_features,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_open_hand() -> np.ndarray:
    """Synthetic open hand: fingers extended along Y axis."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist at origin
    lm[0] = [0, 0, 0]
    # Thumb (slightly lateral)
    lm[1] = [0.04, 0.02, 0]
    lm[2] = [0.06, 0.04, 0]
    lm[3] = [0.07, 0.06, 0]
    lm[4] = [0.08, 0.08, 0]  # thumb tip
    # Index finger — straight up
    lm[5] = [0.02, 0.04, 0]
    lm[6] = [0.02, 0.07, 0]
    lm[7] = [0.02, 0.09, 0]
    lm[8] = [0.02, 0.12, 0]  # tip
    # Middle
    lm[9] =  [0.00, 0.04, 0]
    lm[10] = [0.00, 0.07, 0]
    lm[11] = [0.00, 0.09, 0]
    lm[12] = [0.00, 0.12, 0]
    # Ring
    lm[13] = [-0.02, 0.04, 0]
    lm[14] = [-0.02, 0.07, 0]
    lm[15] = [-0.02, 0.09, 0]
    lm[16] = [-0.02, 0.12, 0]
    # Pinky
    lm[17] = [-0.04, 0.04, 0]
    lm[18] = [-0.04, 0.06, 0]
    lm[19] = [-0.04, 0.08, 0]
    lm[20] = [-0.04, 0.10, 0]
    return lm


def _make_fist() -> np.ndarray:
    """Synthetic fist: all fingertips near wrist."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0, 0, 0]
    # Put all tips close to wrist (curled)
    for i in range(1, 21):
        lm[i] = [0.01 * (i % 5), 0.01 * (i % 4), 0.0]
    # MCP joints needed for palm normal
    lm[5] = [0.02, 0.03, 0]
    lm[17] = [-0.02, 0.03, 0]
    return lm


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFingerAngles:
    def test_output_shape(self):
        lm = _make_open_hand()
        angles = compute_finger_angles(lm)
        assert angles.shape == (15,)
        assert angles.dtype == np.float32

    def test_values_in_range(self):
        lm = _make_open_hand()
        angles = compute_finger_angles(lm)
        assert np.all(angles >= 0.0)
        assert np.all(angles <= np.pi + 1e-6)

    def test_no_nan(self):
        lm = _make_open_hand()
        angles = compute_finger_angles(lm)
        assert not np.any(np.isnan(angles))

    def test_zero_hand_returns_zeros(self):
        lm = np.zeros((21, 3), dtype=np.float32)
        angles = compute_finger_angles(lm)
        # Degenerate input → zeros from _safe_angle
        assert np.all(angles == 0.0)


class TestAbductionAngles:
    def test_output_shape(self):
        lm = _make_open_hand()
        angles = compute_abduction_angles(lm)
        assert angles.shape == (4,)

    def test_values_in_range(self):
        lm = _make_open_hand()
        angles = compute_abduction_angles(lm)
        assert np.all(angles >= 0.0)
        assert np.all(angles <= np.pi + 1e-6)


class TestPalmNormal:
    def test_open_hand_normal_z(self):
        """For a hand in the XY plane, normal should point along Z."""
        lm = _make_open_hand()
        normal = compute_palm_normal(lm)
        assert normal.shape == (3,)
        # Normal nearly along Z axis
        assert abs(float(normal[2])) > 0.9

    def test_unit_length(self):
        lm = _make_open_hand()
        normal = compute_palm_normal(lm)
        norm = float(np.linalg.norm(normal))
        assert abs(norm - 1.0) < 1e-5 or norm < 1e-8  # zero or unit

    def test_degenerate_returns_zeros(self):
        lm = np.zeros((21, 3), dtype=np.float32)
        normal = compute_palm_normal(lm)
        assert np.all(normal == 0.0)


class TestFingertipDistances:
    def test_output_shape(self):
        lm = _make_open_hand()
        dists = compute_fingertip_distances(lm)
        assert dists.shape == (5,)

    def test_non_negative(self):
        lm = _make_open_hand()
        dists = compute_fingertip_distances(lm)
        assert np.all(dists >= 0.0)


class TestGlobalCurvature:
    def test_open_hand_higher_than_fist(self):
        open_h = _make_open_hand()
        fist = _make_fist()
        curv_open = compute_global_curvature(open_h)[0]
        curv_fist = compute_global_curvature(fist)[0]
        # Open hand should have higher curvature score (tips farther from wrist rel to MCP)
        assert curv_open >= curv_fist

    def test_output_shape(self):
        lm = _make_open_hand()
        curv = compute_global_curvature(lm)
        assert curv.shape == (1,)


class TestExtractHandshapeFeatures:
    def test_output_dim(self):
        lm = _make_open_hand()
        feats = extract_handshape_features(lm)
        assert feats.shape == (HANDSHAPE_FEATURE_DIM,), f"Expected {HANDSHAPE_FEATURE_DIM}, got {feats.shape}"
        assert HANDSHAPE_FEATURE_DIM == 42

    def test_dtype_float32(self):
        lm = _make_open_hand()
        assert extract_handshape_features(lm).dtype == np.float32

    def test_absent_hand_zeros(self):
        lm = np.zeros((21, 3), dtype=np.float32)
        feats = extract_handshape_features(lm)
        assert np.all(feats == 0.0)

    def test_flat_input_accepted(self):
        lm = _make_open_hand().reshape(-1)
        feats = extract_handshape_features(lm)
        assert feats.shape == (HANDSHAPE_FEATURE_DIM,)

    def test_no_nan_or_inf(self):
        lm = _make_open_hand()
        feats = extract_handshape_features(lm)
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))

    def test_open_vs_fist_differ(self):
        feats_open = extract_handshape_features(_make_open_hand())
        feats_fist = extract_handshape_features(_make_fist())
        # Features should differ meaningfully
        diff = float(np.mean(np.abs(feats_open - feats_fist)))
        assert diff > 1e-4


class TestExtractBothHandsHandshape:
    def test_output_dim_84(self):
        lm_open = _make_open_hand()
        lm_fist = _make_fist()
        feats = extract_both_hands_handshape(lm_open, lm_fist)
        assert feats.shape == (84,)

    def test_layout_left_right(self):
        lm_open = _make_open_hand()
        lm_zero = np.zeros((21, 3), dtype=np.float32)
        feats_left_open = extract_both_hands_handshape(lm_open, lm_zero)
        feats_right_open = extract_both_hands_handshape(lm_zero, lm_open)
        # Left part non-zero, right part zero (and vice versa)
        assert not np.all(feats_left_open[:42] == 0.0)
        assert np.all(feats_left_open[42:] == 0.0)
        assert np.all(feats_right_open[:42] == 0.0)
        assert not np.all(feats_right_open[42:] == 0.0)

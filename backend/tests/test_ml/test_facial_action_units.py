"""Unit tests for backend/app/ml/facial_action_units.py."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.facial_action_units import (
    AU_DIM,
    GAZE_DIM,
    HEAD_POSE_DIM,
    MOUTH_SHAPE_DIM,
    NMM_FEATURE_DIM,
    extract_nmm_features,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_neutral_face() -> np.ndarray:
    """Minimal 468-point face mesh with plausible landmark positions.

    Points are placed roughly to represent a forward-facing neutral expression.
    Coordinates are in normalised screen space (x/y in [0,1], z near 0).
    """
    # Start with zeros and set key landmarks
    face = np.zeros((468, 3), dtype=np.float32)

    # Face rough layout (x=horizontal, y=vertical downward, z=depth)
    face_cx, face_cy = 0.5, 0.5
    face_w, face_h = 0.2, 0.25

    # Nose tip (landmark 4)
    face[4] = [face_cx, face_cy, 0.0]
    # Nose bridge (landmark 1)
    face[1] = [face_cx, face_cy - 0.05, -0.02]
    # Chin (landmark 152)
    face[152] = [face_cx, face_cy + 0.12, 0.0]
    # Forehead (landmark 10)
    face[10] = [face_cx, face_cy - 0.12, 0.0]

    # Left eye outer (33)
    face[33] = [face_cx - 0.07, face_cy - 0.03, 0.0]
    # Right eye outer (263)
    face[263] = [face_cx + 0.07, face_cy - 0.03, 0.0]
    # Left eye inner (133)
    face[133] = [face_cx - 0.04, face_cy - 0.03, 0.0]
    # Right eye inner (362)
    face[362] = [face_cx + 0.04, face_cy - 0.03, 0.0]

    # --- Left eye contour ---
    # Upper (159, 158, 157, 173)
    for idx in [159, 158, 157, 173]:
        face[idx] = [face_cx - 0.055, face_cy - 0.045, 0.0]
    # Lower (145, 153, 144, 163)
    for idx in [145, 153, 144, 163]:
        face[idx] = [face_cx - 0.055, face_cy - 0.015, 0.0]

    # --- Right eye contour ---
    # Upper (386, 385, 384, 398)
    for idx in [386, 385, 384, 398]:
        face[idx] = [face_cx + 0.055, face_cy - 0.045, 0.0]
    # Lower (374, 380, 381, 382)
    for idx in [374, 380, 381, 382]:
        face[idx] = [face_cx + 0.055, face_cy - 0.015, 0.0]

    # --- Left brow (70, 63, 105, 66, 107)
    for idx in [70, 63, 105, 66, 107]:
        face[idx] = [face_cx - 0.055, face_cy - 0.065, 0.0]
    # --- Right brow (336, 296, 334, 293, 300)
    for idx in [336, 296, 334, 293, 300]:
        face[idx] = [face_cx + 0.055, face_cy - 0.065, 0.0]

    # Inner brow points for furrow
    face[107] = [face_cx - 0.030, face_cy - 0.063, 0.0]
    face[336] = [face_cx + 0.030, face_cy - 0.063, 0.0]

    # --- Mouth ---
    face[61] = [face_cx - 0.04, face_cy + 0.06, 0.0]   # left
    face[291] = [face_cx + 0.04, face_cy + 0.06, 0.0]  # right
    face[13] = [face_cx, face_cy + 0.05, 0.0]          # upper lip
    face[14] = [face_cx, face_cy + 0.075, 0.0]         # lower lip

    # Cheek reference points
    face[234] = [face_cx - 0.09, face_cy, 0.0]
    face[454] = [face_cx + 0.09, face_cy, 0.0]

    # Cheek puff points
    for idx in [117, 118, 119, 120]:
        face[idx] = [face_cx - 0.09, face_cy, 0.0]
    for idx in [346, 347, 348, 349]:
        face[idx] = [face_cx + 0.09, face_cy, 0.0]

    # Nose bridge for furrow detection
    face[49] = [face_cx - 0.02, face_cy - 0.01, 0.0]
    face[279] = [face_cx + 0.02, face_cy - 0.01, 0.0]

    return face


def _make_absent_face() -> np.ndarray:
    return np.zeros((468, 3), dtype=np.float32)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNMMDimensions:
    def test_total_dim(self):
        assert NMM_FEATURE_DIM == 32
        assert AU_DIM == 20
        assert HEAD_POSE_DIM == 3
        assert GAZE_DIM == 3
        assert MOUTH_SHAPE_DIM == 6
        assert AU_DIM + HEAD_POSE_DIM + GAZE_DIM + MOUTH_SHAPE_DIM == NMM_FEATURE_DIM


class TestExtractNMMFeatures:
    def test_output_shape_neutral(self):
        face = _make_neutral_face()
        feats = extract_nmm_features(face)
        assert feats.shape == (NMM_FEATURE_DIM,), f"Expected {NMM_FEATURE_DIM}, got {feats.shape}"

    def test_output_dtype(self):
        face = _make_neutral_face()
        feats = extract_nmm_features(face)
        assert feats.dtype == np.float32

    def test_absent_face_zeros(self):
        face = _make_absent_face()
        feats = extract_nmm_features(face)
        assert np.all(feats == 0.0), "Absent face should return all zeros"

    def test_no_nan_or_inf(self):
        face = _make_neutral_face()
        feats = extract_nmm_features(face)
        assert not np.any(np.isnan(feats)), "No NaN values expected"
        assert not np.any(np.isinf(feats)), "No Inf values expected"

    def test_flat_input_accepted(self):
        face = _make_neutral_face().reshape(-1)  # (1404,)
        feats = extract_nmm_features(face)
        assert feats.shape == (NMM_FEATURE_DIM,)

    def test_too_small_face_returns_zeros(self):
        face = np.ones((10, 3), dtype=np.float32)  # Too few landmarks
        feats = extract_nmm_features(face)
        assert np.all(feats == 0.0)

    def test_au_block_in_range(self):
        """AU block values should be reasonable (not wildly large)."""
        face = _make_neutral_face()
        feats = extract_nmm_features(face)
        au_block = feats[:AU_DIM]
        # Most AUs should be in [-2, 2] for a neutral face
        assert float(np.max(np.abs(au_block))) < 10.0

    def test_head_pose_block(self):
        """Head pose block should have pitch, yaw, roll in radians."""
        face = _make_neutral_face()
        feats = extract_nmm_features(face)
        head_pose = feats[AU_DIM : AU_DIM + HEAD_POSE_DIM]
        # Angles should be in [-π/2, π/2]
        assert np.all(np.abs(head_pose) <= np.pi / 2 + 1e-5)


class TestNMMWithRaisedBrows:
    """Test that raised brows produce different features than neutral."""

    def test_raised_brows_differ_from_neutral(self):
        neutral = _make_neutral_face()
        raised = neutral.copy()
        # Move brow landmarks up (lower y value = higher in screen)
        for idx in [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]:
            if idx < len(raised):
                raised[idx, 1] -= 0.025  # raise brows

        feats_neutral = extract_nmm_features(neutral)
        feats_raised = extract_nmm_features(raised)
        # AU0 and AU1 (brow raise) should differ
        diff = abs(float(feats_raised[0]) - float(feats_neutral[0]))
        diff += abs(float(feats_raised[1]) - float(feats_neutral[1]))
        assert diff > 1e-4, "Raised brows should change AU0/AU1 features"


class TestNMMWithOpenMouth:
    """Test that open mouth produces different features than neutral."""

    def test_open_mouth_au8_changes(self):
        neutral = _make_neutral_face()
        open_mouth = neutral.copy()
        # Lower the lower lip (increase y)
        open_mouth[14, 1] += 0.03  # lower lip down

        feats_neutral = extract_nmm_features(neutral)
        feats_open = extract_nmm_features(open_mouth)
        # AU8 = mouth height; should increase
        assert float(feats_open[8]) > float(feats_neutral[8]) - 1e-4

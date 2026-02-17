"""Tests for per-frame landmark normalization and facial expression features."""

from __future__ import annotations

import numpy as np

from app.ml.feature_engineering import FACIAL_EXPRESSION_FEATURE_DIM, RAW_LANDMARK_FEATURE_DIM
from app.ml.features import (
    FACIAL_FEATURE_MOUTH_DETECTION_SCORE,
    FACIAL_FEATURE_MOUTH_OPEN_FLAG,
    FrameLandmarks,
    normalize_landmarks,
)


def _blank_face() -> list[list[float]]:
    return [[0.0, 0.0, 0.0] for _ in range(468)]


def test_normalize_landmarks_appends_facial_expression_block() -> None:
    frame = FrameLandmarks(
        left_hand=[[0.0, 0.0, 0.0]] * 21,
        right_hand=[[0.0, 0.0, 0.0]] * 21,
        pose=[[0.0, 0.0, 0.0]] * 33,
        face=None,
    )

    normalized = normalize_landmarks(
        frame,
        include_face=False,
        include_face_expressions=True,
    )

    assert normalized.shape == (RAW_LANDMARK_FEATURE_DIM,)
    assert np.allclose(normalized[-FACIAL_EXPRESSION_FEATURE_DIM:], 0.0)


def test_normalize_landmarks_detects_open_mouth_from_face_landmarks() -> None:
    face = _blank_face()
    face[61] = [0.45, 0.50, 0.0]   # mouth left
    face[291] = [0.55, 0.50, 0.0]  # mouth right
    face[13] = [0.50, 0.47, 0.0]   # upper lip
    face[14] = [0.50, 0.55, 0.0]   # lower lip
    face[159] = [0.48, 0.40, 0.0]  # left eye upper
    face[145] = [0.48, 0.44, 0.0]  # left eye lower
    face[386] = [0.52, 0.40, 0.0]  # right eye upper
    face[374] = [0.52, 0.44, 0.0]  # right eye lower
    face[70] = [0.47, 0.34, 0.0]   # left brow
    face[300] = [0.53, 0.34, 0.0]  # right brow

    frame = FrameLandmarks(
        left_hand=[[0.0, 0.0, 0.0]] * 21,
        right_hand=[[0.0, 0.0, 0.0]] * 21,
        pose=[[0.0, 0.0, 0.0]] * 33,
        face=face,
    )

    normalized = normalize_landmarks(
        frame,
        include_face=False,
        include_face_expressions=True,
    )
    facial = normalized[-FACIAL_EXPRESSION_FEATURE_DIM:]

    assert facial[FACIAL_FEATURE_MOUTH_DETECTION_SCORE] >= 0.9
    assert facial[FACIAL_FEATURE_MOUTH_OPEN_FLAG] == 1.0

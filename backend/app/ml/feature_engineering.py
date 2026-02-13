"""Enriched feature computation for sign language landmark sequences.

Transforms raw 225-dimensional landmark coordinates into ~469 features
that capture motion, hand shape, and spatial relationships.

Feature layout of input (225 dims):
  - [0:63]   left hand  (21 points * 3 coords)
  - [63:126] right hand (21 points * 3 coords)
  - [126:225] pose      (33 points * 3 coords)
"""
from __future__ import annotations

import numpy as np

# Landmark index constants
_LEFT_HAND_START = 0
_LEFT_HAND_END = 63
_RIGHT_HAND_START = 63
_RIGHT_HAND_END = 126
_POSE_START = 126
_POSE_END = 225

# Pose landmark indices (within pose block, each point = 3 values)
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16

# Hand landmark indices (within each hand block, each point = 3 values)
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_RING_TIP = 16
_PINKY_TIP = 20

# Enriched feature dimension
ENRICHED_FEATURE_DIM = 225 + 225 + 5 + 4 + 10  # 469


def _pose_point(seq: np.ndarray, point_idx: int) -> np.ndarray:
    """Extract a pose point [num_frames, 3] from raw sequence."""
    start = _POSE_START + point_idx * 3
    return seq[:, start:start + 3]


def _hand_point(seq: np.ndarray, hand_start: int, point_idx: int) -> np.ndarray:
    """Extract a hand point [num_frames, 3] from raw sequence."""
    start = hand_start + point_idx * 3
    return seq[:, start:start + 3]


def _safe_norm(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute norm with epsilon to avoid division by zero."""
    return np.linalg.norm(vectors, axis=axis).clip(min=1e-8)


def compute_velocities(seq: np.ndarray) -> np.ndarray:
    """Frame-to-frame differences. First frame is zero."""
    vel = np.zeros_like(seq)
    if seq.shape[0] > 1:
        vel[1:] = seq[1:] - seq[:-1]
    return vel


def compute_hand_distances(seq: np.ndarray) -> np.ndarray:
    """5 inter-hand distances: wrist-wrist + 4 fingertip pairs."""
    num_frames = seq.shape[0]
    distances = np.zeros((num_frames, 5), dtype=np.float32)

    l_wrist = _hand_point(seq, _LEFT_HAND_START, _WRIST)
    r_wrist = _hand_point(seq, _RIGHT_HAND_START, _WRIST)
    distances[:, 0] = _safe_norm(l_wrist - r_wrist)

    for i, tip_idx in enumerate([_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _PINKY_TIP]):
        l_tip = _hand_point(seq, _LEFT_HAND_START, tip_idx)
        r_tip = _hand_point(seq, _RIGHT_HAND_START, tip_idx)
        distances[:, i + 1] = _safe_norm(l_tip - r_tip)

    return distances


def compute_joint_angles(seq: np.ndarray) -> np.ndarray:
    """4 arm joint angles: left/right elbow flexion, left/right wrist flexion."""
    num_frames = seq.shape[0]
    angles = np.zeros((num_frames, 4), dtype=np.float32)

    l_shoulder = _pose_point(seq, _LEFT_SHOULDER)
    r_shoulder = _pose_point(seq, _RIGHT_SHOULDER)
    l_elbow = _pose_point(seq, _LEFT_ELBOW)
    r_elbow = _pose_point(seq, _RIGHT_ELBOW)
    l_wrist = _pose_point(seq, _LEFT_WRIST)
    r_wrist = _pose_point(seq, _RIGHT_WRIST)

    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Angle at vertex b formed by segments a-b and c-b."""
        ba = a - b
        bc = c - b
        cos = np.sum(ba * bc, axis=-1) / (_safe_norm(ba) * _safe_norm(bc))
        return np.arccos(np.clip(cos, -1.0, 1.0))

    angles[:, 0] = _angle(l_shoulder, l_elbow, l_wrist)
    angles[:, 1] = _angle(r_shoulder, r_elbow, r_wrist)

    l_hand_wrist = _hand_point(seq, _LEFT_HAND_START, _WRIST)
    r_hand_wrist = _hand_point(seq, _RIGHT_HAND_START, _WRIST)
    l_hand_middle = _hand_point(seq, _LEFT_HAND_START, _MIDDLE_TIP)
    r_hand_middle = _hand_point(seq, _RIGHT_HAND_START, _MIDDLE_TIP)

    angles[:, 2] = _angle(l_elbow, l_hand_wrist, l_hand_middle)
    angles[:, 3] = _angle(r_elbow, r_hand_wrist, r_hand_middle)

    return np.nan_to_num(angles, nan=0.0).astype(np.float32)


def compute_hand_shape_features(seq: np.ndarray) -> np.ndarray:
    """10 hand shape features: 5 per hand (openness, 4 finger extensions)."""
    num_frames = seq.shape[0]
    features = np.zeros((num_frames, 10), dtype=np.float32)

    for hand_idx, hand_start in enumerate([_LEFT_HAND_START, _RIGHT_HAND_START]):
        wrist = _hand_point(seq, hand_start, _WRIST)
        tips = [_hand_point(seq, hand_start, t) for t in [_INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP]]

        tip_dists = np.stack([_safe_norm(tip - wrist) for tip in tips], axis=-1)
        features[:, hand_idx * 5] = tip_dists.mean(axis=-1)

        for i, tip in enumerate(tips):
            features[:, hand_idx * 5 + 1 + i] = _safe_norm(tip - wrist)

    return features


def compute_enriched_features(seq: np.ndarray) -> np.ndarray:
    """
    Compute enriched features from raw 225-dim landmark sequence.

    Input:  [num_frames, 225]
    Output: [num_frames, ENRICHED_FEATURE_DIM]  (469)
    """
    velocities = compute_velocities(seq)
    hand_distances = compute_hand_distances(seq)
    joint_angles = compute_joint_angles(seq)
    hand_shapes = compute_hand_shape_features(seq)

    enriched = np.concatenate([
        seq,
        velocities,
        hand_distances,
        joint_angles,
        hand_shapes,
    ], axis=1)

    return np.nan_to_num(enriched, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

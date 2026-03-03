"""Signing space encoder for 3D hand-position features.

Encodes the spatial relationship between the hands and the body —
the "signing space" — which is essential for directional verbs and spatial
grammar in LSFB (Langue des Signes de Belgique Francophone).

MediaPipe Pose landmark indices used:
  11 = left shoulder
  12 = right shoulder
  23 = left hip
  24 = right hip

Output layout (18 dims):
  [0:3]   dominant-hand position relative to right shoulder (3 dims)
  [3:6]   non-dominant hand position relative to left shoulder (3 dims)
  [6:9]   inter-hand vector (right_wrist − left_wrist) (3 dims)
  [9:12]  vertical zone one-hot (low/mid/high) (3 dims OHE)
  [12:14] lateral dominance (right-of-body, left-of-body) soft scores (2 dims)
  [14:15] depth from body plane — forward extension (1 dim)
  [15:16] inter-hand distance normalized by shoulder width (1 dim)
  [16:18] 2-D motion direction (optical-flow proxy, zeros at first call) (2 dims)

Total: 18 dims
"""

from __future__ import annotations

import numpy as np

# ── Pose landmark index constants ────────────────────────────────────────────

_LEFT_SHOULDER_IDX = 11
_RIGHT_SHOULDER_IDX = 12
_LEFT_ELBOW_IDX = 13
_RIGHT_ELBOW_IDX = 14
_LEFT_WRIST_IDX = 15
_RIGHT_WRIST_IDX = 16
_LEFT_HIP_IDX = 23
_RIGHT_HIP_IDX = 24

# Signing space dimension breakdown
DOMINANT_REL_DIM = 3
NONDOMINANT_REL_DIM = 3
INTERHAND_VEC_DIM = 3
ZONE_DIM = 3        # one-hot: low / mid / high
LATERALITY_DIM = 2  # right-of-midline, left-of-midline
DEPTH_DIM = 1
INTERHAND_DIST_DIM = 1
MOTION_DIM = 2

SIGNING_SPACE_FEATURE_DIM = (
    DOMINANT_REL_DIM
    + NONDOMINANT_REL_DIM
    + INTERHAND_VEC_DIM
    + ZONE_DIM
    + LATERALITY_DIM
    + DEPTH_DIM
    + INTERHAND_DIST_DIM
    + MOTION_DIM
)  # = 18

# Vertical zone boundaries (in shoulder-normalized coordinates)
# Low: below shoulder midpoint (body / lower space)
# Mid: between shoulder and head (~torse)
# High: above shoulder height (head / upper space)
_ZONE_LOW_THRESH = -0.1    # below shoulder center
_ZONE_HIGH_THRESH = 0.25   # above shoulder center


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_pose_point(pose: np.ndarray, idx: int) -> np.ndarray:
    """Safely extract a pose landmark as xyz array.

    Pose landmarks may have 3 or 4 values (x, y, z, [visibility]).
    """
    if idx < 0 or idx >= pose.shape[0]:
        return np.zeros(3, dtype=np.float32)
    return pose[idx, :3].astype(np.float32)


def _hand_center(hand_landmarks: np.ndarray | None) -> np.ndarray | None:
    """Compute the hand centre from 21 hand landmarks (wrist + mean of all pts)."""
    if hand_landmarks is None:
        return None
    lm = np.asarray(hand_landmarks, dtype=np.float32)
    if lm.ndim == 1:
        if lm.shape[0] < 3:
            return None
        lm = lm[:63].reshape(-1, 3) if lm.shape[0] >= 63 else lm[:3].reshape(1, 3)
    # All-zero guard (absent hand)
    if float(np.max(np.abs(lm))) < 1e-9:
        return None
    return lm[0, :3].astype(np.float32)  # wrist as representative point


def _vertical_zone(rel_y: float) -> np.ndarray:
    """One-hot encode the vertical signing zone.

    rel_y is the y-coordinate of the hand relative to the shoulder midpoint,
    in shoulder-normalized space (positive = up in image coords means negative
    y in MediaPipe world space where y increases downward).

    In normalized body space (after hip-centre subtraction and shoulder scaling):
      High (head/upper) : rel_y_norm < _ZONE_HIGH_THRESH (higher in world)
      Mid (torse)       : between thresholds
      Low (body/waist)  : rel_y_norm > _ZONE_LOW_THRESH  (lower in world)
    """
    ohe = np.zeros(3, dtype=np.float32)
    # MediaPipe uses screen coordinate y (0=top). After normalizing:
    # hand above shoulders → y < shoulder_y → body-frame y is negative
    if rel_y < -_ZONE_HIGH_THRESH:
        ohe[2] = 1.0  # high zone
    elif rel_y > _ZONE_LOW_THRESH:
        ohe[0] = 1.0  # low zone
    else:
        ohe[1] = 1.0  # mid zone
    return ohe


# ── Public API ───────────────────────────────────────────────────────────────

def extract_signing_space(
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    pose_landmarks: np.ndarray,
    prev_right_wrist: np.ndarray | None = None,
    prev_left_wrist: np.ndarray | None = None,
) -> np.ndarray:
    """Extract 18-dim signing space features.

    Args:
        left_hand: (21, 3) or (63,) left-hand landmarks; None / all-zeros if absent.
        right_hand: (21, 3) or (63,) right-hand landmarks; None / all-zeros if absent.
        pose_landmarks: (33, 3) or (33, 4) BlazePose landmarks.
            Must contain at least 25 points.
        prev_right_wrist: Optional previous right wrist position for motion (3,).
        prev_left_wrist:  Optional previous left wrist position for motion (3,).

    Returns:
        Array of shape (18,) dtype float32.
    """
    features = np.zeros(SIGNING_SPACE_FEATURE_DIM, dtype=np.float32)

    # ── Parse pose ────────────────────────────────────────────────────────────
    pose = np.asarray(pose_landmarks, dtype=np.float32)
    if pose.ndim == 1:
        # Accept flat pose vector (33*3=99 or 33*4=132)
        n_pts = pose.shape[0] // 3
        pose = pose[: n_pts * 3].reshape(n_pts, 3)

    if pose.shape[0] < 25:
        return features  # Cannot compute without shoulder/hip reference

    r_shoulder = _get_pose_point(pose, _RIGHT_SHOULDER_IDX)
    l_shoulder = _get_pose_point(pose, _LEFT_SHOULDER_IDX)
    r_hip = _get_pose_point(pose, _RIGHT_HIP_IDX)
    l_hip = _get_pose_point(pose, _LEFT_HIP_IDX)

    shoulder_mid = (r_shoulder + l_shoulder) * 0.5
    shoulder_width = float(np.linalg.norm(r_shoulder - l_shoulder))
    scale = max(shoulder_width, 1e-6)

    # Body depth reference: mean z of shoulders
    body_plane_z = float(0.5 * (r_shoulder[2] + l_shoulder[2]))

    # ── Hand wrist positions ──────────────────────────────────────────────────
    r_wrist = _hand_center(right_hand)
    l_wrist = _hand_center(left_hand)

    # Fallback: use pose wrist if hand not detected
    if r_wrist is None:
        pose_r_wrist = _get_pose_point(pose, _RIGHT_WRIST_IDX)
        if float(np.linalg.norm(pose_r_wrist)) > 1e-6:
            r_wrist = pose_r_wrist

    if l_wrist is None:
        pose_l_wrist = _get_pose_point(pose, _LEFT_WRIST_IDX)
        if float(np.linalg.norm(pose_l_wrist)) > 1e-6:
            l_wrist = pose_l_wrist

    # ── Feature 0:3 — dominant hand relative to right shoulder ───────────────
    if r_wrist is not None:
        rel_dominant = (r_wrist - r_shoulder) / scale
        features[0:3] = np.clip(rel_dominant, -3.0, 3.0)

    # ── Feature 3:6 — non-dominant hand relative to left shoulder ────────────
    if l_wrist is not None:
        rel_nondominant = (l_wrist - l_shoulder) / scale
        features[3:6] = np.clip(rel_nondominant, -3.0, 3.0)

    # ── Feature 6:9 — inter-hand vector ──────────────────────────────────────
    if r_wrist is not None and l_wrist is not None:
        inter_hand = (r_wrist - l_wrist) / scale
        features[6:9] = np.clip(inter_hand, -3.0, 3.0)

    # ── Feature 9:12 — vertical zone (dominant hand) ─────────────────────────
    if r_wrist is not None:
        rel_y = float((r_wrist[1] - shoulder_mid[1]) / scale)
        features[9:12] = _vertical_zone(rel_y)
    else:
        features[10] = 1.0  # default to mid zone

    # ── Feature 12:14 — lateral dominance ────────────────────────────────────
    # right-of-midline score, left-of-midline score
    mid_x = float(shoulder_mid[0])
    if r_wrist is not None:
        r_lat = float(r_wrist[0] - mid_x) / scale
        features[12] = float(np.clip(r_lat, -1.5, 1.5))
    if l_wrist is not None:
        l_lat = float(mid_x - l_wrist[0]) / scale
        features[13] = float(np.clip(l_lat, -1.5, 1.5))

    # ── Feature 14 — depth from body plane ───────────────────────────────────
    # Positive = hands extended forward (away from camera in many setups)
    if r_wrist is not None:
        depth_ext = float(r_wrist[2] - body_plane_z) / scale
        features[14] = float(np.clip(depth_ext, -2.0, 2.0))

    # ── Feature 15 — inter-hand distance normalized ───────────────────────────
    if r_wrist is not None and l_wrist is not None:
        inter_dist = float(np.linalg.norm(r_wrist - l_wrist)) / scale
        features[15] = float(np.clip(inter_dist, 0.0, 3.0))

    # ── Feature 16:18 — 2D motion direction (xy optical-flow proxy) ──────────
    motion_x = 0.0
    motion_y = 0.0
    motion_count = 0

    if r_wrist is not None and prev_right_wrist is not None:
        delta_r = (r_wrist - np.asarray(prev_right_wrist, dtype=np.float32))[:2] / scale
        motion_x += float(delta_r[0])
        motion_y += float(delta_r[1])
        motion_count += 1

    if l_wrist is not None and prev_left_wrist is not None:
        delta_l = (l_wrist - np.asarray(prev_left_wrist, dtype=np.float32))[:2] / scale
        motion_x += float(delta_l[0])
        motion_y += float(delta_l[1])
        motion_count += 1

    if motion_count > 0:
        features[16] = float(np.clip(motion_x / motion_count, -1.0, 1.0))
        features[17] = float(np.clip(motion_y / motion_count, -1.0, 1.0))

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

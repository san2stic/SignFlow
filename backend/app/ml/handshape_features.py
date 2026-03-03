"""Handshape feature extraction from MediaPipe hand landmarks.

Extracts rich per-hand features encoding finger configuration (handshape),
which carries approximately 40% of sign information in LSFB.

MediaPipe hand landmark indices (21 points per hand):
  0: WRIST
  1-4: THUMB  (CMC=1, MCP=2, IP=3,  TIP=4)
  5-8: INDEX  (MCP=5, PIP=6, DIP=7,  TIP=8)
  9-12: MIDDLE (MCP=9, PIP=10, DIP=11, TIP=12)
  13-16: RING  (MCP=13, PIP=14, DIP=15, TIP=16)
  17-20: PINKY (MCP=17, PIP=18, DIP=19, TIP=20)

Output layout per hand (42 dims):
  [0:15]  inter-phalange flexion angles (15 angles)
  [15:19] abduction angles between fingers (4 angles)
  [19:22] palm orientation normal vector (3 dims)
  [22:27] fingertip-to-fingertip distances (5 dims)
  [27:28] global curvature scalar (1 dim)
  [28:42] handshape embedding — linear projection to 14 dims

Total per hand: 42 dims
Total for 2 hands: 84 dims
"""

from __future__ import annotations

import numpy as np

# ── Landmark index constants ─────────────────────────────────────────────────

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Finger chains: (proximal_base, mid, distal, tip) for angles at mid/distal
# Each tuple is (a, b, c) → angle at vertex b
_FINGER_ANGLE_TRIPLETS: list[tuple[int, int, int]] = [
    # Thumb: CMC-MCP-IP, MCP-IP-TIP
    (THUMB_CMC,  THUMB_MCP, THUMB_IP),
    (THUMB_MCP,  THUMB_IP,  THUMB_TIP),
    # Index: MCP-PIP-DIP, PIP-DIP-TIP
    (INDEX_MCP,  INDEX_PIP,  INDEX_DIP),
    (INDEX_PIP,  INDEX_DIP,  INDEX_TIP),
    # Middle: MCP-PIP-DIP, PIP-DIP-TIP
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    # Ring: MCP-PIP-DIP, PIP-DIP-TIP
    (RING_MCP,   RING_PIP,   RING_DIP),
    (RING_PIP,   RING_DIP,   RING_TIP),
    # Pinky: MCP-PIP-DIP, PIP-DIP-TIP
    (PINKY_MCP,  PINKY_PIP,  PINKY_DIP),
    (PINKY_PIP,  PINKY_DIP,  PINKY_TIP),
    # Wrist-base angles for each finger (knuckle raise)
    (WRIST, INDEX_MCP,  INDEX_TIP),
    (WRIST, MIDDLE_MCP, MIDDLE_TIP),
    (WRIST, RING_MCP,   RING_TIP),
    (WRIST, PINKY_MCP,  PINKY_TIP),
    # Thumb opposition
    (WRIST, THUMB_MCP, THUMB_TIP),
]  # 15 angles total

# Abduction triplets: angle at base MCP between adjacent finger MCPs
_ABDUCTION_TRIPLETS: list[tuple[int, int, int]] = [
    (INDEX_MCP,  WRIST, MIDDLE_MCP),
    (MIDDLE_MCP, WRIST, RING_MCP),
    (RING_MCP,   WRIST, PINKY_MCP),
    (THUMB_MCP,  WRIST, INDEX_MCP),
]  # 4 angles

# Fingertip pair distances: (a, b) → ‖tip_a - tip_b‖
_FINGERTIP_PAIRS: list[tuple[int, int]] = [
    (THUMB_TIP, INDEX_TIP),
    (THUMB_TIP, MIDDLE_TIP),
    (INDEX_TIP, MIDDLE_TIP),
    (INDEX_TIP, PINKY_TIP),
    (MIDDLE_TIP, PINKY_TIP),
]  # 5 distances

# Palm normal: computed from wrist → INDEX_MCP and wrist → PINKY_MCP
_PALM_BASE = WRIST
_PALM_A = INDEX_MCP
_PALM_B = PINKY_MCP

# Feature dimensions
FLEXION_DIM = 15
ABDUCTION_DIM = 4
PALM_NORMAL_DIM = 3
FINGERTIP_DIST_DIM = 5
CURVATURE_DIM = 1
EMBEDDING_DIM = 14

# Raw features before embedding
_RAW_DIM = FLEXION_DIM + ABDUCTION_DIM + PALM_NORMAL_DIM + FINGERTIP_DIST_DIM + CURVATURE_DIM
# = 28 raw dims

HANDSHAPE_FEATURE_DIM = _RAW_DIM + EMBEDDING_DIM  # = 42

# Fixed random projection for the lightweight linear embedding (seed=42)
_RNG = np.random.default_rng(42)
_EMBED_W = _RNG.standard_normal((_RAW_DIM, EMBEDDING_DIM)).astype(np.float32) * 0.1


# ── Internal helpers ─────────────────────────────────────────────────────────

def _safe_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute interior angle (radians) at vertex *b* formed by vectors b→a and b→c.

    Returns 0.0 on degenerate input.
    """
    ba = a - b
    bc = c - b
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_val = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))


# ── Public API ───────────────────────────────────────────────────────────────

def compute_finger_angles(landmarks: np.ndarray) -> np.ndarray:
    """Compute 15 inter-phalange flexion angles for one hand.

    Args:
        landmarks: Array of shape (21, 3) with xyz landmarks.

    Returns:
        Array of shape (15,) with angles in radians [0, π].
    """
    angles = np.zeros(FLEXION_DIM, dtype=np.float32)
    for i, (a_idx, b_idx, c_idx) in enumerate(_FINGER_ANGLE_TRIPLETS):
        angles[i] = _safe_angle(landmarks[a_idx], landmarks[b_idx], landmarks[c_idx])
    return angles


def compute_abduction_angles(landmarks: np.ndarray) -> np.ndarray:
    """Compute 4 inter-finger abduction angles.

    Args:
        landmarks: Array of shape (21, 3).

    Returns:
        Array of shape (4,) with abduction angles in radians.
    """
    angles = np.zeros(ABDUCTION_DIM, dtype=np.float32)
    for i, (a_idx, b_idx, c_idx) in enumerate(_ABDUCTION_TRIPLETS):
        angles[i] = _safe_angle(landmarks[a_idx], landmarks[b_idx], landmarks[c_idx])
    return angles


def compute_palm_normal(landmarks: np.ndarray) -> np.ndarray:
    """Compute unit normal vector of the palm plane (3 dims).

    The palm plane is defined by wrist, INDEX_MCP, and PINKY_MCP.

    Args:
        landmarks: Array of shape (21, 3).

    Returns:
        Array of shape (3,) — unit normal; zeros when degenerate.
    """
    v1 = landmarks[_PALM_A] - landmarks[_PALM_BASE]
    v2 = landmarks[_PALM_B] - landmarks[_PALM_BASE]
    normal = np.cross(v1, v2)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (normal / norm).astype(np.float32)


def compute_fingertip_distances(landmarks: np.ndarray) -> np.ndarray:
    """Compute 5 inter-fingertip distances, normalized by hand span.

    Hand span is estimated as the wrist-to-MIDDLE_TIP distance.

    Args:
        landmarks: Array of shape (21, 3).

    Returns:
        Array of shape (5,) — normalized distances in [0, ∞).
    """
    hand_span = float(np.linalg.norm(landmarks[MIDDLE_TIP] - landmarks[WRIST]))
    scale = max(hand_span, 1e-6)

    dists = np.zeros(FINGERTIP_DIST_DIM, dtype=np.float32)
    for i, (a_idx, b_idx) in enumerate(_FINGERTIP_PAIRS):
        dists[i] = float(np.linalg.norm(landmarks[a_idx] - landmarks[b_idx])) / scale
    return dists


def compute_global_curvature(landmarks: np.ndarray) -> np.ndarray:
    """Compute global hand curvature scalar (1 dim): mean fingertip-to-wrist distance.

    1.0 ≈ fully open; 0.0 ≈ fully closed fist.

    Args:
        landmarks: Array of shape (21, 3).

    Returns:
        Array of shape (1,).
    """
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

    # Reference: distance from each MCP to wrist (open hand reference)
    mcp_dists = np.array(
        [float(np.linalg.norm(landmarks[m] - landmarks[WRIST])) for m in mcps],
        dtype=np.float32,
    )
    avg_mcp_dist = float(np.mean(mcp_dists))
    scale = max(avg_mcp_dist, 1e-6)

    tip_dists = np.array(
        [float(np.linalg.norm(landmarks[t] - landmarks[WRIST])) for t in tips],
        dtype=np.float32,
    )
    curvature = float(np.mean(tip_dists)) / scale
    return np.array([curvature], dtype=np.float32)


def extract_handshape_features(hand_landmarks_21x3: np.ndarray) -> np.ndarray:
    """Extract 42-dim handshape feature vector from 21 MediaPipe landmarks.

    Layout:
      [0:15]  flexion angles (15 dims)
      [15:19] abduction angles (4 dims)
      [19:22] palm normal (3 dims)
      [22:27] fingertip distances (5 dims)
      [27:28] global curvature (1 dim)
      [28:42] lightweight embedding (14 dims)

    Args:
        hand_landmarks_21x3: Array of shape (21, 3) or (63,).
            If all-zeros (hand absent), returns zero vector.

    Returns:
        Array of shape (42,) dtype float32.
    """
    # Accept flat input
    landmarks = np.asarray(hand_landmarks_21x3, dtype=np.float32)
    if landmarks.ndim == 1:
        if landmarks.shape[0] < 63:
            return np.zeros(HANDSHAPE_FEATURE_DIM, dtype=np.float32)
        landmarks = landmarks[:63].reshape(21, 3)
    elif landmarks.shape[0] < 21:
        return np.zeros(HANDSHAPE_FEATURE_DIM, dtype=np.float32)

    # Guard: absent hand (all zeros)
    if float(np.max(np.abs(landmarks))) < 1e-9:
        return np.zeros(HANDSHAPE_FEATURE_DIM, dtype=np.float32)

    flexion = compute_finger_angles(landmarks)
    abduction = compute_abduction_angles(landmarks)
    palm_normal = compute_palm_normal(landmarks)
    tip_dists = compute_fingertip_distances(landmarks)
    curvature = compute_global_curvature(landmarks)

    raw = np.concatenate([flexion, abduction, palm_normal, tip_dists, curvature])  # (28,)
    embedding = (raw @ _EMBED_W).astype(np.float32)  # (14,)

    features = np.concatenate([raw, embedding]).astype(np.float32)  # (42,)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_both_hands_handshape(
    left_landmarks: np.ndarray,
    right_landmarks: np.ndarray,
) -> np.ndarray:
    """Extract combined handshape features for both hands (84 dims).

    Layout: [left_42_dims | right_42_dims]

    Args:
        left_landmarks: (21, 3) or (63,) left-hand landmarks (zeros if absent).
        right_landmarks: (21, 3) or (63,) right-hand landmarks (zeros if absent).

    Returns:
        Array of shape (84,) dtype float32.
    """
    left_feats = extract_handshape_features(left_landmarks)
    right_feats = extract_handshape_features(right_landmarks)
    return np.concatenate([left_feats, right_feats]).astype(np.float32)

"""Non-Manual Marker (NMM) feature extraction from MediaPipe FaceMesh landmarks.

Extracts 32-dim Facial Action Unit (AU) descriptors relevant to LSFB grammar
(negation, WH-questions, yes/no questions, emphasis, spatial reference).

MediaPipe FaceMesh provides 468 landmarks (x, y, z).
This module derives compact, grammar-informed features instead of using all 1404 raw values.

Output layout (32 dims):
  [0:20]  Facial Action Units — 20 ratios/distances (AU1+2, AU4, AU5, AU6, AU12, AU25+26, …)
  [20:23] Head pose angles — pitch (nod), yaw (shake), roll (tilt) in radians
  [23:26] Gaze direction proxy — mean eye-centre displacement (3 dims)
  [26:32] Mouth shape PCA proxy — 6 relative mouth landmark displacements

Total: 32 dims

Key LSFB NMM:
  - Raised brows (AU1+AU2): yes/no questions
  - Furrowed brows (AU4): WH-questions, negation
  - Cheek puff / wide eyes: emphasis intensifiers
  - Mouth aperture (AU25+AU26): speech-mouth approximations in LSFB
  - Head nod (pitch↑): affirmation
  - Head shake (yaw oscillation): negation
"""

from __future__ import annotations

import numpy as np

# ── FaceMesh landmark index groups (MediaPipe canonical) ────────────────────

# Eyebrows (outer → inner order, left = wearer's left)
_LEFT_EYEBROW = [70, 63, 105, 66, 107]    # Left brow (5 pts)
_RIGHT_EYEBROW = [336, 296, 334, 293, 300] # Right brow (5 pts)

# Eyes (upper/lower contour)
_LEFT_EYE_UPPER = [159, 158, 157, 173]
_LEFT_EYE_LOWER = [145, 153, 144, 163]
_RIGHT_EYE_UPPER = [386, 385, 384, 398]
_RIGHT_EYE_LOWER = [374, 380, 381, 382]

# Iris-centre proxies (MediaPipe refinement landmarks, safe fallback to corners)
_LEFT_IRIS = 468   # Only available with refineFaceLandmarks; safe fallback below
_RIGHT_IRIS = 473

# Mouth
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_MOUTH_UPPER = 13
_MOUTH_LOWER = 14
_MOUTH_UPPER_INNER = [82, 13, 312]   # inner upper lip pts
_MOUTH_LOWER_INNER = [87, 14, 317]   # inner lower lip pts
_MOUTH_CORNERS = [61, 291]

# Nose tip (stable anchor for head pose)
_NOSE_TIP = 4
_CHIN = 152
_FOREHEAD = 10
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454

# Cheek points for puff detection
_LEFT_CHEEK_PUFF = [117, 118, 119, 120]
_RIGHT_CHEEK_PUFF = [346, 347, 348, 349]

# Face contour used for head-pose estimation (6-point PnP-lite)
# Rough canonical 3D positions (normalized to face size 1)
_POSE_ANCHOR_IDX = [1, 33, 263, 61, 291, 199]  # nose, l-eye-corner, r-eye-corner, mouth-l, mouth-r, chin
_CANONICAL_WORLD = np.array([
    [0.0,  0.0,   0.0],    # nose tip
    [-0.3,  0.3,  -0.05],  # left eye outer
    [ 0.3,  0.3,  -0.05],  # right eye outer
    [-0.2, -0.25, -0.05],  # mouth left
    [ 0.2, -0.25, -0.05],  # mouth right
    [ 0.0, -0.5,  -0.1],   # chin
], dtype=np.float32)

# ── Dimension constants ──────────────────────────────────────────────────────

AU_DIM = 20
HEAD_POSE_DIM = 3
GAZE_DIM = 3
MOUTH_SHAPE_DIM = 6

NMM_FEATURE_DIM = AU_DIM + HEAD_POSE_DIM + GAZE_DIM + MOUTH_SHAPE_DIM  # = 32


# ── Internal helpers ─────────────────────────────────────────────────────────

def _mean_point(face: np.ndarray, indices: list[int]) -> np.ndarray:
    """Return mean (x, y, z) of given landmark indices, zeros if any OOB."""
    valid = [i for i in indices if 0 <= i < face.shape[0]]
    if not valid:
        return np.zeros(3, dtype=np.float32)
    return face[valid].mean(axis=0).astype(np.float32)


def _dist_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in xy plane."""
    return float(np.linalg.norm(a[:2] - b[:2]))


def _dist_3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _safe_get(face: np.ndarray, idx: int) -> np.ndarray:
    if 0 <= idx < face.shape[0]:
        return face[idx].astype(np.float32)
    return np.zeros(3, dtype=np.float32)


# ── Action Unit computation ──────────────────────────────────────────────────

def _compute_au_features(face: np.ndarray) -> np.ndarray:
    """Compute 20 AU-inspired ratios.

    Index | Description                   | LSFB role
    ------|-------------------------------|-----------------------------
    0     | Left brow raise (AU1+AU2)      | Yes/no question (with AU2)
    1     | Right brow raise (AU1+AU2)     | Yes/no question
    2     | Brow asymmetry                | Marked utterance
    3     | Left brow furrow (AU4)        | WH-question / negation
    4     | Right brow furrow (AU4)       | WH-question / negation
    5     | Left eye width ratio (AU5)    | Surprise / emphasis
    6     | Right eye width ratio (AU5)   | Surprise / emphasis
    7     | Eye asymmetry                 | Expressive affect
    8     | Mouth height (AU26)           | Mouth movement / speech
    9     | Mouth width (AU20/AU25)       | Retract / spread
    10    | Mouth roundness               | /o/ mouth shape
    11    | Upper lip raise (AU10)        | Intensifier
    12    | Smile ratio (AU12)            | Social / affective
    13    | Lip corner left raise         | Directional affect
    14    | Lip corner right raise        | Directional affect
    15    | Left cheek puff               | Intensifier
    16    | Right cheek puff              | Intensifier
    17    | Jaw drop ratio (AU27+AU26)    | Surprise / strong negation
    18    | Nose wrinkle proxy (AU9)      | Negation intensifier
    19    | Overall expression intensity  | Global activation
    """
    aus = np.zeros(AU_DIM, dtype=np.float32)

    # --- Reference distances (for normalization) ---
    nose_tip = _safe_get(face, _NOSE_TIP)
    chin_pt = _safe_get(face, _CHIN)
    face_height = max(_dist_3d(nose_tip, chin_pt), 1e-6)

    left_eye_outer = _safe_get(face, 33)
    right_eye_outer = _safe_get(face, 263)
    face_width = max(_dist_3d(left_eye_outer, right_eye_outer), 1e-6)

    # --- Brow / eye positions ---
    l_brow_pt = _mean_point(face, _LEFT_EYEBROW)
    r_brow_pt = _mean_point(face, _RIGHT_EYEBROW)
    l_eye_upper_pt = _mean_point(face, _LEFT_EYE_UPPER)
    r_eye_upper_pt = _mean_point(face, _RIGHT_EYE_UPPER)
    l_eye_lower_pt = _mean_point(face, _LEFT_EYE_LOWER)
    r_eye_lower_pt = _mean_point(face, _RIGHT_EYE_LOWER)

    # AU0/1: brow raise = positive y offset of brow above eye upper
    # (FaceMesh y increases downward)
    l_brow_raise = float(l_eye_upper_pt[1] - l_brow_pt[1]) / face_height
    r_brow_raise = float(r_eye_upper_pt[1] - r_brow_pt[1]) / face_height
    aus[0] = np.clip(l_brow_raise, -1.0, 1.0)
    aus[1] = np.clip(r_brow_raise, -1.0, 1.0)
    aus[2] = float(abs(l_brow_raise - r_brow_raise))

    # AU3/4: furrow — horizontal distance between inner brow points
    l_inner_brow = _safe_get(face, 107)
    r_inner_brow = _safe_get(face, 336)
    furrow_dist = _dist_2d(l_inner_brow, r_inner_brow) / face_width
    # Small distance → furrowed
    aus[3] = float(np.clip(1.0 - furrow_dist, 0.0, 1.0))
    aus[4] = aus[3]  # symmetric approximation

    # AU5/6/7: eye openness
    l_eye_h = _dist_2d(l_eye_upper_pt, l_eye_lower_pt) / face_height
    r_eye_h = _dist_2d(r_eye_upper_pt, r_eye_lower_pt) / face_height
    aus[5] = float(np.clip(l_eye_h * 10, 0.0, 1.0))
    aus[6] = float(np.clip(r_eye_h * 10, 0.0, 1.0))
    aus[7] = float(abs(l_eye_h - r_eye_h) * 10)

    # AU8–12: mouth features
    mouth_up = _safe_get(face, _MOUTH_UPPER)
    mouth_dn = _safe_get(face, _MOUTH_LOWER)
    mouth_l = _safe_get(face, _MOUTH_LEFT)
    mouth_r = _safe_get(face, _MOUTH_RIGHT)

    mouth_height = _dist_2d(mouth_up, mouth_dn) / face_height
    mouth_width = _dist_2d(mouth_l, mouth_r) / face_width
    mouth_roundness = mouth_height / max(mouth_width, 1e-6)
    upper_lip_raise = float(mouth_up[1] - _safe_get(face, 13)[1]) / face_height

    aus[8] = float(np.clip(mouth_height * 5, 0.0, 1.0))
    aus[9] = float(np.clip(mouth_width, 0.0, 1.0))
    aus[10] = float(np.clip(mouth_roundness, 0.0, 2.0))
    aus[11] = float(np.clip(abs(upper_lip_raise) * 20, 0.0, 1.0))

    # AU12/13/14: smile and corner raises
    mouth_center_y = float(0.5 * (mouth_up[1] + mouth_dn[1]))
    l_corner_raise = (mouth_center_y - float(mouth_l[1])) / face_height
    r_corner_raise = (mouth_center_y - float(mouth_r[1])) / face_height
    smile = float(0.5 * (l_corner_raise + r_corner_raise))
    aus[12] = float(np.clip(smile * 20, -1.0, 1.0))
    aus[13] = float(np.clip(l_corner_raise * 20, -1.0, 1.0))
    aus[14] = float(np.clip(r_corner_raise * 20, -1.0, 1.0))

    # AU15/16: cheek puff — deviation of cheek points from face plane
    l_cheek_pt = _mean_point(face, _LEFT_CHEEK_PUFF)
    r_cheek_pt = _mean_point(face, _RIGHT_CHEEK_PUFF)
    l_cheek_ref = _safe_get(face, 234)
    r_cheek_ref = _safe_get(face, 454)
    aus[15] = float(np.clip(abs(l_cheek_pt[2] - l_cheek_ref[2]) * 10, 0.0, 1.0))
    aus[16] = float(np.clip(abs(r_cheek_pt[2] - r_cheek_ref[2]) * 10, 0.0, 1.0))

    # AU17: jaw drop — chin displacement
    jaw_ref = _safe_get(face, _NOSE_TIP)
    jaw_drop = float(chin_pt[1] - jaw_ref[1]) / face_height
    aus[17] = float(np.clip(jaw_drop, 0.0, 1.0))

    # AU18: nose wrinkle proxy — narrowing at nose bridge
    nose_bridge_l = _safe_get(face, 49)
    nose_bridge_r = _safe_get(face, 279)
    nose_width = _dist_2d(nose_bridge_l, nose_bridge_r) / face_width
    aus[18] = float(np.clip(1.0 - nose_width, 0.0, 1.0))

    # AU19: overall expression intensity
    intensity_vals = [
        float(abs(aus[0])), float(abs(aus[1])),
        float(aus[3]), float(aus[5]), float(aus[6]),
        float(abs(aus[8])), float(abs(aus[12])),
    ]
    aus[19] = float(np.mean(intensity_vals))

    return aus


# ── Head pose estimation ─────────────────────────────────────────────────────

def _compute_head_pose(face: np.ndarray) -> np.ndarray:
    """Estimate head pitch, yaw, roll (3 dims) using 6 stable anchor points.

    Returns angles in radians: [pitch, yaw, roll].
    pitch  > 0 → head up (nod up)
    yaw    > 0 → head turned right
    roll   > 0 → head tilted right

    Method: derive a local coordinate frame and project principal axes.
    This is a geometry-based approximation without full PnP.
    """
    angles = np.zeros(3, dtype=np.float32)

    # Extract key points
    nose = _safe_get(face, 1)        # nose bridge
    chin_pt = _safe_get(face, _CHIN)
    l_eye = _safe_get(face, 33)
    r_eye = _safe_get(face, 263)
    forehead = _safe_get(face, _FOREHEAD)

    # Check non-degenerate
    eye_vec = r_eye - l_eye
    vert_vec = chin_pt - forehead

    face_width = float(np.linalg.norm(eye_vec))
    face_height = float(np.linalg.norm(vert_vec))
    if face_width < 1e-6 or face_height < 1e-6:
        return angles

    # Roll: angle of eye-line with horizontal
    roll = float(np.arctan2(float(eye_vec[1]), float(eye_vec[0])))

    # Yaw: left-right asymmetry – compare distance from nose to each eye
    nose_to_l = float(np.linalg.norm(nose[:2] - l_eye[:2]))
    nose_to_r = float(np.linalg.norm(nose[:2] - r_eye[:2]))
    yaw_ratio = (nose_to_r - nose_to_l) / max(nose_to_r + nose_to_l, 1e-6)
    yaw = float(np.arcsin(np.clip(yaw_ratio, -1.0, 1.0)))

    # Pitch: z-component of nose tip relative to chin (depth proxy)
    nose_tip = _safe_get(face, _NOSE_TIP)
    pitch_proxy = float(nose_tip[2] - chin_pt[2]) / face_height
    pitch = float(np.arctan(pitch_proxy))

    angles[0] = np.clip(pitch, -np.pi / 2, np.pi / 2)
    angles[1] = np.clip(yaw,   -np.pi / 2, np.pi / 2)
    angles[2] = np.clip(roll,  -np.pi / 2, np.pi / 2)
    return angles


# ── Gaze direction ───────────────────────────────────────────────────────────

def _compute_gaze(face: np.ndarray) -> np.ndarray:
    """Estimate gaze direction proxy (3 dims) from eye corner and iris positions.

    Returns mean displacement of iris-centre from eye-centre, normalized.
    Iris landmarks (468, 473) are only available with refineFaceLandmarks.
    Falls back to eye centre when iris is zero.
    """
    gaze = np.zeros(3, dtype=np.float32)

    l_eye_center = _mean_point(face, _LEFT_EYE_UPPER + _LEFT_EYE_LOWER)
    r_eye_center = _mean_point(face, _RIGHT_EYE_UPPER + _RIGHT_EYE_LOWER)

    # Safe iris access (may be 0 if not refined)
    l_iris = _safe_get(face, 468) if face.shape[0] > 468 else np.zeros(3, dtype=np.float32)
    r_iris = _safe_get(face, 473) if face.shape[0] > 473 else np.zeros(3, dtype=np.float32)

    # If iris landmarks are available (non-zero)
    if float(np.linalg.norm(l_iris)) > 1e-6 and float(np.linalg.norm(r_iris)) > 1e-6:
        l_gaze = l_iris - l_eye_center
        r_gaze = r_iris - r_eye_center
        mean_gaze = (l_gaze + r_gaze) * 0.5
    else:
        # Fallback: pupil-to-corner asymmetry
        l_corner1 = _safe_get(face, 33)
        l_corner2 = _safe_get(face, 133)
        r_corner1 = _safe_get(face, 263)
        r_corner2 = _safe_get(face, 362)
        l_gaze = l_eye_center - (l_corner1 + l_corner2) * 0.5
        r_gaze = r_eye_center - (r_corner1 + r_corner2) * 0.5
        mean_gaze = (l_gaze + r_gaze) * 0.5

    norm = float(np.linalg.norm(mean_gaze))
    if norm < 1e-8:
        return gaze

    gaze[:] = np.clip(mean_gaze * 20.0, -1.0, 1.0)  # scale to ~[-1, 1] range
    return gaze.astype(np.float32)


# ── Mouth shape ──────────────────────────────────────────────────────────────

def _compute_mouth_shape(face: np.ndarray) -> np.ndarray:
    """Compute 6 relative mouth landmark displacements for shape encoding.

    Captures phoneme-mouth approximations relevant to LSFB (mouth-pictures).
    Layout: [height, width, roundness, left_droop, right_droop, upper_protrude]
    """
    mouth_l = _safe_get(face, _MOUTH_LEFT)
    mouth_r = _safe_get(face, _MOUTH_RIGHT)
    mouth_up = _safe_get(face, 13)
    mouth_dn = _safe_get(face, 14)
    upper_protrude_pt = _mean_point(face, [61, 82, 13, 312, 291])
    lower_protrude_pt = _mean_point(face, [61, 87, 14, 317, 291])

    face_height = max(
        _dist_3d(_safe_get(face, _NOSE_TIP), _safe_get(face, _CHIN)), 1e-6
    )
    face_width = max(
        _dist_3d(_safe_get(face, 33), _safe_get(face, 263)), 1e-6
    )

    height = _dist_2d(mouth_up, mouth_dn) / face_height
    width = _dist_2d(mouth_l, mouth_r) / face_width
    roundness = float(np.clip(height / max(width, 1e-6), 0.0, 3.0))

    mouth_center = (mouth_l + mouth_r) * 0.5
    left_droop = float(mouth_l[1] - mouth_center[1]) / face_height
    right_droop = float(mouth_r[1] - mouth_center[1]) / face_height

    upper_protrude = float(upper_protrude_pt[2] - lower_protrude_pt[2]) / face_height

    return np.array(
        [height, width, roundness, left_droop, right_droop, upper_protrude],
        dtype=np.float32,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def extract_nmm_features(face_landmarks_468x3: np.ndarray) -> np.ndarray:
    """Extract 32-dim NMM feature vector from MediaPipe FaceMesh landmarks.

    Layout:
      [0:20]  Facial Action Units (20 dims)
      [20:23] Head pose — pitch, yaw, roll (3 dims)
      [23:26] Gaze direction proxy (3 dims)
      [26:32] Mouth shape descriptors (6 dims)

    Args:
        face_landmarks_468x3: Array of shape (468, 3) or (1404,) or ≥(478, 3)
            (with iris refinement landmarks). If absent / all-zeros, returns zeros.

    Returns:
        Array of shape (32,) dtype float32.
    """
    face = np.asarray(face_landmarks_468x3, dtype=np.float32)

    # Accept flat layout
    if face.ndim == 1:
        if face.shape[0] < 468 * 3:
            return np.zeros(NMM_FEATURE_DIM, dtype=np.float32)
        face = face[: 478 * 3].reshape(-1, 3)
    elif face.ndim == 2:
        if face.shape[0] < 468:
            return np.zeros(NMM_FEATURE_DIM, dtype=np.float32)

    # Guard: absent face (all zeros)
    if float(np.max(np.abs(face))) < 1e-9:
        return np.zeros(NMM_FEATURE_DIM, dtype=np.float32)

    aus = _compute_au_features(face)
    head_pose = _compute_head_pose(face)
    gaze = _compute_gaze(face)
    mouth_shape = _compute_mouth_shape(face)

    features = np.concatenate([aus, head_pose, gaze, mouth_shape]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

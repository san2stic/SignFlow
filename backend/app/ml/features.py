"""Landmark feature extraction and normalization helpers.

V1 pipeline: ENRICHED_FEATURE_DIM = 493 (backward compatible)
V2 pipeline: ENRICHED_FEATURE_DIM_V2 = 611 — adds handshape (84), NMM (32),
             signing space (18), minus 16 redundant V1 hand-shape features.

Use ``extract_features_v2()`` or ``normalize_landmarks_v2()`` for the new pipeline.
Pass ``version=2`` to ``extract_features()`` for router-level dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from app.ml.feature_engineering import (
    ENRICHED_FEATURE_DIM,
    FACIAL_EXPRESSION_FEATURE_DIM,
    HAND_SHAPE_FEATURE_DIM,
)
from app.ml.handshape_features import (
    HANDSHAPE_FEATURE_DIM,
    extract_both_hands_handshape,
)
from app.ml.facial_action_units import NMM_FEATURE_DIM, extract_nmm_features
from app.ml.signing_space import SIGNING_SPACE_FEATURE_DIM, extract_signing_space

logger = structlog.get_logger(__name__)

# MediaPipe FaceMesh indices for compact expression and mouth descriptors.
_FACE_IDX_MOUTH_LEFT = 61
_FACE_IDX_MOUTH_RIGHT = 291
_FACE_IDX_UPPER_LIP = 13
_FACE_IDX_LOWER_LIP = 14
_FACE_IDX_LEFT_EYE_UPPER = 159
_FACE_IDX_LEFT_EYE_LOWER = 145
_FACE_IDX_RIGHT_EYE_UPPER = 386
_FACE_IDX_RIGHT_EYE_LOWER = 374
_FACE_IDX_LEFT_BROW = 70
_FACE_IDX_RIGHT_BROW = 300

FACIAL_FEATURE_MOUTH_OPEN = 0
FACIAL_FEATURE_MOUTH_WIDTH = 1
FACIAL_FEATURE_MOUTH_ROUNDNESS = 2
FACIAL_FEATURE_SMILE = 3
FACIAL_FEATURE_MOUTH_DETECTION_SCORE = 4
FACIAL_FEATURE_MOUTH_OPEN_FLAG = 5
FACIAL_FEATURE_LEFT_EYE_OPEN = 6
FACIAL_FEATURE_RIGHT_EYE_OPEN = 7
FACIAL_FEATURE_LEFT_BROW_RAISE = 8
FACIAL_FEATURE_RIGHT_BROW_RAISE = 9
FACIAL_FEATURE_BROW_ASYMMETRY = 10
FACIAL_FEATURE_EXPRESSION_INTENSITY = 11


@dataclass
class FrameLandmarks:
    """Landmarks payload for one frame."""

    left_hand: list[list[float]]
    right_hand: list[list[float]]
    pose: list[list[float]]
    face: list[list[float]] | None = None


def _flatten(points: list[list[float]], expected_len: int) -> list[float]:
    """Flatten xyz list into fixed-size vector with zero-padding."""
    values = [coord for point in points for coord in point[:3]]
    target_size = expected_len * 3
    if len(values) < target_size:
        values.extend([0.0] * (target_size - len(values)))
    return values[:target_size]


def _safe_norm(vector: np.ndarray) -> float:
    """Return stable vector norm."""
    return float(np.linalg.norm(vector)) if vector.size else 0.0


def _resolve_body_scale(pose_xyz: np.ndarray) -> float:
    """
    Estimate body scale to normalize signer size and camera distance.

    Uses shoulder and hip distances as robust anchors.
    """
    if pose_xyz.shape[0] < 25:
        return 1.0

    shoulder_left = pose_xyz[11]
    shoulder_right = pose_xyz[12]
    hip_left = pose_xyz[23]
    hip_right = pose_xyz[24]
    shoulder_width = _safe_norm(shoulder_left - shoulder_right)
    hip_width = _safe_norm(hip_left - hip_right)
    torso_height = _safe_norm((shoulder_left + shoulder_right) / 2.0 - (hip_left + hip_right) / 2.0)

    # Keep values stable even when detections are partial/noisy.
    scale = max(shoulder_width, hip_width, torso_height, 1e-3)
    return scale


def _face_point(face_xyz: np.ndarray, index: int) -> np.ndarray:
    """Safely access one face landmark point as xyz vector."""
    if index < 0 or index >= face_xyz.shape[0]:
        return np.zeros(3, dtype=np.float32)
    return face_xyz[index]


def _compute_facial_expression_features(
    face_points: list[list[float]] | None,
    *,
    hip_center: np.ndarray,
    body_scale: float,
) -> np.ndarray:
    """
    Build compact facial descriptors for expression and mouth detection.

    Output layout (12 dims):
      0 mouth_open
      1 mouth_width
      2 mouth_roundness
      3 smile
      4 mouth_detection_score
      5 mouth_open_flag
      6 left_eye_open
      7 right_eye_open
      8 left_brow_raise
      9 right_brow_raise
      10 brow_asymmetry
      11 expression_intensity
    """
    features = np.zeros(FACIAL_EXPRESSION_FEATURE_DIM, dtype=np.float32)
    if not face_points:
        return features

    face_raw = np.array(_flatten(face_points, 468), dtype=np.float32).reshape(-1, 3)
    if face_raw.shape[0] == 0:
        return features

    normalized_face = (face_raw - hip_center.reshape(1, 3)) / max(float(body_scale), 1e-3)

    mouth_upper = _face_point(normalized_face, _FACE_IDX_UPPER_LIP)
    mouth_lower = _face_point(normalized_face, _FACE_IDX_LOWER_LIP)
    mouth_left = _face_point(normalized_face, _FACE_IDX_MOUTH_LEFT)
    mouth_right = _face_point(normalized_face, _FACE_IDX_MOUTH_RIGHT)

    left_eye_upper = _face_point(normalized_face, _FACE_IDX_LEFT_EYE_UPPER)
    left_eye_lower = _face_point(normalized_face, _FACE_IDX_LEFT_EYE_LOWER)
    right_eye_upper = _face_point(normalized_face, _FACE_IDX_RIGHT_EYE_UPPER)
    right_eye_lower = _face_point(normalized_face, _FACE_IDX_RIGHT_EYE_LOWER)

    left_brow = _face_point(normalized_face, _FACE_IDX_LEFT_BROW)
    right_brow = _face_point(normalized_face, _FACE_IDX_RIGHT_BROW)

    mouth_open = _safe_norm(mouth_upper - mouth_lower)
    mouth_width = _safe_norm(mouth_left - mouth_right)
    mouth_roundness = mouth_open / max(mouth_width, 1e-6)

    mouth_center_y = 0.5 * (mouth_upper[1] + mouth_lower[1])
    smile_left = mouth_center_y - mouth_left[1]
    smile_right = mouth_center_y - mouth_right[1]
    smile = 0.5 * (smile_left + smile_right)

    left_eye_open = _safe_norm(left_eye_upper - left_eye_lower)
    right_eye_open = _safe_norm(right_eye_upper - right_eye_lower)

    left_brow_raise = max(0.0, float(left_eye_upper[1] - left_brow[1]))
    right_brow_raise = max(0.0, float(right_eye_upper[1] - right_brow[1]))
    brow_asymmetry = abs(left_brow_raise - right_brow_raise)

    tracked_points = [
        _FACE_IDX_MOUTH_LEFT,
        _FACE_IDX_MOUTH_RIGHT,
        _FACE_IDX_UPPER_LIP,
        _FACE_IDX_LOWER_LIP,
        _FACE_IDX_LEFT_EYE_UPPER,
        _FACE_IDX_LEFT_EYE_LOWER,
        _FACE_IDX_RIGHT_EYE_UPPER,
        _FACE_IDX_RIGHT_EYE_LOWER,
        _FACE_IDX_LEFT_BROW,
        _FACE_IDX_RIGHT_BROW,
    ]
    detected_count = sum(
        1
        for point_index in tracked_points
        if _safe_norm(_face_point(face_raw, point_index)) > 1e-6
    )
    mouth_detection_score = detected_count / len(tracked_points)
    mouth_open_flag = 1.0 if mouth_detection_score >= 0.5 and mouth_open > 0.012 else 0.0

    expression_intensity = float(
        np.mean(
            [
                mouth_open,
                mouth_roundness,
                abs(smile),
                left_eye_open,
                right_eye_open,
                left_brow_raise,
                right_brow_raise,
            ]
        )
    )

    features[FACIAL_FEATURE_MOUTH_OPEN] = float(mouth_open)
    features[FACIAL_FEATURE_MOUTH_WIDTH] = float(mouth_width)
    features[FACIAL_FEATURE_MOUTH_ROUNDNESS] = float(mouth_roundness)
    features[FACIAL_FEATURE_SMILE] = float(smile)
    features[FACIAL_FEATURE_MOUTH_DETECTION_SCORE] = float(mouth_detection_score)
    features[FACIAL_FEATURE_MOUTH_OPEN_FLAG] = float(mouth_open_flag)
    features[FACIAL_FEATURE_LEFT_EYE_OPEN] = float(left_eye_open)
    features[FACIAL_FEATURE_RIGHT_EYE_OPEN] = float(right_eye_open)
    features[FACIAL_FEATURE_LEFT_BROW_RAISE] = float(left_brow_raise)
    features[FACIAL_FEATURE_RIGHT_BROW_RAISE] = float(right_brow_raise)
    features[FACIAL_FEATURE_BROW_ASYMMETRY] = float(brow_asymmetry)
    features[FACIAL_FEATURE_EXPRESSION_INTENSITY] = float(expression_intensity)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_landmarks(
    frame: FrameLandmarks,
    include_face: bool = False,
    include_face_expressions: bool = True,
) -> np.ndarray:
    """Normalize landmarks and return feature vector for one frame."""
    pose = _flatten(frame.pose, 33)
    hip_left = np.array(pose[23 * 3 : 23 * 3 + 3]) if len(pose) >= 75 else np.zeros(3)
    hip_right = np.array(pose[24 * 3 : 24 * 3 + 3]) if len(pose) >= 78 else np.zeros(3)
    hip_center = (hip_left + hip_right) / 2.0

    left = np.array(_flatten(frame.left_hand, 21)).reshape(-1, 3)
    right = np.array(_flatten(frame.right_hand, 21)).reshape(-1, 3)
    pose_vec = np.array(pose).reshape(-1, 3)
    body_scale = _resolve_body_scale(pose_vec)

    left = (left - hip_center) / body_scale
    right = (right - hip_center) / body_scale
    pose_vec = (pose_vec - hip_center) / body_scale

    chunks = [left.reshape(-1), right.reshape(-1), pose_vec.reshape(-1)]
    if include_face_expressions:
        chunks.append(
            _compute_facial_expression_features(
                frame.face,
                hip_center=hip_center,
                body_scale=body_scale,
            )
        )
    if include_face:
        face = np.array(_flatten(frame.face or [], 468)).reshape(-1, 3)
        chunks.append(((face - hip_center) / body_scale).reshape(-1))

    normalized = np.concatenate(chunks).astype(np.float32)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(normalized, -5.0, 5.0)


@dataclass
class LandmarkExtractionResult:
    """Result of landmark extraction from a video."""

    landmarks: np.ndarray  # Shape: [num_frames, num_features]
    fps: float
    num_frames: int
    detection_rate: float  # Ratio of frames where landmarks were detected
    duration_sec: float


def extract_landmarks_from_video(
    video_path: str | Path,
    include_face: bool = False,
    include_face_expressions: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> LandmarkExtractionResult:
    """
    Extract MediaPipe Holistic landmarks from a video file.

    Args:
        video_path: Path to the video file
        include_face: Whether to include face landmarks (468 points)
        include_face_expressions: Whether to append compact facial-expression features
        min_detection_confidence: Minimum confidence for detection
        min_tracking_confidence: Minimum confidence for tracking

    Returns:
        LandmarkExtractionResult with landmarks array and metadata

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or has no frames
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info("extracting_landmarks", video_path=str(video_path))

    try:
        import cv2
        import mediapipe as mp
    except ModuleNotFoundError as exc:
        raise RuntimeError("MediaPipe/OpenCV dependencies are required for landmark extraction") from exc

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,  # Balance between speed and accuracy
        smooth_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    logger.debug(
        "video_metadata",
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
    )

    landmarks_list = []
    frames_with_detection = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe
            results = holistic.process(frame_rgb)

            # Extract landmarks
            left_hand = []
            right_hand = []
            pose = []
            face = []

            # Left hand landmarks (21 points)
            if results.left_hand_landmarks:
                left_hand = [
                    [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
                ]
            else:
                left_hand = [[0.0, 0.0, 0.0]] * 21

            # Right hand landmarks (21 points)
            if results.right_hand_landmarks:
                right_hand = [
                    [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
                ]
            else:
                right_hand = [[0.0, 0.0, 0.0]] * 21

            # Pose landmarks (33 points)
            if results.pose_landmarks:
                pose = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            else:
                pose = [[0.0, 0.0, 0.0]] * 33

            # Face landmarks are needed for full face export or compact expression features.
            if include_face or include_face_expressions:
                if results.face_landmarks:
                    face = [
                        [lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark
                    ]
                else:
                    face = [[0.0, 0.0, 0.0]] * 468

            # Check if we have at least some detection
            has_detection = (
                results.left_hand_landmarks is not None
                or results.right_hand_landmarks is not None
                or results.pose_landmarks is not None
            )

            if has_detection:
                frames_with_detection += 1

            # Create FrameLandmarks and normalize
            frame_landmarks = FrameLandmarks(
                left_hand=left_hand,
                right_hand=right_hand,
                pose=pose,
                face=face if (include_face or include_face_expressions) else None,
            )

            normalized = normalize_landmarks(
                frame_landmarks,
                include_face=include_face,
                include_face_expressions=include_face_expressions,
            )
            landmarks_list.append(normalized)

    finally:
        cap.release()
        holistic.close()

    if frame_count == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # Convert to numpy array
    landmarks_array = np.stack(landmarks_list, axis=0)  # Shape: [num_frames, num_features]

    detection_rate = frames_with_detection / frame_count if frame_count > 0 else 0.0

    logger.info(
        "extraction_complete",
        num_frames=frame_count,
        detection_rate=detection_rate,
        shape=landmarks_array.shape,
    )

    if detection_rate < 0.8:
        logger.warning(
            "low_detection_rate",
            detection_rate=detection_rate,
            video_path=str(video_path),
            msg="Less than 80% of frames have landmarks detected. Video quality may be poor.",
        )

    return LandmarkExtractionResult(
        landmarks=landmarks_array,
        fps=fps,
        num_frames=frame_count,
        detection_rate=detection_rate,
        duration_sec=duration_sec,
    )


# ── V2 feature pipeline ───────────────────────────────────────────────────────

# V2 layout (explicit, matches normalize_landmarks_v2 output):
#   coord_block     (left_hand xyz + right_hand xyz + pose xyz) = 63+63+99 = 225
#   facial_expr     compact expression features                  =  12
#   coord_vel       frame velocities of coordinate block         = 225
#   inter_dist      5 inter-hand distances                       =   5
#   joint_angles    4 arm joint angles                           =   4
#   facial_vel      6 facial velocity dims (reduced from 12)     =   6
#   ── NEW ──
#   handshape       2 × 42 per-hand handshape features           =  84
#   NMM             facial action units + head pose + gaze        =  32
#   signing_space   spatial signing space features               =  18
# ─────────────────────────────────────────────────────────────────────────────
#   TOTAL                                                        = 611

_V2_BASE_DIM = (
    225          # coord_block (left 63 + right 63 + pose 99)
    + 12         # facial_expr
    + 225        # coord_vel
    + 5          # inter_dist
    + 4          # joint_angles
    + 6          # facial_vel (6 channels)
)  # = 477

ENRICHED_FEATURE_DIM_V2: int = (
    _V2_BASE_DIM                        # 477 — V2 base (no hand_shape, partial facial_vel)
    + 2 * HANDSHAPE_FEATURE_DIM         # +84 — rich per-hand handshape (2 × 42)
    + NMM_FEATURE_DIM                   # +32 — NMM / facial AUs
    + SIGNING_SPACE_FEATURE_DIM         # +18 — signing space
)
# = 477 + 84 + 32 + 18 = 611


def normalize_landmarks_v2(
    frame: FrameLandmarks,
    *,
    include_handshape: bool = True,
    include_nmm: bool = True,
    include_signing_space: bool = True,
    prev_right_wrist: np.ndarray | None = None,
    prev_left_wrist: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize landmarks and return V2 feature vector for one frame.

    V2 layout (611 dims by default):
      [0:63]    left-hand xyz (normalized)
      [63:126]  right-hand xyz (normalized)
      [126:225] pose xyz (normalized)
      [225:237] compact facial expression (12 dims)
      [237:462] coordinate velocities (225 dims)
      [462:467] inter-hand distances (5 dims)
      [467:471] joint angles (4 dims)
      — V1 hand_shape removed (10 dims) —
      [471:477] facial expression velocity (6 dims — first 6 of 12 velocity channels)
      [477:561] handshape features 84 dims (if include_handshape)
      [561:593] NMM features 32 dims (if include_nmm)
      [593:611] signing space features 18 dims (if include_signing_space)

    Args:
        frame: FrameLandmarks with left_hand, right_hand, pose, face.
        include_handshape: Whether to append handshape features (84 dims).
        include_nmm: Whether to append NMM/AU features (32 dims). Requires face data.
        include_signing_space: Whether to append signing space features (18 dims).
        prev_right_wrist: Previous right wrist for motion (optional).
        prev_left_wrist: Previous left wrist for motion (optional).

    Returns:
        Array of shape (611,) dtype float32 when all modules enabled.
    """
    # ── V1 base (stripped of hand_shape and reduced facial_velocity) ── #
    pose_raw = _flatten(frame.pose, 33)
    hip_left = np.array(pose_raw[23 * 3 : 23 * 3 + 3]) if len(pose_raw) >= 75 else np.zeros(3)
    hip_right = np.array(pose_raw[24 * 3 : 24 * 3 + 3]) if len(pose_raw) >= 78 else np.zeros(3)
    hip_center = (hip_left + hip_right) / 2.0

    left_np = np.array(_flatten(frame.left_hand, 21), dtype=np.float32).reshape(-1, 3)
    right_np = np.array(_flatten(frame.right_hand, 21), dtype=np.float32).reshape(-1, 3)
    pose_vec = np.array(pose_raw, dtype=np.float32).reshape(-1, 3)
    body_scale = _resolve_body_scale(pose_vec)

    left_norm = ((left_np - hip_center) / body_scale).reshape(-1)
    right_norm = ((right_np - hip_center) / body_scale).reshape(-1)
    pose_norm = ((pose_vec - hip_center) / body_scale).reshape(-1)

    facial_expr = _compute_facial_expression_features(
        frame.face, hip_center=hip_center, body_scale=body_scale
    )  # (12,)

    # Velocities (225 dims using coordinate block only)
    coord_block = np.concatenate([left_norm, right_norm, pose_norm])  # (225,)
    # We approximate per-frame velocity as zeros here; multi-frame is handled by
    # compute_enriched_features in feature_engineering.py for training sequences.
    coord_vel = np.zeros_like(coord_block)  # (225,) for single-frame inference

    # Distances (5 dims)
    l_wrist_pt = left_np[0] if left_np.shape[0] > 0 else np.zeros(3)
    r_wrist_pt = right_np[0] if right_np.shape[0] > 0 else np.zeros(3)
    l_tips = left_np[[4, 8, 12, 20]] if left_np.shape[0] >= 21 else np.zeros((4, 3))
    r_tips = right_np[[4, 8, 12, 20]] if right_np.shape[0] >= 21 else np.zeros((4, 3))
    inter_dist = np.zeros(5, dtype=np.float32)
    inter_dist[0] = float(np.linalg.norm(l_wrist_pt - r_wrist_pt)) / max(body_scale, 1e-6)
    for i in range(4):
        inter_dist[i + 1] = float(np.linalg.norm(l_tips[i] - r_tips[i])) / max(body_scale, 1e-6)

    # Joint angles (4 dims) — elbow + wrist
    def _angle_vec(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba, bc = a - b, c - b
        n_ba, n_bc = float(np.linalg.norm(ba)), float(np.linalg.norm(bc))
        if n_ba < 1e-8 or n_bc < 1e-8:
            return 0.0
        return float(np.arccos(np.clip(float(np.dot(ba, bc) / (n_ba * n_bc)), -1.0, 1.0)))

    jnt = np.zeros(4, dtype=np.float32)
    if pose_vec.shape[0] >= 17:
        jnt[0] = _angle_vec(pose_vec[11], pose_vec[13], pose_vec[15])  # L elbow
        jnt[1] = _angle_vec(pose_vec[12], pose_vec[14], pose_vec[16])  # R elbow
    if pose_vec.shape[0] >= 17 and left_np.shape[0] >= 13:
        jnt[2] = _angle_vec(pose_vec[13], left_np[0], left_np[12])   # L wrist
    if pose_vec.shape[0] >= 17 and right_np.shape[0] >= 13:
        jnt[3] = _angle_vec(pose_vec[14], right_np[0], right_np[12]) # R wrist

    # Facial velocity (first 6 channels only, reduced from 12 in V1)
    facial_vel = np.zeros(6, dtype=np.float32)  # per-frame = 0; meaningful over time

    # Base V2 vector (471 dims)
    v2_base = np.concatenate([
        coord_block,   # 225
        facial_expr,   # 12
        coord_vel,     # 225
        inter_dist,    # 5
        jnt,           # 4
        facial_vel,    # 6
    ])  # = 477

    chunks = [v2_base]

    # ── Handshape (84 dims) ──────────────────────────────────────────────────
    if include_handshape:
        # Use normalized (body-centered) hand landmarks for handshape
        left_hs = left_norm.reshape(21, 3) if left_norm.shape[0] >= 63 else np.zeros((21, 3))
        right_hs = right_norm.reshape(21, 3) if right_norm.shape[0] >= 63 else np.zeros((21, 3))
        hs_feats = extract_both_hands_handshape(left_hs, right_hs)  # (84,)
        chunks.append(hs_feats)

    # ── NMM (32 dims) ─────────────────────────────────────────────────────────
    if include_nmm:
        face_pts = frame.face or []
        if face_pts:
            face_arr = np.array(_flatten(face_pts, 468), dtype=np.float32).reshape(-1, 3)
        else:
            face_arr = np.zeros((468, 3), dtype=np.float32)
        nmm_feats = extract_nmm_features(face_arr)  # (32,)
        chunks.append(nmm_feats)

    # ── Signing space (18 dims) ───────────────────────────────────────────────
    if include_signing_space:
        left_raw = left_np  # raw (before normalization) for signed-space computation
        right_raw = right_np
        space_feats = extract_signing_space(
            left_hand=left_raw,
            right_hand=right_raw,
            pose_landmarks=pose_vec,
            prev_right_wrist=prev_right_wrist,
            prev_left_wrist=prev_left_wrist,
        )  # (18,)
        chunks.append(space_feats)

    result = np.concatenate(chunks).astype(np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -5.0, 5.0)


def extract_features_v2(
    landmarks_data: dict,
    *,
    include_handshape: bool = True,
    include_nmm: bool = True,
    include_signing_space: bool = True,
    prev_right_wrist: np.ndarray | None = None,
    prev_left_wrist: np.ndarray | None = None,
) -> np.ndarray:
    """Extract V2 feature vector from a landmarks payload dict.

    Convenience wrapper for use in the inference pipeline; accepts
    the same dict format as ``process_frame()`` in the pipeline.

    Args:
        landmarks_data: Dict with keys ``hands`` (left/right), ``pose``, ``face``.
        include_handshape: Enable rich handshape features (84 dims).
        include_nmm: Enable NMM / facial AU features (32 dims, requires face).
        include_signing_space: Enable spatial signing features (18 dims).
        prev_right_wrist: Previous right wrist xyz for motion estimation.
        prev_left_wrist: Previous left wrist xyz for motion estimation.

    Returns:
        Feature vector of shape (ENRICHED_FEATURE_DIM_V2,) = (611,) when all enabled.
    """
    frame = FrameLandmarks(
        left_hand=landmarks_data.get("hands", {}).get("left", []) or [],
        right_hand=landmarks_data.get("hands", {}).get("right", []) or [],
        pose=landmarks_data.get("pose", []) or [],
        face=landmarks_data.get("face", []) or [],
    )
    return normalize_landmarks_v2(
        frame,
        include_handshape=include_handshape,
        include_nmm=include_nmm,
        include_signing_space=include_signing_space,
        prev_right_wrist=prev_right_wrist,
        prev_left_wrist=prev_left_wrist,
    )

"""Landmark feature extraction and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from app.ml.feature_engineering import FACIAL_EXPRESSION_FEATURE_DIM

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

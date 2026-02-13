"""Landmark feature extraction and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


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


def normalize_landmarks(frame: FrameLandmarks, include_face: bool = False) -> np.ndarray:
    """Normalize coordinates relative to hip center and return feature vector."""
    pose = _flatten(frame.pose, 33)
    hip_left = np.array(pose[23 * 3 : 23 * 3 + 3]) if len(pose) >= 75 else np.zeros(3)
    hip_right = np.array(pose[24 * 3 : 24 * 3 + 3]) if len(pose) >= 78 else np.zeros(3)
    hip_center = (hip_left + hip_right) / 2.0

    left = np.array(_flatten(frame.left_hand, 21)).reshape(-1, 3)
    right = np.array(_flatten(frame.right_hand, 21)).reshape(-1, 3)
    pose_vec = np.array(pose).reshape(-1, 3)

    left = left - hip_center
    right = right - hip_center
    pose_vec = pose_vec - hip_center

    chunks = [left.reshape(-1), right.reshape(-1), pose_vec.reshape(-1)]
    if include_face:
        face = np.array(_flatten(frame.face or [], 468)).reshape(-1, 3)
        chunks.append((face - hip_center).reshape(-1))

    return np.concatenate(chunks).astype(np.float32)


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
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> LandmarkExtractionResult:
    """
    Extract MediaPipe Holistic landmarks from a video file.

    Args:
        video_path: Path to the video file
        include_face: Whether to include face landmarks (468 points)
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

            # Face landmarks (468 points) - optional
            if include_face:
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
                face=face if include_face else None,
            )

            normalized = normalize_landmarks(frame_landmarks, include_face=include_face)
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

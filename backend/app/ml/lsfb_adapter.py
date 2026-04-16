"""Adapter to convert LSFB-ISOL landmark files into SignFlow's 225-dim format.

LSFB-ISOL stores MediaPipe landmarks in **four separate** ``.npy`` files per
clip instance (one per body part):

  poses/left_hand/{instance_id}.npy   → shape [frames, 21, 3]
  poses/right_hand/{instance_id}.npy  → shape [frames, 21, 3]
  poses/pose/{instance_id}.npy        → shape [frames, 30, 3]
  poses/face/{instance_id}.npy        → shape [frames, 468, 3]  (optional)

SignFlow expects a **single** flattened array per clip:

  [frames, 225]  where  left_hand(63) + right_hand(63) + pose(99)

Pose mismatch:
  LSFB uses **30** pose landmarks, SignFlow/MediaPipe uses **33**.
  The three missing landmarks (indices 30-32) are zero-padded.
  This is safe because the feature engineering only accesses indices 0-24.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# SignFlow constants
_SIGNFLOW_LEFT_HAND_DIM = 63   # 21 landmarks * 3
_SIGNFLOW_RIGHT_HAND_DIM = 63  # 21 landmarks * 3
_SIGNFLOW_POSE_DIM = 99        # 33 landmarks * 3
_SIGNFLOW_TOTAL_DIM = _SIGNFLOW_LEFT_HAND_DIM + _SIGNFLOW_RIGHT_HAND_DIM + _SIGNFLOW_POSE_DIM  # 225

# LSFB pose has 30 landmarks, SignFlow expects 33 → pad 3 extra landmarks
_LSFB_POSE_LANDMARKS = 30
_SIGNFLOW_POSE_LANDMARKS = 33
_POSE_PAD = (_SIGNFLOW_POSE_LANDMARKS - _LSFB_POSE_LANDMARKS) * 3  # 9 values


@dataclass
class ConversionStats:
    """Statistics from a batch conversion run."""

    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[dict] = field(default_factory=list)
    per_sign_counts: dict[str, int] = field(default_factory=dict)


def convert_lsfb_instance(
    instance_id: str,
    lsfb_poses_dir: Path,
) -> np.ndarray:
    """Convert one LSFB-ISOL instance from separate body-part files to a 225-dim array.

    Args:
        instance_id: Clip identifier (e.g. ``CLSFBI0103A_S001_B_286114_286297``).
        lsfb_poses_dir: Path to the ``poses/`` directory containing subdirectories
            ``left_hand/``, ``right_hand/``, ``pose/``.

    Returns:
        ndarray of shape ``[num_frames, 225]``.

    Raises:
        FileNotFoundError: If the required pose body-part file is missing.
        ValueError: If body-part arrays have incompatible frame counts.
    """
    left_path = lsfb_poses_dir / "left_hand" / f"{instance_id}.npy"
    right_path = lsfb_poses_dir / "right_hand" / f"{instance_id}.npy"
    pose_path = lsfb_poses_dir / "pose" / f"{instance_id}.npy"

    # Pose (body) is required; hands are optional (zero-filled when absent).
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing landmark file: {pose_path}")

    pose = np.load(pose_path)
    num_frames = pose.shape[0]
    if num_frames == 0:
        raise ValueError(f"Instance {instance_id} has 0 frames")

    # Load hand arrays if available, otherwise zero-fill.
    if left_path.exists():
        left = np.load(left_path)[:num_frames]
        if left.shape[0] < num_frames:
            pad = np.zeros((num_frames - left.shape[0], *left.shape[1:]), dtype=left.dtype)
            left = np.concatenate([left, pad])
    else:
        left = np.zeros((num_frames, 21, 3), dtype=np.float32)

    if right_path.exists():
        right = np.load(right_path)[:num_frames]
        if right.shape[0] < num_frames:
            pad = np.zeros((num_frames - right.shape[0], *right.shape[1:]), dtype=right.dtype)
            right = np.concatenate([right, pad])
    else:
        right = np.zeros((num_frames, 21, 3), dtype=np.float32)

    pose = pose[:num_frames]

    # Flatten from [frames, N_landmarks, 3] → [frames, N_landmarks*3]
    left_flat = left.reshape(num_frames, -1)    # [frames, 63]
    right_flat = right.reshape(num_frames, -1)  # [frames, 63]
    pose_flat = pose.reshape(num_frames, -1)    # [frames, 90]

    # Pad pose from 30 landmarks (90 values) → 33 landmarks (99 values)
    pose_padded = np.zeros((num_frames, _SIGNFLOW_POSE_DIM), dtype=pose_flat.dtype)
    pose_padded[:, : pose_flat.shape[1]] = pose_flat

    # Concatenate: [frames, 63 + 63 + 99] = [frames, 225]
    result = np.concatenate([left_flat, right_flat, pose_padded], axis=1)
    assert result.shape[1] == _SIGNFLOW_TOTAL_DIM, f"Expected {_SIGNFLOW_TOTAL_DIM}, got {result.shape[1]}"

    # Replace NaN / inf with 0
    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return result.astype(np.float32)


def compute_detection_rate(landmarks: np.ndarray) -> float:
    """Compute the ratio of frames with non-zero landmarks.

    A frame is considered "detected" if the sum of absolute coordinate values
    exceeds a small threshold (some frames are all zeros when MediaPipe
    failed to detect the signer).
    """
    frame_norms = np.abs(landmarks).sum(axis=1)
    detected = (frame_norms > 1e-6).sum()
    return float(detected / max(landmarks.shape[0], 1))


def load_instances_csv(csv_path: Path) -> list[dict]:
    """Parse the LSFB-ISOL ``instances.csv`` into a list of dicts.

    Expected columns: ``id, sign, signer, start, end``
    """
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def load_split_json(split_path: Path) -> set[str]:
    """Load a split JSON file and return the set of instance IDs it contains."""
    data = json.loads(split_path.read_text(encoding="utf-8"))
    # The LSFB split files can be a flat list of instance IDs or a dict of lists.
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        ids: set[str] = set()
        for v in data.values():
            if isinstance(v, list):
                ids.update(v)
        return ids
    return set()


def convert_all_instances(
    lsfb_dir: Path,
    output_dir: Path,
    *,
    instances_csv: Path | None = None,
    split_ids: set[str] | None = None,
    skip_existing: bool = True,
) -> ConversionStats:
    """Convert all LSFB-ISOL instances to SignFlow 225-dim format.

    Args:
        lsfb_dir: Root of the downloaded LSFB-ISOL dataset.
        output_dir: Where to save the converted ``{instance_id}_landmarks.npy`` files.
        instances_csv: Path to ``instances.csv``.  Auto-detected if None.
        split_ids: Optional set of instance IDs to limit conversion to.
        skip_existing: Skip instances whose output file already exists.

    Returns:
        ConversionStats summary.
    """
    if instances_csv is None:
        instances_csv = lsfb_dir / "instances.csv"

    poses_dir = lsfb_dir / "poses"
    if not poses_dir.is_dir():
        raise FileNotFoundError(f"poses/ directory not found in {lsfb_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_instances_csv(instances_csv)
    stats = ConversionStats()

    for row in rows:
        instance_id = row.get("id", "").strip()
        sign_gloss = row.get("sign", "").strip()

        if not instance_id:
            continue

        if split_ids is not None and instance_id not in split_ids:
            continue

        stats.total += 1

        out_path = output_dir / f"{instance_id}_landmarks.npy"
        if skip_existing and out_path.exists():
            stats.skipped += 1
            stats.per_sign_counts[sign_gloss] = stats.per_sign_counts.get(sign_gloss, 0) + 1
            continue

        try:
            converted = convert_lsfb_instance(instance_id, poses_dir)
            np.save(out_path, converted)
            stats.success += 1
            stats.per_sign_counts[sign_gloss] = stats.per_sign_counts.get(sign_gloss, 0) + 1
        except Exception as exc:
            stats.failed += 1
            stats.errors.append({"instance_id": instance_id, "error": str(exc)})
            if stats.failed <= 20:
                logger.warning("lsfb_conversion_failed", instance_id=instance_id, error=str(exc))

    return stats

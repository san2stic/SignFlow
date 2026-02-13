"""Dataset abstraction for sign landmark sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
import torch
from torch.utils.data import Dataset

logger = structlog.get_logger(__name__)


@dataclass
class SignSample:
    """One training sample composed of sequence and class label."""

    landmarks: np.ndarray
    label: int


class LandmarkDataset(Dataset):
    """PyTorch Dataset for sign language landmark sequences with sliding window."""

    def __init__(
        self,
        samples: list[SignSample],
        sequence_length: int = 30,
        stride: int = 10,
        apply_sliding_window: bool = True,
    ) -> None:
        """
        Initialize dataset with optional sliding window processing.

        Args:
            samples: List of SignSample with landmarks and labels
            sequence_length: Number of frames per sequence (default: 30 frames = 1 sec @ 30fps)
            stride: Stride for sliding window (default: 10 frames)
            apply_sliding_window: If True, apply sliding window; if False, use full sequences
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.apply_sliding_window = apply_sliding_window

        # Process samples with sliding window
        self.processed_samples: list[tuple[torch.Tensor, int]] = []

        for sample in samples:
            if apply_sliding_window:
                # Apply sliding window to create multiple sequences from one sample
                sequences = self._create_sliding_windows(sample.landmarks, sample.label)
                self.processed_samples.extend(sequences)
            else:
                # Use full sequence with padding if needed
                padded = self._pad_or_truncate(sample.landmarks)
                tensor = torch.from_numpy(padded).float()
                self.processed_samples.append((tensor, sample.label))

        logger.debug(
            "dataset_initialized",
            num_samples=len(samples),
            num_sequences=len(self.processed_samples),
            sequence_length=sequence_length,
            stride=stride,
        )

    def _create_sliding_windows(
        self, landmarks: np.ndarray, label: int
    ) -> list[tuple[torch.Tensor, int]]:
        """
        Create multiple sequences from landmarks using sliding window.

        Args:
            landmarks: Array of shape [num_frames, num_features]
            label: Class label

        Returns:
            List of (tensor, label) tuples
        """
        num_frames = landmarks.shape[0]
        sequences = []

        # If video is shorter than sequence_length, pad and return single sequence
        if num_frames < self.sequence_length:
            padded = self._pad_or_truncate(landmarks)
            tensor = torch.from_numpy(padded).float()
            sequences.append((tensor, label))
            return sequences

        # Apply sliding window
        for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
            end_idx = start_idx + self.sequence_length
            window = landmarks[start_idx:end_idx]

            tensor = torch.from_numpy(window).float()
            sequences.append((tensor, label))

        return sequences

    def _pad_or_truncate(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Pad or truncate landmarks to fixed sequence length.

        Args:
            landmarks: Array of shape [num_frames, num_features]

        Returns:
            Array of shape [sequence_length, num_features]
        """
        num_frames = landmarks.shape[0]
        num_features = landmarks.shape[1]

        if num_frames == self.sequence_length:
            return landmarks

        elif num_frames < self.sequence_length:
            # Pad with zeros at the end
            padding = np.zeros((self.sequence_length - num_frames, num_features), dtype=landmarks.dtype)
            return np.vstack([landmarks, padding])

        else:
            # Truncate to sequence_length
            return landmarks[: self.sequence_length]

    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.processed_samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Get one sequence by index.

        Returns:
            Tuple of (landmarks_tensor, label)
            landmarks_tensor shape: [sequence_length, num_features]
        """
        return self.processed_samples[index]


def load_landmarks_from_file(landmarks_path: str | Path) -> np.ndarray:
    """
    Load landmarks from a .npy file.

    Args:
        landmarks_path: Path to .npy file

    Returns:
        Numpy array of landmarks with shape [num_frames, num_features]

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid numpy array
    """
    landmarks_path = Path(landmarks_path)

    if not landmarks_path.exists():
        raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")

    try:
        landmarks = np.load(str(landmarks_path))

        if landmarks.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {landmarks.shape}")

        logger.debug("landmarks_loaded", path=str(landmarks_path), shape=landmarks.shape)
        return landmarks

    except Exception as e:
        logger.error("failed_to_load_landmarks", path=str(landmarks_path), error=str(e))
        raise ValueError(f"Failed to load landmarks from {landmarks_path}: {e}") from e

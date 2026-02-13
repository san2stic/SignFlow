"""Video similarity computation using ML embeddings."""
from __future__ import annotations

import numpy as np
import structlog
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.dataset import temporal_resample
from app.ml.feature_engineering import compute_enriched_features

logger = structlog.get_logger(__name__)


def compute_video_similarity(
    landmarks_path_a: str | Path,
    landmarks_path_b: str | Path
) -> float:
    """
    Compute cosine similarity between two videos based on landmarks.

    Pipeline:
    1. Load landmarks (.npy) → [frames, 225]
    2. Temporal resampling → [64, 225]
    3. Feature engineering → [64, 469]
    4. Mean pooling → [469] (embedding vector)
    5. Cosine similarity between vectors

    Args:
        landmarks_path_a: Path to first video landmarks .npy file
        landmarks_path_b: Path to second video landmarks .npy file

    Returns:
        float: Similarity score 0.0-1.0 (1.0 = identical)
    """
    # Load landmarks
    landmarks_a = np.load(landmarks_path_a)
    landmarks_b = np.load(landmarks_path_b)

    # Temporal normalization
    seq_a = temporal_resample(landmarks_a, target_len=64)
    seq_b = temporal_resample(landmarks_b, target_len=64)

    # Feature enrichment
    features_a = compute_enriched_features(seq_a)  # [64, 469]
    features_b = compute_enriched_features(seq_b)

    # Mean pooling → embeddings
    embedding_a = features_a.mean(axis=0)  # [469]
    embedding_b = features_b.mean(axis=0)

    # Cosine similarity
    similarity = cosine_similarity(
        embedding_a.reshape(1, -1),
        embedding_b.reshape(1, -1)
    )[0, 0]

    return float(np.clip(similarity, 0.0, 1.0))


def find_similar_videos(
    target_video_path: str | Path,
    candidate_videos: list[Path],
    threshold: float = 0.75,
    top_k: int = 5
) -> list[tuple[Path, float]]:
    """
    Find the K most similar videos to a target.

    Args:
        target_video_path: Landmarks path of reference video
        candidate_videos: List of candidate landmarks paths
        threshold: Minimum similarity threshold (default: 0.75)
        top_k: Maximum number of suggestions (default: 5)

    Returns:
        List of (path, score) tuples sorted by descending similarity
    """
    results = []

    for candidate_path in candidate_videos:
        try:
            score = compute_video_similarity(target_video_path, candidate_path)
            if score >= threshold:
                results.append((candidate_path, score))
        except Exception as e:
            # Skip corrupted videos or invalid landmarks
            logger.warning(
                "similarity_computation_failed",
                candidate=str(candidate_path),
                error=str(e)
            )
            continue

    # Sort by descending score and limit to top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

"""Tests for video similarity computation."""
import numpy as np
import pytest
from pathlib import Path
from app.ml.similarity import compute_video_similarity, find_similar_videos


def test_identical_videos_have_perfect_similarity(tmp_path):
    """Identical videos should have similarity ~1.0"""
    landmarks = np.random.rand(60, 225).astype(np.float32)

    path_a = tmp_path / "video_a.npy"
    path_b = tmp_path / "video_b.npy"

    np.save(path_a, landmarks)
    np.save(path_b, landmarks)

    similarity = compute_video_similarity(path_a, path_b)

    assert similarity > 0.99, f"Expected >0.99, got {similarity}"
    assert similarity <= 1.0


def test_different_videos_have_low_similarity(tmp_path):
    """Very different videos should have low similarity"""
    # Create truly different movement patterns with different temporal dynamics
    np.random.seed(42)

    # Video A: gradual movement
    landmarks_a = np.linspace(0, 1, 60 * 225).reshape(60, 225).astype(np.float32)

    # Video B: rapid oscillation with different scale
    t = np.linspace(0, 4 * np.pi, 60)
    landmarks_b = np.zeros((60, 225), dtype=np.float32)
    for i in range(225):
        landmarks_b[:, i] = np.sin(t * (i + 1)) * 5

    path_a = tmp_path / "video_a.npy"
    path_b = tmp_path / "video_b.npy"

    np.save(path_a, landmarks_a)
    np.save(path_b, landmarks_b)

    similarity = compute_video_similarity(path_a, path_b)

    assert similarity < 0.5, f"Expected <0.5, got {similarity}"


def test_find_similar_videos_respects_threshold(tmp_path):
    """find_similar_videos should filter by threshold"""
    # Create target video
    target_landmarks = np.random.rand(60, 225).astype(np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, target_landmarks)

    # Create 3 candidates: identical, similar, different
    candidates = []

    # Identical (similarity ~1.0)
    identical_path = tmp_path / "identical.npy"
    np.save(identical_path, target_landmarks)
    candidates.append(identical_path)

    # Similar (similarity ~0.8)
    similar_landmarks = target_landmarks + np.random.rand(60, 225).astype(np.float32) * 0.1
    similar_path = tmp_path / "similar.npy"
    np.save(similar_path, similar_landmarks)
    candidates.append(similar_path)

    # Different (similarity <0.5)
    different_landmarks = np.random.rand(60, 225).astype(np.float32) * 5
    different_path = tmp_path / "different.npy"
    np.save(different_path, different_landmarks)
    candidates.append(different_path)

    # Find similar with threshold 0.75
    results = find_similar_videos(target_path, candidates, threshold=0.75, top_k=5)

    # Should return only identical and similar (not different)
    assert len(results) >= 1, "Should find at least identical video"
    assert all(score >= 0.75 for _, score in results), "All scores should be >= threshold"


def test_find_similar_videos_limits_top_k(tmp_path):
    """find_similar_videos should limit results to top_k"""
    target_landmarks = np.random.rand(60, 225).astype(np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, target_landmarks)

    # Create 10 similar candidates
    candidates = []
    for i in range(10):
        path = tmp_path / f"candidate_{i}.npy"
        np.save(path, target_landmarks + np.random.rand(60, 225).astype(np.float32) * 0.05)
        candidates.append(path)

    # Request only top 3
    results = find_similar_videos(target_path, candidates, threshold=0.7, top_k=3)

    assert len(results) <= 3, f"Expected max 3 results, got {len(results)}"

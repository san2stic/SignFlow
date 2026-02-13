"""Tests for videos API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import SessionLocal
from app.models.video import Video
from app.models.sign import Sign


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def setup_videos(db):
    """Create test videos: 3 labeled, 2 unlabeled"""
    # Create a sign for labeled videos
    sign = Sign(
        id="test-sign-id",
        name="test_sign",
        slug="test-sign",
        category="test"
    )
    db.add(sign)

    # Labeled videos
    for i in range(3):
        video = Video(
            id=f"labeled-video-{i}",
            sign_id="test-sign-id",
            file_path=f"/test/labeled_{i}.mp4",
            landmarks_extracted=True,
            landmarks_path=f"/test/labeled_{i}_landmarks.npy"
        )
        db.add(video)

    # Unlabeled videos
    for i in range(2):
        video = Video(
            id=f"unlabeled-video-{i}",
            sign_id=None,
            file_path=f"/test/unlabeled_{i}.mp4",
            landmarks_extracted=True,
            landmarks_path=f"/test/unlabeled_{i}_landmarks.npy"
        )
        db.add(video)

    db.commit()

    yield

    # Cleanup
    db.query(Video).delete()
    db.query(Sign).delete()
    db.commit()


def test_get_unlabeled_videos_returns_only_unlabeled(client, setup_videos):
    """GET /videos/unlabeled should return only videos without sign_id"""
    response = client.get("/api/v1/videos/unlabeled")

    assert response.status_code == 200
    data = response.json()

    assert "items" in data
    assert len(data["items"]) == 2

    # All returned videos should have sign_id = null
    for video in data["items"]:
        assert video["sign_id"] is None


def test_label_video_updates_sign_id(client, setup_videos, db):
    """PATCH /videos/{id}/label should update sign_id"""
    video_id = "unlabeled-video-0"
    sign_id = "test-sign-id"

    response = client.patch(
        f"/api/v1/videos/{video_id}/label",
        json={"sign_id": sign_id}
    )

    assert response.status_code == 200

    # Verify in database
    video = db.query(Video).filter(Video.id == video_id).first()
    assert video.sign_id == sign_id


def test_bulk_label_updates_multiple_videos(client, setup_videos, db):
    """PATCH /videos/bulk-label should update multiple videos"""
    video_ids = ["unlabeled-video-0", "unlabeled-video-1"]
    sign_id = "test-sign-id"

    response = client.patch(
        "/api/v1/videos/bulk-label",
        json={"video_ids": video_ids, "sign_id": sign_id}
    )

    assert response.status_code == 200

    # Verify all videos were updated
    for video_id in video_ids:
        video = db.query(Video).filter(Video.id == video_id).first()
        assert video.sign_id == sign_id

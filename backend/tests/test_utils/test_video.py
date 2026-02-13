"""Unit tests for video upload validation helpers."""

from __future__ import annotations

from app.utils.video import normalize_content_type, validate_video_upload


def test_normalize_content_type_strips_parameters() -> None:
    """MIME parameters should be ignored during validation."""
    assert normalize_content_type("video/mp4;codecs=avc1.42E01E,mp4a.40.2") == "video/mp4"


def test_validate_video_upload_accepts_video_with_codec_parameters() -> None:
    """Uploads using MIME parameters should remain accepted."""
    validate_video_upload(
        filename="clip.mp4",
        content_type="video/mp4;codecs=avc1.42E01E,mp4a.40.2",
        duration_ms=3000,
        max_video_seconds=10,
    )


def test_validate_video_upload_rejects_non_video_content_type() -> None:
    """Non-video MIME types must still be rejected."""
    try:
        validate_video_upload(
            filename="clip.mp4",
            content_type="text/plain; charset=utf-8",
            duration_ms=3000,
            max_video_seconds=10,
        )
    except ValueError as exc:
        assert "content type" in str(exc).lower()
        return

    raise AssertionError("Expected ValueError for non-video content type")

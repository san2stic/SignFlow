"""Video helpers for validation and ffmpeg compression."""

from __future__ import annotations

import mimetypes
import os
import subprocess
from pathlib import Path

ALLOWED_VIDEO_MIME_TYPES = {
    "video/webm",
    "video/mp4",
    "video/quicktime",
    "video/x-m4v",
    "application/octet-stream",
}
ALLOWED_EXTENSIONS = {".webm", ".mp4", ".mov", ".m4v"}


def normalize_content_type(content_type: str | None) -> str | None:
    """Normalize MIME content type by stripping parameters and spacing."""
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower() or None


def validate_video_filename(filename: str) -> None:
    """Validate extension and MIME type heuristics for uploaded video."""
    extension = Path(filename).suffix.lower()
    mime, _ = mimetypes.guess_type(filename)
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported video extension. Allowed: {allowed}")
    if mime and mime not in ALLOWED_VIDEO_MIME_TYPES:
        raise ValueError("Unsupported video MIME type")


def validate_video_upload(
    *,
    filename: str,
    content_type: str | None,
    duration_ms: int,
    max_video_seconds: int,
) -> None:
    """Validate upload metadata for extension, MIME type, and duration."""
    validate_video_filename(filename)

    normalized_content_type = normalize_content_type(content_type)
    if normalized_content_type and normalized_content_type not in ALLOWED_VIDEO_MIME_TYPES:
        if not normalized_content_type.startswith("video/"):
            raise ValueError("Unsupported video content type")

    if duration_ms < 0:
        raise ValueError("Invalid duration metadata")

    if duration_ms > max_video_seconds * 1000:
        raise ValueError(f"Video is longer than allowed maximum ({max_video_seconds}s)")


def compress_video(input_path: str, output_path: str) -> None:
    """Compress video using ffmpeg H.264 baseline profile."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-acodec",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

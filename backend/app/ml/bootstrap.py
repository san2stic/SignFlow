"""Dataset bootstrap helpers for preparing WLASL subsets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re
from urllib.request import Request, urlopen

WLASL_DEFAULT_METADATA_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"

_YOUTUBE_HOST_PATTERN = re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE)
_DIRECT_VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv")
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class BootstrapClip:
    """One clip candidate from the dataset metadata."""

    clip_id: str
    source_url: str
    split: str
    frame_start: int | None
    frame_end: int | None
    source_kind: str


@dataclass(frozen=True)
class BootstrapSign:
    """One selected sign entry containing candidate clips."""

    gloss: str
    slug: str
    clips: list[BootstrapClip]


def slugify_label(label: str) -> str:
    """Create a stable lowercase slug from a sign label."""
    normalized = _NON_ALNUM_PATTERN.sub("-", label.strip().lower()).strip("-")
    return normalized or "unnamed"


def classify_source_url(url: str) -> str:
    """Classify source URL into direct/youtube/unknown/missing."""
    raw = url.strip()
    if not raw:
        return "missing"

    lower = raw.lower()
    if lower.startswith(("http://", "https://")) and lower.endswith(_DIRECT_VIDEO_SUFFIXES):
        return "direct"
    if _YOUTUBE_HOST_PATTERN.search(lower):
        return "youtube"
    return "unknown"


def load_json_from_source(source: str, timeout_sec: int = 30) -> Any:
    """Load JSON payload from local path or HTTP(S) URL."""
    if source.startswith(("http://", "https://")):
        request = Request(source, headers={"User-Agent": "SignFlow/1.0 DatasetBootstrap"})
        with urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))

    path = Path(source).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def parse_wlasl_entries(
    raw_entries: list[dict[str, Any]],
    *,
    max_signs: int = 100,
    clips_per_sign: int = 20,
) -> list[BootstrapSign]:
    """
    Parse and select a WLASL subset.

    Selection strategy:
    - Keep entries with at least one usable instance.
    - Sort signs by available clip count descending, then alphabetically.
    - Keep top ``max_signs``.
    - Keep up to ``clips_per_sign`` clips per sign with split priority train->val->test.
    """
    if max_signs < 1:
        raise ValueError("max_signs must be >= 1")
    if clips_per_sign < 1:
        raise ValueError("clips_per_sign must be >= 1")

    candidates: list[BootstrapSign] = []
    for entry in raw_entries:
        gloss = str(entry.get("gloss", "")).strip()
        if not gloss:
            continue

        clips: list[BootstrapClip] = []
        instances = entry.get("instances") or []
        if not isinstance(instances, list):
            continue

        for index, instance in enumerate(instances):
            if not isinstance(instance, dict):
                continue

            video_id = str(instance.get("video_id", "")).strip()
            source_url = str(instance.get("url", "")).strip()
            if not source_url and not video_id:
                continue

            clip_id = video_id or f"{slugify_label(gloss)}-{index:04d}"
            split = str(instance.get("split", "train")).strip().lower() or "train"
            frame_start = _safe_int(instance.get("frame_start"))
            frame_end = _safe_int(instance.get("frame_end"))

            clips.append(
                BootstrapClip(
                    clip_id=clip_id,
                    source_url=source_url,
                    split=split,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    source_kind=classify_source_url(source_url),
                )
            )

        if clips:
            candidates.append(BootstrapSign(gloss=gloss, slug=slugify_label(gloss), clips=clips))

    candidates.sort(key=lambda sign: (-len(sign.clips), sign.gloss.lower()))

    selected: list[BootstrapSign] = []
    for sign in candidates[:max_signs]:
        ordered = sorted(sign.clips, key=lambda clip: (_split_order(clip.split), clip.clip_id))
        selected.append(BootstrapSign(gloss=sign.gloss, slug=sign.slug, clips=ordered[:clips_per_sign]))

    return selected


def build_manifest(
    *,
    dataset: str,
    metadata_source: str,
    signs: list[BootstrapSign],
    max_signs: int,
    clips_per_sign: int,
) -> dict[str, Any]:
    """Build deterministic JSON-serializable manifest for selected subset."""
    clips_total = sum(len(sign.clips) for sign in signs)
    breakdown = {
        "direct": 0,
        "youtube": 0,
        "unknown": 0,
        "missing": 0,
    }

    serialized_signs: list[dict[str, Any]] = []
    for sign in signs:
        serialized_clips: list[dict[str, Any]] = []
        for clip in sign.clips:
            breakdown[clip.source_kind] = breakdown.get(clip.source_kind, 0) + 1
            serialized_clips.append(
                {
                    "clip_id": clip.clip_id,
                    "source_url": clip.source_url,
                    "split": clip.split,
                    "frame_start": clip.frame_start,
                    "frame_end": clip.frame_end,
                    "source_kind": clip.source_kind,
                }
            )
        serialized_signs.append(
            {
                "gloss": sign.gloss,
                "slug": sign.slug,
                "clip_count": len(serialized_clips),
                "clips": serialized_clips,
            }
        )

    return {
        "dataset": dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_source": metadata_source,
        "selection": {
            "max_signs": max_signs,
            "clips_per_sign": clips_per_sign,
        },
        "stats": {
            "sign_count": len(signs),
            "clip_count": clips_total,
            "source_kind_breakdown": breakdown,
        },
        "signs": serialized_signs,
    }


def _safe_int(value: Any) -> int | None:
    """Best-effort integer conversion with ``None`` fallback."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _split_order(split: str) -> int:
    """Stable ordering for train/val/test splits."""
    lowered = split.lower()
    if lowered == "train":
        return 0
    if lowered in {"val", "valid", "validation"}:
        return 1
    if lowered == "test":
        return 2
    return 3

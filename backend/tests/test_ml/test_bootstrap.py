"""Dataset bootstrap helpers tests."""

from __future__ import annotations

from app.ml.bootstrap import build_manifest, parse_wlasl_entries, slugify_label


def test_parse_wlasl_entries_applies_limits_and_ordering() -> None:
    """Parser should keep top signs by clip count and cap clips per sign."""
    raw = [
        {
            "gloss": "Bonjour",
            "instances": [
                {"video_id": "b1", "url": "https://example.org/a.mp4", "split": "test"},
                {"video_id": "b2", "url": "https://example.org/b.mp4", "split": "train"},
                {"video_id": "b3", "url": "https://example.org/c.mp4", "split": "val"},
            ],
        },
        {
            "gloss": "Merci",
            "instances": [
                {"video_id": "m1", "url": "https://example.org/m1.mp4", "split": "train"},
                {"video_id": "m2", "url": "https://example.org/m2.mp4", "split": "train"},
            ],
        },
        {
            "gloss": "Salut",
            "instances": [
                {"video_id": "s1", "url": "https://example.org/s1.mp4", "split": "train"},
            ],
        },
    ]

    selected = parse_wlasl_entries(raw, max_signs=2, clips_per_sign=2)
    assert [item.gloss for item in selected] == ["Bonjour", "Merci"]

    bonjour = selected[0]
    assert bonjour.slug == "bonjour"
    assert len(bonjour.clips) == 2
    assert [clip.clip_id for clip in bonjour.clips] == ["b2", "b3"]  # train before val before test


def test_build_manifest_includes_source_breakdown() -> None:
    """Manifest should include deterministic stats and sign payload."""
    raw = [
        {
            "gloss": "Need Help",
            "instances": [
                {"video_id": "h1", "url": "https://youtu.be/demo1", "split": "train"},
                {"video_id": "h2", "url": "https://cdn.example.com/h2.mp4", "split": "val"},
                {"video_id": "h3", "url": "", "split": "test"},
            ],
        }
    ]
    signs = parse_wlasl_entries(raw, max_signs=1, clips_per_sign=3)
    manifest = build_manifest(
        dataset="wlasl",
        metadata_source="local-test",
        signs=signs,
        max_signs=1,
        clips_per_sign=3,
    )

    assert manifest["stats"]["sign_count"] == 1
    assert manifest["stats"]["clip_count"] == 3
    assert manifest["stats"]["source_kind_breakdown"]["youtube"] == 1
    assert manifest["stats"]["source_kind_breakdown"]["direct"] == 1
    assert manifest["stats"]["source_kind_breakdown"]["missing"] == 1
    assert manifest["signs"][0]["slug"] == "need-help"


def test_slugify_label_fallback() -> None:
    """Slugify should return fallback value for empty/invalid labels."""
    assert slugify_label("  ") == "unnamed"
    assert slugify_label("Ça va?") == "a-va"

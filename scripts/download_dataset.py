#!/usr/bin/env python3
"""Bootstrap a subset dataset manifest (and optional clips) for SignFlow."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.bootstrap import (  # noqa: E402
    WLASL_DEFAULT_METADATA_URL,
    build_manifest,
    load_json_from_source,
    parse_wlasl_entries,
)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset bootstrap."""
    parser = argparse.ArgumentParser(
        description="Prepare a WLASL subset for SignFlow training bootstrap.",
    )
    parser.add_argument(
        "--dataset",
        default="wlasl",
        choices=["wlasl", "autsl"],
        help="Dataset source (AUTSL metadata bootstrap is not implemented yet).",
    )
    parser.add_argument(
        "--metadata-source",
        default=WLASL_DEFAULT_METADATA_URL,
        help="Metadata source URL or local JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "backend" / "data" / "datasets" / "wlasl"),
        help="Directory where manifest/report/clips are written.",
    )
    parser.add_argument("--max-signs", type=int, default=100, help="Maximum number of signs to keep.")
    parser.add_argument("--clips-per-sign", type=int, default=20, help="Maximum clips to keep per sign.")
    parser.add_argument("--timeout-sec", type=int, default=30, help="Network timeout for metadata/downloads.")
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download available clips (direct URLs + YouTube via yt-dlp).",
    )
    parser.add_argument(
        "--yt-dlp-binary",
        default="yt-dlp",
        help="Binary to use for YouTube clip downloads.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download clips that already exist in output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write metadata/manifest, never download clips.",
    )
    return parser.parse_args()


def main() -> int:
    """Run dataset bootstrap workflow and return process exit code."""
    args = parse_args()

    if args.dataset != "wlasl":
        print("AUTSL bootstrap is not implemented yet. Use --dataset wlasl.", file=sys.stderr)
        return 2

    if args.max_signs < 1 or args.clips_per_sign < 1:
        print("--max-signs and --clips-per-sign must both be >= 1", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading metadata from: {args.metadata_source}")
    raw = load_json_from_source(args.metadata_source, timeout_sec=args.timeout_sec)
    if not isinstance(raw, list):
        print("Metadata payload is not a list.", file=sys.stderr)
        return 2

    metadata_raw_path = output_dir / "metadata_raw.json"
    metadata_raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[2/4] Selecting dataset subset")
    signs = parse_wlasl_entries(
        raw,
        max_signs=args.max_signs,
        clips_per_sign=args.clips_per_sign,
    )
    manifest = build_manifest(
        dataset="wlasl",
        metadata_source=args.metadata_source,
        signs=signs,
        max_signs=args.max_signs,
        clips_per_sign=args.clips_per_sign,
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = manifest["stats"]
    print(
        f"Selected {stats['sign_count']} signs and {stats['clip_count']} clips "
        f"(direct={stats['source_kind_breakdown']['direct']}, "
        f"youtube={stats['source_kind_breakdown']['youtube']}, "
        f"unknown={stats['source_kind_breakdown']['unknown']}, "
        f"missing={stats['source_kind_breakdown']['missing']})"
    )
    print(f"Manifest written to: {manifest_path}")

    if args.dry_run or not args.download_videos:
        if args.dry_run:
            print("Dry-run enabled, skipping download step.")
        else:
            print("Download step disabled. Re-run with --download-videos to fetch clips.")
        return 0

    print("[3/4] Downloading clips")
    report = download_clips(
        manifest=manifest,
        output_dir=output_dir,
        timeout_sec=args.timeout_sec,
        yt_dlp_binary=args.yt_dlp_binary,
        skip_existing=args.skip_existing,
    )

    report_path = output_dir / "download_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/4] Completed")
    print(
        f"Downloaded={report['summary']['downloaded']} "
        f"existing={report['summary']['existing']} "
        f"unsupported={report['summary']['unsupported']} "
        f"failed={report['summary']['failed']}"
    )
    print(f"Report written to: {report_path}")
    return 0


def download_clips(
    *,
    manifest: dict[str, Any],
    output_dir: Path,
    timeout_sec: int,
    yt_dlp_binary: str,
    skip_existing: bool,
) -> dict[str, Any]:
    """Download available clips from manifest and return execution report."""
    clips_root = output_dir / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    yt_dlp_available = shutil.which(yt_dlp_binary) is not None

    summary = {
        "downloaded": 0,
        "existing": 0,
        "unsupported": 0,
        "failed": 0,
    }
    entries: list[dict[str, Any]] = []

    signs: list[dict[str, Any]] = manifest.get("signs", [])
    for sign in signs:
        sign_slug = _safe_slug(str(sign.get("slug", "unknown")))
        sign_dir = clips_root / sign_slug
        sign_dir.mkdir(parents=True, exist_ok=True)

        for clip in sign.get("clips", []):
            clip_id = _safe_slug(str(clip.get("clip_id", "")) or "clip")
            source_url = str(clip.get("source_url", "")).strip()
            source_kind = str(clip.get("source_kind", "unknown")).strip().lower()
            destination = sign_dir / f"{clip_id}{_extension_for_source(source_url)}"

            result = {
                "sign": sign_slug,
                "clip_id": clip_id,
                "source_kind": source_kind,
                "source_url": source_url,
                "path": str(destination),
                "status": "",
                "error": None,
            }

            if skip_existing and destination.exists():
                result["status"] = "existing"
                summary["existing"] += 1
                entries.append(result)
                continue

            try:
                if source_kind == "direct" and source_url:
                    _download_direct(source_url, destination, timeout_sec)
                    result["status"] = "downloaded"
                    summary["downloaded"] += 1
                elif source_kind == "youtube" and source_url:
                    if not yt_dlp_available:
                        result["status"] = "unsupported"
                        result["error"] = f"{yt_dlp_binary} is not available in PATH"
                        summary["unsupported"] += 1
                    else:
                        _download_with_ytdlp(source_url, destination, yt_dlp_binary)
                        result["status"] = "downloaded"
                        summary["downloaded"] += 1
                else:
                    result["status"] = "unsupported"
                    result["error"] = "Source URL is missing or unsupported"
                    summary["unsupported"] += 1
            except Exception as exc:  # noqa: BLE001
                result["status"] = "failed"
                result["error"] = str(exc)
                summary["failed"] += 1

            entries.append(result)

    return {
        "summary": summary,
        "entries": entries,
    }


def _download_direct(source_url: str, destination: Path, timeout_sec: int) -> None:
    """Download one direct clip URL to destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(source_url, headers={"User-Agent": "SignFlow/1.0 DatasetBootstrap"})
    with urlopen(request, timeout=timeout_sec) as response:
        destination.write_bytes(response.read())


def _download_with_ytdlp(source_url: str, destination: Path, yt_dlp_binary: str) -> None:
    """Download one YouTube clip using yt-dlp."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        yt_dlp_binary,
        "--no-progress",
        "--no-warnings",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o",
        str(destination),
        source_url,
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        raise RuntimeError(stderr or stdout or "yt-dlp failed")


def _safe_slug(value: str) -> str:
    """Create a filesystem-safe identifier."""
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.lower()).strip("-")
    return cleaned or "item"


def _extension_for_source(source_url: str) -> str:
    """Guess destination file extension from source URL."""
    suffix = Path(urlparse(source_url).path).suffix.lower()
    if suffix in ALLOWED_VIDEO_EXTENSIONS:
        return suffix
    return ".mp4"


if __name__ == "__main__":
    raise SystemExit(main())

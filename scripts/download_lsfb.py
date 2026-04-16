#!/usr/bin/env python3
"""Download LSFB-ISOL dataset (poses / landmarks) from lsfb.info.unamur.be.

Uses the official ``lsfb-dataset`` Python library published on PyPI.
Only **poses** (MediaPipe landmarks) are downloaded by default (~10 GB).
Pass ``--include-videos`` to also fetch video clips (~25 GB extra).

Licence: CC BY-NC-SA 4.0 — non-commercial use only.
Citation:
    Fink, J., Frenay, B., Meurant, L. & Cleve, A. (2021).
    LSFB-CONT and LSFB-ISOL: Two New Datasets for Vision-Based
    Sign Language Recognition.  IJCNN 2021.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the LSFB-ISOL dataset for SignFlow training.",
    )
    parser.add_argument(
        "--destination",
        default=str(REPO_ROOT / "backend" / "data" / "datasets" / "lsfb_isol"),
        help="Directory where the dataset files will be stored.",
    )
    parser.add_argument(
        "--dataset",
        default="isol",
        choices=["isol", "cont"],
        help="LSFB dataset variant (default: isol for isolated sign recognition).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Subset of splits to download (e.g. train test). Default: all.",
    )
    parser.add_argument(
        "--landmarks",
        nargs="*",
        default=["pose", "left_hand", "right_hand"],
        help="Landmark groups to download (default: pose left_hand right_hand).",
    )
    parser.add_argument(
        "--include-face",
        action="store_true",
        help="Also download face landmarks (468 points, adds ~5 GB).",
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Download raw video clips as well (~25 GB for isol).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist locally (default: true).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-download even if files already exist.",
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        default=False,
        help="Disable SSL certificate verification (use if server cert is untrusted).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from lsfb_dataset import Downloader
    except ImportError:
        print(
            "ERROR: lsfb-dataset package not installed.\n"
            "Install it with: pip install lsfb-dataset",
            file=sys.stderr,
        )
        return 1

    destination = Path(args.destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    landmarks = list(args.landmarks)
    if args.include_face and "face" not in landmarks:
        landmarks.append("face")

    print("=" * 60)
    print("LSFB-ISOL Dataset Downloader")
    print("=" * 60)
    print(f"  Dataset:      {args.dataset}")
    print(f"  Destination:  {destination}")
    print(f"  Landmarks:    {landmarks}")
    print(f"  Videos:       {args.include_videos}")
    print(f"  Splits:       {args.splits or 'all'}")
    print(f"  Skip existing:{args.skip_existing}")
    print("=" * 60)

    downloader_kwargs: dict = {
        "dataset": args.dataset,
        "destination": str(destination),
        "include_videos": args.include_videos,
        "landmarks": landmarks,
        "skip_existing_files": args.skip_existing,
        "check_ssl": not args.no_ssl_verify,
        "max_parallel_connections": 5,
        "timeout": 120,
    }
    if args.splits:
        downloader_kwargs["splits"] = args.splits

    start = time.time()

    max_retries = 50
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[Attempt {attempt}/{max_retries}]")
            downloader = Downloader(**downloader_kwargs)
            downloader.download()
            break  # Success
        except Exception as exc:
            print(f"\nDownload interrupted: {exc}", file=sys.stderr)
            if attempt == max_retries:
                print("Max retries reached. Exiting.", file=sys.stderr)
                return 1
            print(f"Retrying in 10s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(10)

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # Write a small download report for traceability.
    report = {
        "dataset": args.dataset,
        "destination": str(destination),
        "landmarks": landmarks,
        "include_videos": args.include_videos,
        "splits": args.splits,
        "elapsed_seconds": round(elapsed, 1),
        "citation": (
            "Fink, J. et al. (2021). LSFB-CONT and LSFB-ISOL: "
            "Two New Datasets for Vision-Based Sign Language Recognition. IJCNN 2021."
        ),
        "licence": "CC BY-NC-SA 4.0",
    }
    report_path = destination / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Download complete in {minutes}m {seconds}s")
    print(f"Report saved to {report_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

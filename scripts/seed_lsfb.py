#!/usr/bin/env python3
"""Seed the database with LSFB-ISOL signs and their converted landmark videos.

Reads the LSFB-ISOL metadata (``instances.csv``, ``sign_to_index.csv``) and the
converted landmarks from ``converted/`` to create:
  • One ``Sign`` DB entry per unique gloss (category ``lsfb-isol``)
  • One ``Video`` DB entry per converted clip (``landmarks_extracted=True``)

Usage:
    python scripts/seed_lsfb.py [--lsfb-dir DIR] [--split train]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database import Base  # noqa: E402
from app.models.sign import Sign  # noqa: E402
from app.models.video import Video  # noqa: E402
from app.ml.lsfb_adapter import compute_detection_rate  # noqa: E402

# Ensure all models are imported so create_all works
import app.models.user  # noqa: E402, F401
import app.models.training  # noqa: E402, F401
import app.models.model_version  # noqa: E402, F401

try:
    import app.models.deployment  # noqa: E402, F401
except ImportError:
    pass

import numpy as np  # noqa: E402

from slugify import slugify  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed DB with LSFB-ISOL signs and video entries.")
    parser.add_argument(
        "--lsfb-dir",
        default=str(REPO_ROOT / "backend" / "data" / "datasets" / "lsfb_isol"),
        help="Root of the LSFB-ISOL dataset directory.",
    )
    parser.add_argument(
        "--converted-dir",
        default=None,
        help="Directory with converted landmarks (default: <lsfb-dir>/converted).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Only seed instances from a specific split (train, test).",
    )
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.5,
        help="Minimum detection rate to mark a video as trainable (default: 0.5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing to DB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    lsfb_dir = Path(args.lsfb_dir).expanduser().resolve()
    converted_dir = Path(args.converted_dir) if args.converted_dir else lsfb_dir / "converted"
    converted_dir = converted_dir.expanduser().resolve()

    instances_csv = lsfb_dir / "instances.csv"
    if not instances_csv.exists():
        print(f"ERROR: instances.csv not found at {instances_csv}", file=sys.stderr)
        return 1

    if not converted_dir.is_dir():
        print(f"ERROR: Converted directory not found: {converted_dir}", file=sys.stderr)
        print("Run scripts/convert_lsfb.py first.", file=sys.stderr)
        return 1

    # Load split filter if requested
    split_ids: set[str] | None = None
    if args.split:
        split_path = lsfb_dir / "metadata" / "splits" / f"{args.split}.json"
        if not split_path.exists():
            print(f"ERROR: Split file not found: {split_path}", file=sys.stderr)
            return 1
        data = json.loads(split_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            split_ids = set(data)
        elif isinstance(data, dict):
            split_ids = set()
            for v in data.values():
                if isinstance(v, list):
                    split_ids.update(v)
        print(f"Filtering to split '{args.split}': {len(split_ids or [])} instance IDs")

    # Parse instances.csv
    instances: list[dict] = []
    with open(instances_csv, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            instance_id = row.get("id", "").strip()
            gloss = row.get("sign", "").strip()
            if not instance_id or not gloss:
                continue
            if split_ids is not None and instance_id not in split_ids:
                continue
            instances.append({"id": instance_id, "sign": gloss, "signer": row.get("signer", "")})

    print("=" * 60)
    print("LSFB-ISOL Database Seeding")
    print("=" * 60)
    print(f"  Instances:      {len(instances)}")
    print(f"  Converted dir:  {converted_dir}")
    print(f"  Split:          {args.split or 'all'}")
    print(f"  Min det. rate:  {args.min_detection_rate}")
    print(f"  Dry run:        {args.dry_run}")

    # Collect unique glosses
    unique_glosses: dict[str, str] = {}  # gloss → slug
    for inst in instances:
        gloss = inst["sign"]
        if gloss not in unique_glosses:
            unique_glosses[gloss] = slugify(f"lsfb_{gloss}", lowercase=True, max_length=140)

    print(f"  Unique signs:   {len(unique_glosses)}")
    print("=" * 60)

    if args.dry_run:
        print("\nDRY RUN — no DB writes.")
        print(f"Would create {len(unique_glosses)} signs and up to {len(instances)} videos.")
        return 0

    # Connect to DB
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    start = time.time()
    signs_created = 0
    signs_existing = 0
    videos_created = 0
    videos_skipped = 0
    trainable_count = 0

    # Phase 1: Create Sign entries
    slug_to_sign_id: dict[str, str] = {}
    try:
        for gloss, sign_slug in unique_glosses.items():
            existing = session.query(Sign).filter(Sign.slug == sign_slug).first()
            if existing:
                slug_to_sign_id[sign_slug] = existing.id
                signs_existing += 1
                continue

            sign = Sign(
                name=gloss,
                slug=sign_slug,
                description=f"Signe LSFB-ISOL: {gloss}",
                category="lsfb-isol",
                tags=["lsfb", "isol", "v2"],
                variants=[],
                notes="",
                usage_count=0,
            )
            session.add(sign)
            session.flush()
            slug_to_sign_id[sign_slug] = sign.id
            signs_created += 1

        session.commit()
        print(f"\n[1/2] Signs: {signs_created} created, {signs_existing} already existed")

        # Phase 2: Create Video entries
        batch_size = 500
        for i, inst in enumerate(instances):
            instance_id = inst["id"]
            gloss = inst["sign"]
            sign_slug = unique_glosses[gloss]
            sign_id = slug_to_sign_id.get(sign_slug)

            if not sign_id:
                videos_skipped += 1
                continue

            landmarks_path = converted_dir / f"{instance_id}_landmarks.npy"
            if not landmarks_path.exists():
                videos_skipped += 1
                continue

            # Check if video already exists (by file_path)
            rel_path = f"datasets/lsfb_isol/converted/{instance_id}_landmarks.npy"
            existing_video = session.query(Video).filter(Video.file_path == rel_path).first()
            if existing_video:
                videos_skipped += 1
                continue

            # Compute detection rate from landmarks
            try:
                landmarks = np.load(landmarks_path)
                det_rate = compute_detection_rate(landmarks)
            except Exception:
                videos_skipped += 1
                continue

            is_trainable = det_rate >= args.min_detection_rate

            video = Video(
                sign_id=sign_id,
                file_path=rel_path,
                landmarks_extracted=True,
                landmarks_path=rel_path,
                detection_rate=det_rate,
                quality_score=det_rate,
                is_trainable=is_trainable,
                landmark_feature_dim=225,
                type="training",
                duration_ms=0,
                fps=25,
            )
            session.add(video)
            videos_created += 1
            if is_trainable:
                trainable_count += 1

            if (i + 1) % batch_size == 0:
                session.commit()
                print(f"  ... processed {i + 1}/{len(instances)} instances", end="\r")

        session.commit()

        # Update video_count on signs
        for sign_slug, sign_id in slug_to_sign_id.items():
            count = session.query(Video).filter(Video.sign_id == sign_id, Video.type == "training").count()
            session.query(Sign).filter(Sign.id == sign_id).update(
                {"video_count": count, "training_sample_count": count}
            )
        session.commit()

    except Exception as exc:
        session.rollback()
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
    finally:
        session.close()

    elapsed = time.time() - start

    print(f"\n[2/2] Videos: {videos_created} created, {videos_skipped} skipped")
    print()
    print("=" * 60)
    print("Seeding Summary")
    print("=" * 60)
    print(f"  Signs created:    {signs_created}")
    print(f"  Signs existing:   {signs_existing}")
    print(f"  Videos created:   {videos_created}")
    print(f"  Videos skipped:   {videos_skipped}")
    print(f"  Trainable clips:  {trainable_count}")
    print(f"  Elapsed:          {elapsed:.1f}s")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

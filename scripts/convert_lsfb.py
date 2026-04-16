#!/usr/bin/env python3
"""Batch-convert LSFB-ISOL separated body-part poses into SignFlow 225-dim arrays.

Reads the downloaded LSFB-ISOL ``poses/`` directory, concatenates and pads each
instance into a single ``[frames, 225]`` array, and writes the result to the
``converted/`` output directory.

Usage:
    python scripts/convert_lsfb.py [--lsfb-dir DIR] [--output-dir DIR] [--split train]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.lsfb_adapter import (  # noqa: E402
    convert_all_instances,
    load_split_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LSFB-ISOL poses to SignFlow 225-dim landmark arrays.",
    )
    parser.add_argument(
        "--lsfb-dir",
        default=str(REPO_ROOT / "backend" / "data" / "datasets" / "lsfb_isol"),
        help="Root of the downloaded LSFB-ISOL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for converted .npy files (default: <lsfb-dir>/converted).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Restrict conversion to a specific split (e.g. train, test).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-convert even if output file already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    lsfb_dir = Path(args.lsfb_dir).expanduser().resolve()
    output_dir = Path(args.output_dir) if args.output_dir else lsfb_dir / "converted"
    output_dir = output_dir.expanduser().resolve()

    instances_csv = lsfb_dir / "instances.csv"
    if not instances_csv.exists():
        print(f"ERROR: instances.csv not found at {instances_csv}", file=sys.stderr)
        print("Run scripts/download_lsfb.py first.", file=sys.stderr)
        return 1

    split_ids = None
    if args.split:
        split_path = lsfb_dir / "metadata" / "splits" / f"{args.split}.json"
        if not split_path.exists():
            print(f"ERROR: Split file not found: {split_path}", file=sys.stderr)
            return 1
        split_ids = load_split_json(split_path)
        print(f"Filtering to split '{args.split}': {len(split_ids)} instance IDs")

    print("=" * 60)
    print("LSFB-ISOL → SignFlow Landmark Conversion")
    print("=" * 60)
    print(f"  Source:      {lsfb_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Split:       {args.split or 'all'}")
    print(f"  Skip exist:  {not args.no_skip_existing}")
    print("=" * 60)

    start = time.time()

    stats = convert_all_instances(
        lsfb_dir=lsfb_dir,
        output_dir=output_dir,
        instances_csv=instances_csv,
        split_ids=split_ids,
        skip_existing=not args.no_skip_existing,
    )

    elapsed = time.time() - start

    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  Total instances:  {stats.total}")
    print(f"  Converted:        {stats.success}")
    print(f"  Skipped:          {stats.skipped}")
    print(f"  Failed:           {stats.failed}")
    print(f"  Unique signs:     {len(stats.per_sign_counts)}")
    print(f"  Elapsed:          {elapsed:.1f}s")

    if stats.total > 0:
        success_rate = (stats.success + stats.skipped) / stats.total * 100
        print(f"  Success rate:     {success_rate:.1f}%")

    # Write report
    report = {
        "total": stats.total,
        "success": stats.success,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "unique_signs": len(stats.per_sign_counts),
        "elapsed_seconds": round(elapsed, 1),
        "errors": stats.errors[:50],  # Cap for readability
        "per_sign_counts": dict(sorted(stats.per_sign_counts.items())),
    }
    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Report:           {report_path}")
    print("=" * 60)

    if stats.failed > 0:
        print(f"\nWARNING: {stats.failed} instances failed conversion.", file=sys.stderr)
        for err in stats.errors[:5]:
            print(f"  - {err['instance_id']}: {err['error']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Repair/activate a valid runtime model artifact in the SignFlow DB."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"

if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = f"sqlite:///{(BACKEND_DIR / 'data' / 'signflow.db').resolve()}"
if "MODEL_DIR" not in os.environ:
    os.environ["MODEL_DIR"] = str((BACKEND_DIR / "data" / "models").resolve())
os.chdir(BACKEND_DIR)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings
from app.database import SessionLocal
from app.models.model_version import ModelVersion
from app.models.training import TrainingSession
from app.utils.model_artifacts import discover_local_model_artifacts, resolve_model_artifact_path


def _infer_version_from_file(path: Path) -> str:
    match = re.search(r"model_(v\d+)$", path.stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    compact = re.sub(r"[^a-zA-Z0-9_-]", "-", path.stem).strip("-")
    if compact.startswith("model_"):
        compact = compact[len("model_") :]
    if not compact:
        compact = "v-repaired"
    return compact[:24]


def _load_checkpoint_info(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    labels = checkpoint.get("class_labels")
    if not isinstance(labels, list):
        labels = []
    labels = [str(item) for item in labels if str(item).strip()]

    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    eval_report = metadata.get("eval_report")
    accuracy = 0.0
    if isinstance(eval_report, dict):
        try:
            accuracy = float(eval_report.get("f1_macro") or eval_report.get("macro_f1") or 0.0)
        except (TypeError, ValueError):
            accuracy = 0.0
    if accuracy <= 0.0:
        try:
            metrics_history = checkpoint.get("metrics_history") or []
            if metrics_history:
                latest = metrics_history[-1]
                accuracy = float(getattr(latest, "val_accuracy", 0.0) or 0.0)
        except Exception:
            accuracy = 0.0

    num_classes = checkpoint.get("num_classes")
    if not isinstance(num_classes, int) or num_classes <= 0:
        num_classes = len(labels) if labels else 1

    return {
        "labels": labels,
        "metadata": metadata,
        "accuracy": accuracy,
        "num_classes": num_classes,
    }


def _ensure_training_session(db: Session) -> TrainingSession:
    existing = db.scalar(
        select(TrainingSession)
        .order_by(TrainingSession.created_at.desc())
    )
    if existing is not None:
        return existing

    session = TrainingSession(
        mode="full-retrain",
        status="completed",
        progress=100.0,
        config={"source": "repair_active_model"},
        metrics={"source": "repair_active_model"},
    )
    db.add(session)
    db.flush()
    return session


def _find_existing_by_artifact(db: Session, artifact_path: Path, *, model_dir: str, version: str) -> ModelVersion | None:
    direct_version = db.scalar(select(ModelVersion).where(ModelVersion.version == version))
    if direct_version is not None:
        return direct_version

    all_rows = db.scalars(select(ModelVersion)).all()
    for row in all_rows:
        resolved = resolve_model_artifact_path(row.file_path, model_dir=model_dir)
        if resolved is not None and str(resolved.resolve()) == str(artifact_path.resolve()):
            return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair active model registration for runtime inference.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Explicit .pt artifact path. If omitted, picks latest local artifact.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="Override version name (max 24 chars).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without committing.")
    args = parser.parse_args()

    settings = get_settings()

    if args.model_path.strip():
        selected = resolve_model_artifact_path(args.model_path.strip(), model_dir=settings.model_dir)
        if selected is None:
            print(f"[repair] model artifact not found: {args.model_path}")
            return 1
        artifact_path = selected
    else:
        artifacts = discover_local_model_artifacts(model_dir=settings.model_dir)
        if not artifacts:
            print(f"[repair] no local .pt artifact found (MODEL_DIR={settings.model_dir})")
            return 1
        artifact_path = artifacts[0]

    info = _load_checkpoint_info(artifact_path)
    version = (args.version.strip() or _infer_version_from_file(artifact_path))[:24]

    with SessionLocal() as db:
        training_session = _ensure_training_session(db)
        existing = _find_existing_by_artifact(
            db,
            artifact_path,
            model_dir=settings.model_dir,
            version=version,
        )

        if existing is None:
            target = ModelVersion(
                version=version,
                is_active=True,
                num_classes=int(info["num_classes"]),
                accuracy=float(info["accuracy"]),
                class_labels=list(info["labels"]),
                artifact_metadata=dict(info["metadata"]),
                training_session_id=training_session.id,
                file_path=str(artifact_path),
                file_size_mb=round(artifact_path.stat().st_size / 1_048_576, 4),
            )
            db.add(target)
            db.flush()
            action = "created"
        else:
            target = existing
            target.version = version
            target.num_classes = int(info["num_classes"])
            target.accuracy = float(info["accuracy"])
            target.class_labels = list(info["labels"])
            target.artifact_metadata = dict(info["metadata"])
            target.training_session_id = training_session.id
            target.file_path = str(artifact_path)
            target.file_size_mb = round(artifact_path.stat().st_size / 1_048_576, 4)
            action = "updated"

        for row in db.scalars(select(ModelVersion)).all():
            row.is_active = row.id == target.id

        if args.dry_run:
            db.rollback()
            print(
                f"[repair] dry-run: would {action} and activate version={version} "
                f"path={artifact_path} labels={len(info['labels'])}"
            )
            return 0

        db.commit()
        print(
            f"[repair] {action} and activated version={version} "
            f"model_id={target.id} path={artifact_path} labels={len(info['labels'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

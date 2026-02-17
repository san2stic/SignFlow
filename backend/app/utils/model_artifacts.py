"""Helpers to resolve model artifact paths across dev/container runtimes."""

from __future__ import annotations

import os
from pathlib import Path


def _repo_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "models"


def candidate_model_dirs(model_dir: str | Path | None = None) -> list[Path]:
    """Return candidate directories that may contain model artifacts."""
    candidates: list[Path] = []
    is_test_env = os.getenv("ENV", "").strip().lower() == "test"

    if model_dir:
        candidates.append(Path(model_dir).expanduser())

    if not is_test_env:
        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / "data" / "models",
                cwd / "backend" / "data" / "models",
                _repo_models_dir(),
            ]
        )

    deduped: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        resolved = item.resolve(strict=False)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def resolve_model_artifact_path(raw_path: str | Path | None, *, model_dir: str | Path | None = None) -> Path | None:
    """Resolve one model path against common runtime locations and return an existing file."""
    if raw_path is None:
        return None

    value = str(raw_path).strip()
    if not value:
        return None

    input_path = Path(value).expanduser()
    probes: list[Path] = []

    if input_path.is_absolute():
        probes.append(input_path)
    else:
        probes.append((Path.cwd() / input_path).resolve(strict=False))

    for directory in candidate_model_dirs(model_dir):
        probes.append((directory / input_path).resolve(strict=False))
        probes.append((directory / input_path.name).resolve(strict=False))

    seen: set[str] = set()
    for candidate in probes:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def discover_local_model_artifacts(*, model_dir: str | Path | None = None) -> list[Path]:
    """List local `.pt` artifacts ordered by most recent modification first."""
    discovered: list[Path] = []
    for directory in candidate_model_dirs(model_dir):
        if not directory.exists() or not directory.is_dir():
            continue
        for item in directory.glob("*.pt"):
            if item.is_file():
                discovered.append(item.resolve(strict=False))

    deduped: list[Path] = []
    seen: set[str] = set()
    for item in sorted(discovered, key=lambda path: path.stat().st_mtime, reverse=True):
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped

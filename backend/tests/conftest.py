"""Pytest environment isolation for backend tests.

These tests must never mutate the runtime SignFlow database or data directories.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import delete

# Configure an isolated filesystem root before app settings are imported.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="signflow-pytest-")).resolve()
_TEST_DB = _TEST_ROOT / "test.db"
_TEST_MODELS = _TEST_ROOT / "models"
_TEST_VIDEOS = _TEST_ROOT / "videos"
_TEST_EXPORTS = _TEST_ROOT / "exports"

os.environ["ENV"] = "test"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
os.environ["MODEL_DIR"] = str(_TEST_MODELS)
os.environ["VIDEO_DIR"] = str(_TEST_VIDEOS)
os.environ["EXPORT_DIR"] = str(_TEST_EXPORTS)
os.environ["SEARCH_BACKEND"] = "sql"
os.environ["TRAINING_USE_CELERY"] = "false"
os.environ["USE_TORCHSERVE"] = "false"
os.environ["MLFLOW_REGISTRY_ENABLED"] = "false"

from app.config import get_settings

get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_test_environment() -> None:
    """Prepare isolated directories and database schema once per test session."""
    _TEST_MODELS.mkdir(parents=True, exist_ok=True)
    _TEST_VIDEOS.mkdir(parents=True, exist_ok=True)
    _TEST_EXPORTS.mkdir(parents=True, exist_ok=True)

    import app.models  # noqa: F401
    from app.database import Base, engine

    Base.metadata.create_all(bind=engine)
    yield

    Base.metadata.drop_all(bind=engine)
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_each_test() -> None:
    """Clear persisted rows and reset cached translation pipeline for every test."""
    from app.api.translate import reload_pipeline
    from app.database import Base, SessionLocal

    with SessionLocal() as db:
        for table in reversed(Base.metadata.sorted_tables):
            db.execute(delete(table))
        db.commit()

    reload_pipeline()
    yield
    reload_pipeline()

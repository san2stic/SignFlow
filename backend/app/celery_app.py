"""Celery app configuration for asynchronous training tasks."""

from __future__ import annotations

from celery import Celery

from app.config import get_settings

settings = get_settings()
celery_app = Celery("signflow", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_default_queue="training",
    task_routes={
        "app.tasks.training.run_training_session": {"queue": "training"},
    },
    include=["app.tasks.training_tasks"],
)

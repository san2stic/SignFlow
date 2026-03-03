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
    worker_concurrency=settings.training_celery_concurrency,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

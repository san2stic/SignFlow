"""Celery tasks for model training execution."""

from __future__ import annotations

from celery import shared_task

from app.services.training_service import training_service


@shared_task(name="app.tasks.training.run_training_session", ignore_result=True, queue="training")
def run_training_session_task(session_id: str) -> None:
    """Run one training session worker from Celery."""
    training_service.run_training_session(session_id)

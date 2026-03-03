"""SQLAlchemy model for deployment history tracking."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class DeploymentHistory(Base):
    """Tracks every Git→Docker deployment attempt with full audit trail."""

    __tablename__ = "deployment_history"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # State machine: idle|fetching|pulling|building|deploying|success|error|rolled_back
    status = Column(String(20), nullable=False, default="pending", index=True)

    # Git metadata
    commit_hash = Column(String(40), nullable=True)
    previous_commit_hash = Column(String(40), nullable=True)
    commit_message = Column(Text, nullable=True)
    commit_author = Column(String(200), nullable=True)
    committed_at = Column(DateTime(timezone=True), nullable=True)

    # Build artifacts
    build_log = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Timing metrics
    build_duration_s = Column(Float, nullable=True)
    deploy_duration_s = Column(Float, nullable=True)
    total_duration_s = Column(Float, nullable=True)

    # Trigger and lineage
    triggered_by = Column(String(20), nullable=False, default="auto")
    rollback_of_id = Column(Integer, ForeignKey("deployment_history.id"), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Self-referential relationship for rollback chain
    rollback_origin = relationship(
        "DeploymentHistory",
        remote_side="DeploymentHistory.id",
        backref="rollbacks",
        foreign_keys=[rollback_of_id],
    )

"""Pydantic schemas for deployment history and updater service."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DeploymentHistoryBase(BaseModel):
    """Shared deployment history fields."""

    status: str
    triggered_by: str = "auto"
    commit_hash: Optional[str] = None
    previous_commit_hash: Optional[str] = None
    commit_message: Optional[str] = None
    commit_author: Optional[str] = None
    committed_at: Optional[datetime] = None
    build_log: Optional[str] = None
    error_message: Optional[str] = None
    build_duration_s: Optional[float] = None
    deploy_duration_s: Optional[float] = None
    total_duration_s: Optional[float] = None
    rollback_of_id: Optional[int] = None


class DeploymentHistoryCreate(DeploymentHistoryBase):
    """Schema for creating a new deployment history entry."""

    pass


class DeploymentHistoryRead(DeploymentHistoryBase):
    """Full deployment history entry returned from the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class UpdaterStatus(BaseModel):
    """Current state of the updater service pipeline."""

    state: str
    current_deployment_id: Optional[int] = None
    last_deployment: Optional[DeploymentHistoryRead] = None
    git_remote_url: str = ""
    git_branch: str
    local_commit: Optional[str] = None
    remote_commit: Optional[str] = None
    last_check_at: Optional[datetime] = None
    poll_interval_s: int
    auto_update_enabled: bool = True


class TriggerResponse(BaseModel):
    """Response schema for POST /trigger and POST /rollback endpoints."""

    deployment_id: int
    message: str
    rollback_of_id: Optional[int] = None


class TriggerRequest(BaseModel):
    """Request body for POST /trigger."""

    force: bool = False

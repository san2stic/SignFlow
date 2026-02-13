"""Pydantic schemas for sign resources."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SignBase(BaseModel):
    """Reusable fields for sign creation and update."""

    name: str = Field(min_length=1, max_length=120)
    description: Optional[str] = None
    category: Optional[str] = Field(default=None, max_length=80)
    tags: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)
    related_signs: list[UUID] = Field(default_factory=list)
    notes: Optional[str] = None


class SignCreate(SignBase):
    """Payload for creating a sign."""


class SignUpdate(BaseModel):
    """Payload for patching an existing sign."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = None
    category: Optional[str] = Field(default=None, max_length=80)
    tags: Optional[list[str]] = None
    variants: Optional[list[str]] = None
    related_signs: Optional[list[UUID]] = None
    notes: Optional[str] = None


class Sign(BaseModel):
    """Public sign schema exposed by API."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    description: Optional[str]
    category: Optional[str]
    tags: list[str]
    variants: list[str]
    related_signs: list[UUID]
    video_count: int
    training_sample_count: int
    accuracy: Optional[float]
    usage_count: int
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class SignListResponse(BaseModel):
    """Paginated sign list response."""

    items: list[Sign]
    page: int
    per_page: int
    total: int

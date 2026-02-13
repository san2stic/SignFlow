"""REST endpoint for overview statistics."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, Query

from app.api.deps import enforce_rate_limit, get_db
from app.models.model_version import ModelVersion
from app.models.sign import Sign
from app.models.training import TrainingSession
from app.models.video import Video

router = APIRouter()


@router.get("/overview", dependencies=[Depends(enforce_rate_limit)])
def stats_overview(db: Session = Depends(get_db)) -> dict:
    """Return global KPIs for dashboard."""
    total_signs = db.scalar(select(func.count()).select_from(Sign)) or 0
    total_videos = db.scalar(select(func.count()).select_from(Video)) or 0
    total_translations = db.scalar(select(func.sum(Sign.usage_count))) or 0

    active_model = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
    model_accuracy = active_model.accuracy if active_model else 0.0

    most_used = db.scalars(select(Sign).order_by(Sign.usage_count.desc()).limit(5)).all()
    recent_sessions = db.scalars(select(TrainingSession).order_by(TrainingSession.created_at.desc()).limit(5)).all()

    return {
        "total_signs": total_signs,
        "total_videos": total_videos,
        "model_accuracy": model_accuracy,
        "total_translations": total_translations,
        "most_used_signs": [{"sign": item.slug, "count": item.usage_count} for item in most_used],
        "recent_activity": [
            {"action": f"training:{item.status}", "timestamp": item.created_at.isoformat()}
            for item in recent_sessions
        ],
    }


@router.get("/accuracy-history", dependencies=[Depends(enforce_rate_limit)])
def stats_accuracy_history(
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Return recent model accuracy evolution for dashboard charts."""
    rows = db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)).all()
    rows = list(reversed(rows))
    return [
        {
            "id": item.id,
            "version": item.version,
            "accuracy": item.accuracy,
            "created_at": item.created_at.isoformat(),
            "is_active": item.is_active,
        }
        for item in rows
    ]


@router.get("/signs-per-category", dependencies=[Depends(enforce_rate_limit)])
def stats_signs_per_category(db: Session = Depends(get_db)) -> list[dict]:
    """Return dictionary distribution grouped by category."""
    rows = db.execute(
        select(Sign.category, func.count(Sign.id))
        .group_by(Sign.category)
        .order_by(func.count(Sign.id).desc())
    ).all()
    return [
        {
            "category": category if category else "uncategorized",
            "count": int(count),
        }
        for category, count in rows
    ]

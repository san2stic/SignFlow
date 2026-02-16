"""Administrative endpoints for search index management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import enforce_rate_limit, enforce_write_rate_limit, get_app_settings, get_db
from app.config import Settings
from app.services.search_service import SearchBackendUnavailable, search_service

router = APIRouter()


@router.post(
    "/reindex",
    dependencies=[Depends(enforce_rate_limit), Depends(enforce_write_rate_limit)],
)
def reindex_search_index(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> dict:
    """Recreate and fully repopulate the search index from relational data."""
    if settings.search_backend != "elasticsearch":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search backend is not Elasticsearch",
        )

    try:
        return search_service.reindex_all(db)
    except SearchBackendUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search backend unavailable",
        ) from exc

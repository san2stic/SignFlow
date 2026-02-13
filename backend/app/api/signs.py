"""REST endpoints for sign CRUD and sign-scoped video operations."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import enforce_rate_limit, get_app_settings, get_db
from app.config import Settings
from app.schemas.sign import Sign, SignCreate, SignListResponse, SignUpdate
from app.schemas.video import Video, VideoCreateMetadata
from app.services.media_service import MediaService
from app.services.sign_service import SignService

router = APIRouter()
sign_service = SignService()
media_service = MediaService()


@router.get("", response_model=SignListResponse)
def list_signs(
    search: str | None = None,
    category: str | None = None,
    tag: list[str] = Query(default=[]),
    sort: str = "name",
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> SignListResponse:
    """List signs with search, filters, and pagination."""
    return sign_service.list_signs(
        db,
        search=search,
        category=category,
        tag=tag,
        sort=sort,
        page=page,
        per_page=per_page,
    )


@router.get("/{sign_id}", response_model=Sign)
def get_sign(sign_id: str, db: Session = Depends(get_db)) -> Sign:
    """Get one sign by UUID."""
    sign = sign_service.get_sign(db, sign_id)
    if not sign:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sign not found")
    return sign


@router.get("/{sign_id}/backlinks")
def get_sign_backlinks(sign_id: str, db: Session = Depends(get_db)) -> dict:
    """List signs that reference this sign through graph relations."""
    sign = sign_service.get_sign(db, sign_id)
    if not sign:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sign not found")
    backlinks = sign_service.get_backlinks(db, sign_id)
    return {"sign_id": sign_id, "backlinks": backlinks}


@router.post("", response_model=Sign, status_code=status.HTTP_201_CREATED, dependencies=[Depends(enforce_rate_limit)])
def create_sign(payload: SignCreate, db: Session = Depends(get_db)) -> Sign:
    """Create a sign entry in the dictionary."""
    return sign_service.create_sign(db, payload)


@router.put("/{sign_id}", response_model=Sign)
def update_sign(sign_id: str, payload: SignUpdate, db: Session = Depends(get_db)) -> Sign:
    """Update existing sign metadata and relations."""
    sign = sign_service.update_sign(db, sign_id, payload)
    if not sign:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sign not found")
    return sign


@router.delete("/{sign_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sign(sign_id: str, db: Session = Depends(get_db)) -> None:
    """Delete sign and linked resources."""
    deleted = sign_service.delete_sign(db, sign_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sign not found")


@router.post("/{sign_id}/videos", response_model=Video, status_code=status.HTTP_201_CREATED, dependencies=[Depends(enforce_rate_limit)])
def upload_sign_video(
    sign_id: str,
    file: UploadFile = File(...),
    type: str = Form("reference"),
    metadata: str | None = Form(default=None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> Video:
    """Upload video attached to a sign and persist metadata."""
    if type not in {"training", "reference", "example"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid type")

    meta = VideoCreateMetadata()
    if metadata:
        try:
            meta = VideoCreateMetadata(**json.loads(metadata))
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid metadata payload") from exc

    return media_service.add_video(
        db,
        sign_id=sign_id,
        upload=file,
        video_type=type,
        metadata=meta,
        settings=settings,
    )


@router.get("/{sign_id}/videos", response_model=list[Video])
def list_sign_videos(sign_id: str, db: Session = Depends(get_db)) -> list[Video]:
    """List videos linked to one sign."""
    return media_service.list_sign_videos(db, sign_id)

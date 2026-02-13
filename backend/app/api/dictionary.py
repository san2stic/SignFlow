"""REST endpoints for dictionary graph, search, and import/export."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db
from app.config import Settings
from app.services.dictionary_service import DictionaryService

router = APIRouter()
dictionary_service = DictionaryService()


@router.get("/graph")
def dictionary_graph(db: Session = Depends(get_db)) -> dict:
    """Return graph nodes and edges for dictionary visualization."""
    return dictionary_service.graph(db)


@router.get("/search")
def dictionary_search(
    q: str,
    fields: str = "all",
    db: Session = Depends(get_db),
) -> list[dict]:
    """Search dictionary by name/description/tags/all."""
    return dictionary_service.search(db, q=q, fields=fields)


@router.post("/export")
def dictionary_export(
    format: str = Form(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> FileResponse:
    """Export dictionary as JSON/Markdown/Obsidian vault ZIP."""
    try:
        path = dictionary_service.export(db, fmt=format, settings=settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(path=path, filename="dictionary_export.zip", media_type="application/zip")


@router.post("/import")
def dictionary_import(archive: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    """Import dictionary bundle from ZIP payload."""
    return dictionary_service.import_archive(db, archive)

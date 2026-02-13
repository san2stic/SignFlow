"""Dictionary API import/export behavior tests."""

from __future__ import annotations

import io
import zipfile
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal
from app.main import app
from app.models.sign import Sign


def test_dictionary_markdown_export_produces_markdown_bundle(monkeypatch, tmp_path) -> None:
    """Markdown export should produce .md files, not dictionary.json."""
    settings = get_settings()
    monkeypatch.setattr(settings, "export_dir", str(tmp_path))

    suffix = uuid4().hex[:8]
    with SessionLocal() as db:
        sign = Sign(
            name=f"Export Sign {suffix}",
            slug=f"export-sign-{suffix}",
            description="Description",
            notes="Notes with [[Export Sign]]",
            category="demo",
            tags=["demo"],
            variants=[],
        )
        db.add(sign)
        db.commit()

    with TestClient(app) as client:
        response = client.post("/api/v1/dictionary/export", data={"format": "markdown"})

    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    assert any(name.startswith("signs/") and name.endswith(".md") for name in names)
    assert "dictionary.json" not in names


def test_dictionary_import_accepts_obsidian_markdown() -> None:
    """Import should accept markdown-only bundles and return enriched metrics."""
    suffix = uuid4().hex[:8]
    markdown = (
        "---\n"
        f"slug: imported-{suffix}\n"
        "category: lsfb-v1\n"
        "tags: lsfb, v1\n"
        "---\n\n"
        f"# Imported {suffix}\n\n"
        "## Description\n"
        "Imported description.\n\n"
        "## Notes\n"
        "This note links to [[Imported Related]].\n"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"SignFlowVault/imported-{suffix}.md", markdown)
    buffer.seek(0)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/dictionary/import",
            files={"archive": ("import.zip", buffer.getvalue(), "application/zip")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_signs"] >= 1
    assert "errors" in payload

    with SessionLocal() as db:
        imported = db.scalar(select(Sign).where(Sign.slug.like(f"imported-{suffix}%")))
        assert imported is not None

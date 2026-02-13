"""Service layer for dictionary graph, search, and export/import operations."""

from __future__ import annotations

import json
import os
import re
import uuid
import zipfile
from pathlib import Path

from fastapi import UploadFile
from slugify import slugify
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from app.config import Settings
from app.models.sign import Sign
from app.utils.export import export_dictionary_json, export_dictionary_markdown, export_dictionary_obsidian
from app.utils.markdown import extract_wikilinks

_HEADING_RE = re.compile(r"(?m)^#\s+(.+?)\s*$")
_SECTION_TEMPLATE = r"(?ms)^##\s+{title}\s*$\n(.*?)(?=^##\s+|\Z)"


class DictionaryService:
    """Builds graph payloads and handles dictionary import/export."""

    def graph(self, db: Session) -> dict:
        """Return graph nodes/edges for interactive dictionary view."""
        signs = db.scalars(select(Sign).options(selectinload(Sign.related_signs))).all()

        nodes = [
            {
                "id": sign.id,
                "label": sign.name,
                "category": sign.category,
                "video_count": sign.video_count,
                "usage_count": sign.usage_count,
                "accuracy": sign.accuracy,
                "thumbnail_url": None,
            }
            for sign in signs
        ]

        edges = []
        seen = set()
        for sign in signs:
            for related in sign.related_signs:
                key = tuple(sorted([sign.id, related.id]))
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "source": sign.id,
                        "target": related.id,
                        "relation_type": "related",
                        "weight": 1,
                    }
                )

        return {"nodes": nodes, "edges": edges}

    def search(self, db: Session, q: str, fields: str) -> list[dict]:
        """Search signs across selected textual fields."""
        ilike = f"%{q}%"
        query = select(Sign)

        if fields == "name":
            query = query.where(Sign.name.ilike(ilike))
        elif fields == "description":
            query = query.where(Sign.description.ilike(ilike))
        elif fields == "tags":
            query = query.where(Sign.tags.contains([q]))
        else:
            query = query.where(or_(Sign.name.ilike(ilike), Sign.description.ilike(ilike), Sign.notes.ilike(ilike)))

        results = db.scalars(query.limit(100)).all()
        return [
            {
                "id": sign.id,
                "name": sign.name,
                "slug": sign.slug,
                "category": sign.category,
                "tags": sign.tags,
            }
            for sign in results
        ]

    def export(self, db: Session, fmt: str, settings: Settings) -> str:
        """Export dictionary to ZIP in JSON, Markdown, or Obsidian format."""
        signs = db.scalars(select(Sign).options(selectinload(Sign.related_signs)).order_by(Sign.name.asc())).all()
        Path(settings.export_dir).mkdir(parents=True, exist_ok=True)
        output = os.path.join(settings.export_dir, f"dictionary_{uuid.uuid4()}.zip")

        if fmt == "json":
            return export_dictionary_json(signs, output)
        if fmt == "markdown":
            return export_dictionary_markdown(signs, output)
        if fmt == "obsidian-vault":
            return export_dictionary_obsidian(signs, output)
        raise ValueError("Unsupported export format")

    def import_archive(self, db: Session, archive: UploadFile) -> dict:
        """Import dictionary bundle from JSON or markdown-based archives."""
        tmp_path = Path("/tmp") / f"signflow_import_{uuid.uuid4()}.zip"
        tmp_path.write_bytes(archive.file.read())

        existing_slugs = set(db.scalars(select(Sign.slug)).all())
        imported_signs = 0
        imported_notes = 0
        skipped = 0
        errors: list[str] = []
        pending: list[dict] = []

        with zipfile.ZipFile(tmp_path, "r") as zf:
            names = zf.namelist()

            if "dictionary.json" in names:
                try:
                    raw = zf.read("dictionary.json").decode("utf-8")
                    payload = json.loads(raw)
                    if isinstance(payload, list):
                        for item in payload:
                            if not isinstance(item, dict):
                                skipped += 1
                                continue
                            pending.append(self._normalize_json_item(item))
                    else:
                        errors.append("dictionary.json must contain a list payload")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Failed to parse dictionary.json: {exc}")

            markdown_files = [
                name
                for name in names
                if name.lower().endswith(".md")
                and not name.lower().endswith("readme.md")
                and "/." not in name
            ]
            for path in markdown_files:
                try:
                    content = zf.read(path).decode("utf-8")
                    fallback_slug = slugify(Path(path).stem) or f"imported-{uuid.uuid4().hex[:8]}"
                    pending.append(self._parse_markdown_item(content, fallback_slug=fallback_slug))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Failed to parse markdown file {path}: {exc}")
                    skipped += 1

        if not pending:
            return {
                "imported_signs": 0,
                "imported_notes": 0,
                "skipped": skipped,
                "errors": errors or ["No supported dictionary payload found in archive"],
            }

        imported_by_id: dict[str, set[str]] = {}
        for item in pending:
            name = str(item.get("name") or "").strip()
            if not name:
                skipped += 1
                continue

            raw_slug = str(item.get("slug") or name).strip()
            normalized_slug = slugify(raw_slug) or f"imported-{uuid.uuid4().hex[:8]}"
            candidate_slug = self._resolve_unique_slug(normalized_slug, existing_slugs)
            existing_slugs.add(candidate_slug)

            sign = Sign(
                name=name,
                slug=candidate_slug,
                description=item.get("description"),
                category=item.get("category"),
                tags=item.get("tags", []),
                variants=item.get("variants", []),
                notes=item.get("notes"),
            )
            db.add(sign)
            db.flush()

            imported_signs += 1
            if sign.notes and sign.notes.strip():
                imported_notes += 1
            imported_by_id[sign.id] = set(item.get("relation_hints", []))

        # Resolve wikilinks and explicit relation hints after all rows are inserted.
        all_signs = db.scalars(select(Sign).options(selectinload(Sign.related_signs))).all()
        by_id = {sign.id: sign for sign in all_signs}
        by_slug = {sign.slug.lower(): sign for sign in all_signs if sign.slug}
        by_name = {sign.name.lower(): sign for sign in all_signs if sign.name}

        for sign_id, hints in imported_by_id.items():
            source = by_id.get(sign_id)
            if not source:
                continue
            for hint in hints:
                key = hint.lower()
                target = by_slug.get(key) or by_name.get(key)
                if not target or target.id == source.id:
                    continue
                if target not in source.related_signs:
                    source.related_signs.append(target)
                if source not in target.related_signs:
                    target.related_signs.append(source)

        db.commit()
        return {
            "imported_signs": imported_signs,
            "imported_notes": imported_notes,
            "skipped": skipped,
            "errors": errors,
        }

    @staticmethod
    def _normalize_json_item(item: dict) -> dict:
        """Normalize JSON import payload into internal sign structure."""
        name = str(item.get("name") or "").strip()
        description = item.get("description")
        notes = item.get("notes")
        related = item.get("related_signs")
        relation_hints: list[str] = []
        if isinstance(related, list):
            relation_hints.extend([str(value).strip() for value in related if str(value).strip()])
        relation_hints.extend(extract_wikilinks(description if isinstance(description, str) else None))
        relation_hints.extend(extract_wikilinks(notes if isinstance(notes, str) else None))

        return {
            "name": name,
            "slug": item.get("slug"),
            "description": description if isinstance(description, str) else None,
            "category": item.get("category") if isinstance(item.get("category"), str) else None,
            "tags": item.get("tags") if isinstance(item.get("tags"), list) else [],
            "variants": item.get("variants") if isinstance(item.get("variants"), list) else [],
            "notes": notes if isinstance(notes, str) else None,
            "relation_hints": relation_hints,
        }

    def _parse_markdown_item(self, markdown: str, *, fallback_slug: str) -> dict:
        """Parse one markdown sign note (Obsidian-like) into import payload."""
        frontmatter, body = self._parse_frontmatter(markdown)
        heading = self._extract_heading(body)
        description = self._extract_section(body, "Description")
        notes = self._extract_section(body, "Notes")

        if not description:
            # Fallback to body without heading for generic markdown files.
            description = body.strip() if body.strip() else None

        name = heading or str(frontmatter.get("name") or fallback_slug).replace("_", " ").strip()
        tags = self._split_csv(frontmatter.get("tags", ""))
        variants = self._split_csv(frontmatter.get("variants", ""))
        explicit_related = self._split_csv(frontmatter.get("related_signs", ""))

        relation_hints = []
        relation_hints.extend(explicit_related)
        relation_hints.extend(extract_wikilinks(description))
        relation_hints.extend(extract_wikilinks(notes))

        return {
            "name": name,
            "slug": frontmatter.get("slug") or fallback_slug,
            "description": description,
            "category": (frontmatter.get("category") or None),
            "tags": tags,
            "variants": variants,
            "notes": notes,
            "relation_hints": relation_hints,
        }

    @staticmethod
    def _parse_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
        """Parse markdown frontmatter and return (frontmatter, body)."""
        stripped = markdown.lstrip()
        if not stripped.startswith("---"):
            return {}, markdown

        lines = stripped.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, markdown

        boundary = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                boundary = idx
                break
        if boundary is None:
            return {}, markdown

        frontmatter: dict[str, str] = {}
        for line in lines[1:boundary]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip().lower()] = value.strip()

        body = "\n".join(lines[boundary + 1 :]).strip()
        return frontmatter, body

    @staticmethod
    def _extract_heading(markdown: str) -> str | None:
        """Extract first markdown heading."""
        match = _HEADING_RE.search(markdown)
        if not match:
            return None
        return match.group(1).strip()

    @staticmethod
    def _extract_section(markdown: str, title: str) -> str | None:
        """Extract markdown section content by heading title."""
        pattern = re.compile(_SECTION_TEMPLATE.format(title=re.escape(title)))
        match = pattern.search(markdown)
        if not match:
            return None
        content = match.group(1).strip()
        return content or None

    @staticmethod
    def _split_csv(value: str) -> list[str]:
        """Parse comma-separated values into a normalized string list."""
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    @staticmethod
    def _resolve_unique_slug(initial_slug: str, existing_slugs: set[str]) -> str:
        """Resolve slug collisions by appending an incrementing suffix."""
        candidate = initial_slug
        suffix = 2
        while candidate in existing_slugs:
            candidate = f"{initial_slug}-{suffix}"
            suffix += 1
        return candidate

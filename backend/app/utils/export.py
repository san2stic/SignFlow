"""Dictionary export/import helpers for JSON/Markdown/Obsidian vault formats."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from app.models.sign import Sign


def _serialize_sign(sign: Sign) -> dict:
    """Serialize one sign object to JSON-compatible metadata."""
    related = []
    for candidate in getattr(sign, "related_signs", []) or []:
        if getattr(candidate, "slug", None):
            related.append(candidate.slug)

    return {
        "id": sign.id,
        "name": sign.name,
        "slug": sign.slug,
        "description": sign.description,
        "category": sign.category,
        "tags": sign.tags or [],
        "variants": sign.variants or [],
        "related_signs": related,
        "notes": sign.notes,
        "video_count": sign.video_count,
        "training_sample_count": sign.training_sample_count,
        "accuracy": sign.accuracy,
        "usage_count": sign.usage_count,
        "created_at": sign.created_at.isoformat() if sign.created_at else None,
        "updated_at": sign.updated_at.isoformat() if sign.updated_at else None,
    }


def _build_markdown_document(sign: Sign) -> str:
    """Render one sign as markdown with frontmatter metadata."""
    related = []
    for candidate in getattr(sign, "related_signs", []) or []:
        if getattr(candidate, "name", None):
            related.append(candidate.name)

    lines = [
        "---",
        f"slug: {sign.slug}",
        f"category: {sign.category or ''}",
        f"tags: {', '.join(sign.tags or [])}",
        f"variants: {', '.join(sign.variants or [])}",
        f"related_signs: {', '.join(related)}",
        "---",
        "",
        f"# {sign.name}",
        "",
        "## Description",
        sign.description or "",
        "",
        "## Notes",
        sign.notes or "",
        "",
        "## Stats",
        f"- usage_count: {sign.usage_count}",
        f"- video_count: {sign.video_count}",
        f"- training_sample_count: {sign.training_sample_count}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _zip_directory(root: Path, output_zip_path: str) -> str:
    """Zip a directory content recursively."""
    with zipfile.ZipFile(output_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            archive.write(candidate, arcname=str(candidate.relative_to(root)))
    return output_zip_path


def export_dictionary_json(signs: list[Sign], output_zip_path: str) -> str:
    """Export signs metadata to a zipped JSON file."""
    payload = [_serialize_sign(sign) for sign in signs]

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "dictionary.json"
        data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with zipfile.ZipFile(output_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(data_path, arcname="dictionary.json")

    return output_zip_path


def export_dictionary_markdown(signs: list[Sign], output_zip_path: str) -> str:
    """Export signs as a generic markdown bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        signs_dir = root / "signs"
        signs_dir.mkdir(parents=True, exist_ok=True)

        for sign in signs:
            file_path = signs_dir / f"{sign.slug}.md"
            file_path.write_text(_build_markdown_document(sign), encoding="utf-8")

        readme = root / "README.md"
        readme.write_text(
            "# SignFlow Dictionary Export\n\n"
            "This archive contains markdown files in `signs/`.\n",
            encoding="utf-8",
        )

        return _zip_directory(root=root, output_zip_path=output_zip_path)


def export_dictionary_obsidian(signs: list[Sign], output_zip_path: str) -> str:
    """Export each sign into markdown files compatible with Obsidian vaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        vault_dir = root / "SignFlowVault"
        vault_dir.mkdir(parents=True, exist_ok=True)

        for sign in signs:
            file_path = vault_dir / f"{sign.slug}.md"
            file_path.write_text(_build_markdown_document(sign), encoding="utf-8")

        index_path = vault_dir / "Index.md"
        index_lines = ["# SignFlow Vault", ""]
        for sign in sorted(signs, key=lambda item: item.name.lower()):
            index_lines.append(f"- [[{sign.slug}]]")
        index_path.write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8")

        attachments_dir = vault_dir / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)
        (attachments_dir / ".gitkeep").write_text("", encoding="utf-8")

        return _zip_directory(root=root, output_zip_path=output_zip_path)

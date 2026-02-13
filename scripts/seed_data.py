"""Seed demo signs for local development."""

from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings  # noqa: E402
from app.database import Base  # noqa: E402
from app.models.sign import Sign  # noqa: E402


def main() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()

    lsfb_v1_signs = [
        ("Bonjour", "lsfb_bonjour"),
        ("Merci", "lsfb_merci"),
        ("S'il vous plait", "lsfb_sil_vous_plait"),
        ("Oui", "lsfb_oui"),
        ("Non", "lsfb_non"),
        ("Pardon", "lsfb_pardon"),
        ("Au revoir", "lsfb_au_revoir"),
        ("Comment", "lsfb_comment"),
        ("Je", "lsfb_je"),
        ("Toi", "lsfb_toi"),
    ]

    for idx, (name, slug) in enumerate(lsfb_v1_signs):
        existing = session.query(Sign).filter(Sign.slug == slug).first()
        if existing:
            existing.name = name
            existing.category = "lsfb-v1"
            existing.tags = ["lsfb", "v1"]
            if not existing.description:
                existing.description = f"Signe LSFB V1: {name}"
            continue

        session.add(
            Sign(
                name=name,
                slug=slug,
                description=f"Signe LSFB V1: {name}",
                category="lsfb-v1",
                tags=["lsfb", "v1"],
                variants=[],
                notes="",
                usage_count=idx * 3,
            )
        )

    session.commit()
    session.close()
    print("Seed complete")


if __name__ == "__main__":
    main()

"""Service layer for sign CRUD and relationship graph maintenance."""

from __future__ import annotations

from typing import Optional

from slugify import slugify
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.models.sign import Sign, sign_relations
from app.schemas.sign import Sign as SignSchema
from app.schemas.sign import SignCreate, SignListResponse, SignUpdate
from app.utils.markdown import extract_wikilinks


class SignService:
    """Encapsulates sign dictionary operations and graph synchronization."""

    def list_signs(
        self,
        db: Session,
        *,
        search: Optional[str],
        category: Optional[str],
        tag: list[str],
        sort: str,
        page: int,
        per_page: int,
    ) -> SignListResponse:
        """Return paginated sign results with filters and sorting."""
        query = select(Sign).options(selectinload(Sign.related_signs))

        if search:
            ilike = f"%{search}%"
            query = query.where(
                or_(
                    Sign.name.ilike(ilike),
                    Sign.description.ilike(ilike),
                    Sign.notes.ilike(ilike),
                )
            )
        if category:
            query = query.where(Sign.category == category)
        for single_tag in tag:
            query = query.where(Sign.tags.contains([single_tag]))

        if sort == "created_at":
            query = query.order_by(Sign.created_at.desc())
        elif sort == "usage_count":
            query = query.order_by(Sign.usage_count.desc())
        else:
            query = query.order_by(Sign.name.asc())

        total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
        items = db.scalars(query.offset((page - 1) * per_page).limit(per_page)).all()

        return SignListResponse(
            items=[self._to_schema(item) for item in items],
            page=page,
            per_page=per_page,
            total=total,
        )

    def get_sign(self, db: Session, sign_id: str) -> SignSchema | None:
        """Fetch one sign with eager loaded relations."""
        sign = db.scalar(select(Sign).where(Sign.id == sign_id).options(selectinload(Sign.related_signs)))
        return self._to_schema(sign) if sign else None

    def get_backlinks(self, db: Session, sign_id: str) -> list[dict[str, str]]:
        """Return incoming relation references for one sign."""
        sources = db.scalars(
            select(Sign)
            .join(sign_relations, sign_relations.c.source_sign_id == Sign.id)
            .where(sign_relations.c.target_sign_id == sign_id)
            .order_by(Sign.name.asc())
        ).all()

        deduped: dict[str, dict[str, str]] = {}
        for source in sources:
            deduped[source.id] = {"id": source.id, "name": source.name, "slug": source.slug}
        return list(deduped.values())

    def create_sign(self, db: Session, payload: SignCreate) -> SignSchema:
        """Create sign and synchronize explicit + wikilink relations."""
        sign = Sign(
            name=payload.name,
            slug=self._generate_unique_slug(db, payload.name),
            description=payload.description,
            category=payload.category,
            tags=payload.tags,
            variants=payload.variants,
            notes=payload.notes,
        )
        db.add(sign)
        db.flush()

        self._set_relations(db, sign, [str(rel_id) for rel_id in payload.related_signs], payload.notes)

        db.commit()
        db.refresh(sign)
        return self._to_schema(sign)

    def update_sign(self, db: Session, sign_id: str, payload: SignUpdate) -> SignSchema | None:
        """Update sign fields and rebuild relationships if needed."""
        sign = db.scalar(select(Sign).where(Sign.id == sign_id).options(selectinload(Sign.related_signs)))
        if not sign:
            return None

        for field in ["name", "description", "category", "tags", "variants", "notes"]:
            value = getattr(payload, field)
            if value is not None:
                setattr(sign, field, value)

        if payload.name:
            sign.slug = self._generate_unique_slug(db, payload.name, current_id=sign.id)

        if payload.related_signs is not None or payload.notes is not None:
            explicit_ids = [str(rel_id) for rel_id in payload.related_signs] if payload.related_signs else []
            self._set_relations(db, sign, explicit_ids, sign.notes)

        db.commit()
        db.refresh(sign)
        return self._to_schema(sign)

    def delete_sign(self, db: Session, sign_id: str) -> bool:
        """Delete a sign and dependent records."""
        sign = db.get(Sign, sign_id)
        if not sign:
            return False
        db.delete(sign)
        db.commit()
        return True

    def _generate_unique_slug(self, db: Session, name: str, current_id: str | None = None) -> str:
        """Create a URL-safe unique slug from sign name."""
        base = slugify(name)
        candidate = base
        index = 2

        while True:
            query = select(Sign).where(Sign.slug == candidate)
            existing = db.scalar(query)
            if not existing or existing.id == current_id:
                return candidate
            candidate = f"{base}-{index}"
            index += 1

    def _set_relations(self, db: Session, sign: Sign, explicit_ids: list[str], notes: str | None) -> None:
        """Set bidirectional relations from explicit ids and markdown wikilinks."""
        links = extract_wikilinks(notes)
        link_matches = db.scalars(select(Sign).where(func.lower(Sign.name).in_([item.lower() for item in links]))).all()

        related_ids = set(explicit_ids)
        for linked_sign in link_matches:
            related_ids.add(linked_sign.id)

        relations = db.scalars(select(Sign).where(Sign.id.in_(related_ids))).all() if related_ids else []
        sign.related_signs = [candidate for candidate in relations if candidate.id != sign.id]

        for target in sign.related_signs:
            if sign not in target.related_signs:
                target.related_signs.append(sign)

    def _to_schema(self, sign: Sign) -> SignSchema:
        """Convert ORM sign object into API schema including relation IDs."""
        return SignSchema(
            id=sign.id,
            name=sign.name,
            slug=sign.slug,
            description=sign.description,
            category=sign.category,
            tags=sign.tags or [],
            variants=sign.variants or [],
            related_signs=[rel.id for rel in sign.related_signs],
            video_count=sign.video_count,
            training_sample_count=sign.training_sample_count,
            accuracy=sign.accuracy,
            usage_count=sign.usage_count,
            notes=sign.notes,
            created_at=sign.created_at,
            updated_at=sign.updated_at,
        )

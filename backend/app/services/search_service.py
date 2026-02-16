"""Elasticsearch-backed search/index service for sign resources."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Literal

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models.sign import Sign

logger = logging.getLogger(__name__)

_ALLOWED_FIELDS = {"all", "name", "description", "tags"}


class SearchBackendUnavailable(RuntimeError):
    """Raised when Elasticsearch cannot be reached or queried safely."""


class SearchService:
    """Centralized Elasticsearch index and query operations for sign search."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Elasticsearch | None = None
        self._index_ready = False
        self._lock = threading.Lock()

    @property
    def settings(self) -> Settings:
        """Expose active runtime settings."""
        return self._settings

    @property
    def index_name(self) -> str:
        """Resolve active index name from settings."""
        return self._settings.elasticsearch_index

    def is_enabled(self) -> bool:
        """Return True when Elasticsearch backend is configured."""
        return self._settings.search_backend == "elasticsearch"

    def ensure_available(self) -> None:
        """Check backend availability and ensure index exists."""
        if not self.is_enabled():
            return
        client = self._get_client()
        try:
            if not client.ping():
                raise SearchBackendUnavailable("Elasticsearch ping failed")
            self.ensure_index()
        except SearchBackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SearchBackendUnavailable(f"Elasticsearch unavailable: {exc}") from exc

    def ensure_index(self) -> None:
        """Create index lazily when missing."""
        if self._index_ready:
            return

        with self._lock:
            if self._index_ready:
                return

            client = self._get_client()
            try:
                if not client.indices.exists(index=self.index_name):
                    client.indices.create(
                        index=self.index_name,
                        settings=self._index_settings(),
                        mappings=self._index_mappings(),
                    )
                self._index_ready = True
            except Exception as exc:  # noqa: BLE001
                raise SearchBackendUnavailable(f"Failed to ensure Elasticsearch index: {exc}") from exc

    def recreate_index(self) -> None:
        """Drop and recreate search index with current mapping."""
        client = self._get_client()
        try:
            client.indices.delete(index=self.index_name, ignore_unavailable=True)
            client.indices.create(
                index=self.index_name,
                settings=self._index_settings(),
                mappings=self._index_mappings(),
            )
            self._index_ready = True
        except Exception as exc:  # noqa: BLE001
            raise SearchBackendUnavailable(f"Failed to recreate Elasticsearch index: {exc}") from exc

    def index_sign(self, sign: Sign, *, refresh: bool = False) -> None:
        """Index one sign document."""
        self.ensure_index()
        client = self._get_client()
        try:
            client.index(
                index=self.index_name,
                id=sign.id,
                document=self._serialize_sign(sign),
                refresh=refresh,
            )
        except Exception as exc:  # noqa: BLE001
            raise SearchBackendUnavailable(f"Failed to index sign {sign.id}: {exc}") from exc

    def delete_sign(self, sign_id: str, *, refresh: bool = False) -> None:
        """Delete one sign document from index (idempotent)."""
        client = self._get_client()
        try:
            client.options(ignore_status=[404]).delete(index=self.index_name, id=sign_id, refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            raise SearchBackendUnavailable(f"Failed to delete sign {sign_id} from index: {exc}") from exc

    def bulk_upsert_signs(self, signs: list[Sign], *, refresh: bool = False) -> tuple[int, int]:
        """Bulk index sign documents and return (indexed, failed)."""
        if not signs:
            return 0, 0

        self.ensure_index()
        client = self._get_client()
        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": sign.id,
                "_source": self._serialize_sign(sign),
            }
            for sign in signs
        ]
        try:
            indexed, errors = bulk(
                client,
                actions,
                refresh=refresh,
                raise_on_error=False,
                raise_on_exception=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise SearchBackendUnavailable(f"Failed to bulk index signs: {exc}") from exc

        failed = len(errors) if isinstance(errors, list) else 0
        return int(indexed), int(failed)

    def search_sign_ids(
        self,
        *,
        query: str,
        category: str | None,
        tags: list[str],
        sort: str,
        page: int,
        per_page: int,
    ) -> tuple[list[str], int]:
        """Search sign ids with score-aware sorting and filters."""
        self.ensure_index()
        query_text = (query or "").strip()
        if not query_text:
            return [], 0

        client = self._get_client()
        try:
            response = client.search(
                index=self.index_name,
                query=self._build_sign_query(query_text, category=category, tags=tags),
                sort=self._build_sign_sort(sort),
                from_=(page - 1) * per_page,
                size=per_page,
                track_total_hits=True,
                track_scores=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise SearchBackendUnavailable(f"Failed to search signs in Elasticsearch: {exc}") from exc

        hits = response.get("hits", {}).get("hits", [])
        hit_ids = [str(hit.get("_id")) for hit in hits if hit.get("_id")]
        total = self._extract_total_hits(response)
        return hit_ids, total

    def search_dictionary(self, *, query: str, fields: str, limit: int = 100) -> list[dict]:
        """Search dictionary documents and return lightweight payload."""
        self.ensure_index()
        query_text = (query or "").strip()
        if not query_text:
            return []

        normalized_fields = fields if fields in _ALLOWED_FIELDS else "all"
        client = self._get_client()
        try:
            response = client.search(
                index=self.index_name,
                query=self._build_dictionary_query(query_text, fields=normalized_fields),
                sort=[{"_score": {"order": "desc"}}, {"name.keyword": {"order": "asc"}}],
                size=min(max(limit, 1), 100),
                track_total_hits=False,
                track_scores=True,
                source=["id", "name", "slug", "category", "tags"],
            )
        except Exception as exc:  # noqa: BLE001
            raise SearchBackendUnavailable(f"Failed to search dictionary in Elasticsearch: {exc}") from exc

        payload: list[dict] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source") or {}
            payload.append(
                {
                    "id": source.get("id") or hit.get("_id"),
                    "name": source.get("name"),
                    "slug": source.get("slug"),
                    "category": source.get("category"),
                    "tags": source.get("tags") or [],
                }
            )
        return payload

    def reindex_all(self, db: Session, *, batch_size: int = 500) -> dict:
        """Recreate index and bulk-sync all signs from database."""
        if not self.is_enabled():
            raise SearchBackendUnavailable("Search backend is not Elasticsearch")

        started_at = time.perf_counter()
        self.recreate_index()

        indexed = 0
        failed = 0
        offset = 0
        safe_batch_size = max(1, batch_size)

        while True:
            batch = db.scalars(select(Sign).order_by(Sign.id.asc()).offset(offset).limit(safe_batch_size)).all()
            if not batch:
                break
            batch_indexed, batch_failed = self.bulk_upsert_signs(batch, refresh=False)
            indexed += batch_indexed
            failed += batch_failed
            offset += len(batch)

        # Force index refresh once at the end so reindexed docs are immediately searchable.
        try:
            self._get_client().indices.refresh(index=self.index_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("elasticsearch_refresh_failed", extra={"error": str(exc)})

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "backend": "elasticsearch",
            "index": self.index_name,
            "indexed": indexed,
            "failed": failed,
            "duration_ms": duration_ms,
            "timestamp": timestamp,
        }

    def _get_client(self) -> Elasticsearch:
        if self._client is None:
            timeout_seconds = max(0.1, self._settings.elasticsearch_timeout_ms / 1000)
            self._client = Elasticsearch(
                hosts=[self._settings.elasticsearch_url],
                request_timeout=timeout_seconds,
                verify_certs=self._settings.elasticsearch_verify_certs,
            )
        return self._client

    @staticmethod
    def _serialize_sign(sign: Sign) -> dict:
        """Serialize SQLAlchemy sign model into Elasticsearch document."""
        created_at = sign.created_at.isoformat() if sign.created_at else None
        updated_at = sign.updated_at.isoformat() if sign.updated_at else None
        return {
            "id": sign.id,
            "name": sign.name,
            "slug": sign.slug,
            "description": sign.description,
            "category": sign.category,
            "tags": sign.tags or [],
            "variants": sign.variants or [],
            "notes": sign.notes,
            "usage_count": sign.usage_count,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    @staticmethod
    def _index_settings() -> dict:
        return {
            "analysis": {
                "analyzer": {
                    "folding": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            }
        }

    @staticmethod
    def _index_mappings() -> dict:
        return {
            "dynamic": False,
            "properties": {
                "id": {"type": "keyword"},
                "name": {
                    "type": "text",
                    "analyzer": "folding",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 256},
                        "prefix": {"type": "search_as_you_type", "analyzer": "folding"},
                    },
                },
                "slug": {
                    "type": "text",
                    "analyzer": "folding",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 256},
                    },
                },
                "description": {"type": "text", "analyzer": "folding"},
                "category": {
                    "type": "keyword",
                    "fields": {
                        "text": {"type": "text", "analyzer": "folding"},
                    },
                },
                "tags": {
                    "type": "keyword",
                    "fields": {
                        "text": {"type": "text", "analyzer": "folding"},
                    },
                },
                "variants": {"type": "text", "analyzer": "folding"},
                "notes": {"type": "text", "analyzer": "folding"},
                "usage_count": {"type": "integer"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            },
        }

    def _build_sign_query(self, query: str, *, category: str | None, tags: list[str]) -> dict:
        should_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "name^6",
                        "slug^4",
                        "description^2",
                        "notes",
                        "variants^1.5",
                        "tags.text^2",
                    ],
                    "fuzziness": "AUTO",
                }
            },
            {
                "multi_match": {
                    "query": query,
                    "type": "bool_prefix",
                    "fields": [
                        "name.prefix",
                        "name.prefix._2gram",
                        "name.prefix._3gram",
                    ],
                }
            },
        ]

        bool_query: dict[str, object] = {"should": should_clauses, "minimum_should_match": 1}
        filters = self._build_filters(category=category, tags=tags)
        if filters:
            bool_query["filter"] = filters
        return {"bool": bool_query}

    def _build_dictionary_query(self, query: str, *, fields: Literal["all", "name", "description", "tags"]) -> dict:
        if fields == "name":
            should_clauses = [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["name^7", "slug^4"],
                        "fuzziness": "AUTO",
                    }
                },
                {
                    "multi_match": {
                        "query": query,
                        "type": "bool_prefix",
                        "fields": ["name.prefix", "name.prefix._2gram", "name.prefix._3gram"],
                    }
                },
            ]
        elif fields == "description":
            should_clauses = [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["description^3", "notes^2"],
                        "fuzziness": "AUTO",
                    }
                }
            ]
        elif fields == "tags":
            should_clauses = [
                {"term": {"tags": query}},
                {"match": {"tags.text": {"query": query, "fuzziness": "AUTO"}}},
            ]
        else:
            should_clauses = [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["name^6", "slug^4", "description^2", "notes", "tags.text^2", "variants^1.5"],
                        "fuzziness": "AUTO",
                    }
                },
                {
                    "multi_match": {
                        "query": query,
                        "type": "bool_prefix",
                        "fields": ["name.prefix", "name.prefix._2gram", "name.prefix._3gram"],
                    }
                },
            ]

        return {"bool": {"should": should_clauses, "minimum_should_match": 1}}

    @staticmethod
    def _build_filters(*, category: str | None, tags: list[str]) -> list[dict]:
        filters: list[dict] = []
        if category:
            filters.append({"match": {"category.text": {"query": category, "operator": "and"}}})
        for tag in tags:
            filters.append({"match": {"tags.text": {"query": tag, "operator": "and"}}})
        return filters

    @staticmethod
    def _build_sign_sort(sort: str) -> list[dict]:
        if sort == "usage_count":
            return [
                {"usage_count": {"order": "desc"}},
                {"_score": {"order": "desc"}},
                {"name.keyword": {"order": "asc"}},
            ]
        if sort == "created_at":
            return [
                {"created_at": {"order": "desc"}},
                {"_score": {"order": "desc"}},
                {"name.keyword": {"order": "asc"}},
            ]
        return [
            {"_score": {"order": "desc"}},
            {"name.keyword": {"order": "asc"}},
        ]

    @staticmethod
    def _extract_total_hits(response: dict) -> int:
        total = response.get("hits", {}).get("total", 0)
        if isinstance(total, dict):
            return int(total.get("value", 0))
        return int(total or 0)


search_service = SearchService()

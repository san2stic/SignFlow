"""Singleton factory pour le backend de stockage actif."""

from __future__ import annotations

from functools import lru_cache

from .base import StorageBackend


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    """
    Retourne le backend de stockage selon la configuration.

    - USE_S3_STORAGE=true  → S3StorageBackend (MinIO/AWS)
    - USE_S3_STORAGE=false → LocalStorageBackend (filesystem dev)

    La valeur est mise en cache : un seul client boto3 partagé pour toute l'app.
    """
    from app.config import get_settings

    settings = get_settings()

    if settings.use_s3_storage:
        from .s3_backend import S3StorageBackend

        return S3StorageBackend(
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region=settings.s3_region,
            presigned_expiry=settings.s3_presigned_url_expiry,
        )

    from .local_backend import LocalStorageBackend

    return LocalStorageBackend(base_dir=settings.video_dir)

"""Backend de stockage local (filesystem) — utilisé en développement."""

from __future__ import annotations

import shutil
from pathlib import Path

import structlog

from .base import StorageBackend

logger = structlog.get_logger(__name__)


class LocalStorageBackend(StorageBackend):
    """
    Backend filesystem qui émule l'interface S3 sur le disque local.

    Utilisé en mode développement (USE_S3_STORAGE=false).
    Les objets sont stockés sous base_dir/{bucket}/{object_key}.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def _resolve(self, object_key: str, bucket: str) -> Path:
        return self.base_dir / bucket / object_key

    def upload_file(self, local_path: str | Path, object_key: str, bucket: str) -> str:
        dest = self._resolve(object_key, bucket)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_path), str(dest))
        logger.debug("local_storage.uploaded", key=object_key, bucket=bucket)
        return object_key

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        bucket: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        dest = self._resolve(object_key, bucket)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        logger.debug("local_storage.uploaded_bytes", key=object_key, bucket=bucket, size=len(data))
        return object_key

    def download_bytes(self, object_key: str, bucket: str) -> bytes:
        path = self._resolve(object_key, bucket)
        return path.read_bytes()

    def download_to_file(self, object_key: str, bucket: str, local_path: str | Path) -> None:
        src = self._resolve(object_key, bucket)
        shutil.copy2(str(src), str(local_path))

    def delete_object(self, object_key: str, bucket: str) -> None:
        path = self._resolve(object_key, bucket)
        if path.exists():
            path.unlink()
            logger.debug("local_storage.deleted", key=object_key, bucket=bucket)

    def get_presigned_url(self, object_key: str, bucket: str, expiry: int = 3600) -> str:
        # En local : retourner l'endpoint API interne de streaming
        # L'object_key est de la forme "videos/{type}/{uuid}.mp4"
        # On extrait l'UUID pour construire l'URL API
        parts = Path(object_key).stem.split("_")
        video_id = parts[0]
        return f"/api/v1/media/{video_id}/stream"

    def object_exists(self, object_key: str, bucket: str) -> bool:
        return self._resolve(object_key, bucket).exists()

"""Interface abstraite pour les backends de stockage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Interface commune pour le stockage local (dev) et S3/MinIO (prod)."""

    @abstractmethod
    def upload_file(self, local_path: str | Path, object_key: str, bucket: str) -> str:
        """Uploader un fichier local vers le backend. Retourne object_key."""

    @abstractmethod
    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        bucket: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Uploader des bytes en mémoire vers le backend. Retourne object_key."""

    @abstractmethod
    def download_bytes(self, object_key: str, bucket: str) -> bytes:
        """Télécharger un objet et retourner son contenu en bytes."""

    @abstractmethod
    def download_to_file(self, object_key: str, bucket: str, local_path: str | Path) -> None:
        """Télécharger un objet vers un fichier local."""

    @abstractmethod
    def delete_object(self, object_key: str, bucket: str) -> None:
        """Supprimer un objet. Silencieux si l'objet n'existe pas."""

    @abstractmethod
    def get_presigned_url(self, object_key: str, bucket: str, expiry: int = 3600) -> str:
        """Générer une URL temporaire d'accès direct à un objet."""

    @abstractmethod
    def object_exists(self, object_key: str, bucket: str) -> bool:
        """Vérifier si un objet existe."""

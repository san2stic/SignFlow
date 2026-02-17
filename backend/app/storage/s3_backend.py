"""Backend de stockage S3 compatible (MinIO auto-hébergé ou AWS S3)."""

from __future__ import annotations

from pathlib import Path

import structlog

from .base import StorageBackend

logger = structlog.get_logger(__name__)


class S3StorageBackend(StorageBackend):
    """
    Backend boto3 vers MinIO (prod) ou AWS S3.

    Fonctionnalités :
    - Signature S3v4 (requis par MinIO)
    - Retry adaptatif (3 tentatives)
    - Presigned URLs pour streaming direct sans proxy backend
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
        presigned_expiry: int = 3600,
    ) -> None:
        import boto3
        from botocore.config import Config

        self.presigned_expiry = presigned_expiry
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
        logger.info("s3_backend.initialized", endpoint=endpoint_url, region=region)

    def upload_file(self, local_path: str | Path, object_key: str, bucket: str) -> str:
        self._client.upload_file(str(local_path), bucket, object_key)
        logger.debug("s3.uploaded_file", key=object_key, bucket=bucket)
        return object_key

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        bucket: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        self._client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
        logger.debug("s3.uploaded_bytes", key=object_key, bucket=bucket, size=len(data))
        return object_key

    def stream_object(self, object_key: str, bucket: str, chunk_size: int = 1024 * 256):
        """Yield chunks from an S3 object for streaming responses."""
        resp = self._client.get_object(Bucket=bucket, Key=object_key)
        body = resp["Body"]
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def download_bytes(self, object_key: str, bucket: str) -> bytes:
        resp = self._client.get_object(Bucket=bucket, Key=object_key)
        data = resp["Body"].read()
        logger.debug("s3.downloaded_bytes", key=object_key, bucket=bucket, size=len(data))
        return data

    def download_to_file(self, object_key: str, bucket: str, local_path: str | Path) -> None:
        self._client.download_file(bucket, object_key, str(local_path))
        logger.debug("s3.downloaded_to_file", key=object_key, bucket=bucket, dest=str(local_path))

    def delete_object(self, object_key: str, bucket: str) -> None:
        from botocore.exceptions import ClientError

        try:
            self._client.delete_object(Bucket=bucket, Key=object_key)
            logger.debug("s3.deleted", key=object_key, bucket=bucket)
        except ClientError as exc:
            logger.warning("s3.delete_failed", key=object_key, bucket=bucket, error=str(exc))

    def get_presigned_url(self, object_key: str, bucket: str, expiry: int = 3600) -> str:
        url = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_key},
            ExpiresIn=expiry,
        )
        logger.debug("s3.presigned_url_generated", key=object_key, bucket=bucket, expiry=expiry)
        return url

    def object_exists(self, object_key: str, bucket: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=bucket, Key=object_key)
            return True
        except ClientError:
            return False

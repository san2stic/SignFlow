"""Client HTTP async pour TorchServe inference API."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class TorchServeClient:
    """Wrapper HTTP pour appeler TorchServe depuis FastAPI."""

    def __init__(
        self,
        base_url: str = "http://torchserve:8080",
        timeout_seconds: float = 2.0,
        max_retries: int = 1,
    ) -> None:
        """
        Initialize TorchServe client.

        Args:
            base_url: URL de l'API TorchServe (inference)
            timeout_seconds: Timeout HTTP en secondes
            max_retries: Nombre de retry en cas d'échec
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds)
        self.max_retries = max_retries
        self.session = httpx.AsyncClient(timeout=self.timeout)

    async def predict(
        self, window: np.ndarray, model_name: str = "signflow"
    ) -> tuple[str, float, list[dict[str, Any]]]:
        """
        Envoie une séquence enrichie à TorchServe pour inférence.

        Args:
            window: Séquence enrichie [64, 469]
            model_name: Nom du modèle TorchServe

        Returns:
            Tuple (prediction, confidence, alternatives)

        Raises:
            httpx.HTTPError: Si TorchServe injoignable ou erreur HTTP
        """
        payload = {"window": window.tolist()}
        url = f"{self.base_url}/predictions/{model_name}"

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

                result = response.json()
                prediction = result.get("prediction", "NONE")
                confidence = float(result.get("confidence", 0.0))
                alternatives = result.get("alternatives", [])

                logger.debug(
                    "torchserve_prediction_success",
                    prediction=prediction,
                    confidence=confidence,
                    latency_ms=response.elapsed.total_seconds() * 1000,
                )

                return prediction, confidence, alternatives

            except httpx.TimeoutException as e:
                logger.warning(
                    "torchserve_timeout",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(0.1 * (2**attempt))

            except httpx.HTTPStatusError as e:
                logger.error(
                    "torchserve_http_error",
                    status_code=e.response.status_code,
                    error=str(e),
                )
                raise

            except Exception as e:
                logger.error("torchserve_prediction_failed", error=str(e), exc_info=True)
                raise

        return "NONE", 0.0, []

    async def health_check(self) -> bool:
        """
        Vérifie si TorchServe est accessible.

        Returns:
            True si TorchServe répond, False sinon
        """
        try:
            response = await self.session.get(f"{self.base_url}/ping")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Ferme la session HTTP."""
        await self.session.aclose()

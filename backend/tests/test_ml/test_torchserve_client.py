"""Tests pour TorchServe client HTTP."""

import numpy as np
import pytest
import respx
from httpx import Response
import httpx

from app.ml.torchserve_client import TorchServeClient


@pytest.mark.asyncio
async def test_predict_success():
    """Test successful prediction via TorchServe."""
    client = TorchServeClient(base_url="http://test:8080")

    window = np.random.randn(64, 469).astype(np.float32)

    # Mock TorchServe response
    with respx.mock:
        respx.post("http://test:8080/predictions/signflow").mock(
            return_value=Response(
                200,
                json={
                    "prediction": "HELLO",
                    "confidence": 0.85,
                    "alternatives": [
                        {"sign": "HI", "confidence": 0.10}
                    ]
                }
            )
        )

        prediction, confidence, alternatives = await client.predict(window)

    assert prediction == "HELLO"
    assert confidence == 0.85
    assert len(alternatives) == 1
    assert alternatives[0]["sign"] == "HI"

    await client.close()


@pytest.mark.asyncio
async def test_predict_timeout_retry():
    """Test retry logic on timeout."""
    client = TorchServeClient(
        base_url="http://test:8080",
        timeout_seconds=0.1,
        max_retries=2
    )

    window = np.random.randn(64, 469).astype(np.float32)

    with respx.mock:
        # First 2 attempts timeout, 3rd succeeds
        route = respx.post("http://test:8080/predictions/signflow")
        route.side_effect = [
            httpx.TimeoutException("timeout"),
            httpx.TimeoutException("timeout"),
            Response(200, json={"prediction": "YES", "confidence": 0.9, "alternatives": []})
        ]

        prediction, confidence, _ = await client.predict(window)

    assert prediction == "YES"
    assert route.call_count == 3

    await client.close()


@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    client = TorchServeClient(base_url="http://test:8080")

    with respx.mock:
        respx.get("http://test:8080/ping").mock(return_value=Response(200))

        is_healthy = await client.health_check()

    assert is_healthy is True

    await client.close()

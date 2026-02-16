"""Tests for inference-time TTA generation."""

from __future__ import annotations

import numpy as np

from app.ml.tta import TTAConfig, TTAGenerator


def test_tta_generator_produces_configured_number_of_views() -> None:
    """Generator should return fixed-size list of same-shape views."""
    window = np.random.default_rng(42).standard_normal((64, 469)).astype(np.float32)
    generator = TTAGenerator(
        TTAConfig(
            num_views=5,
            enable_mirror=True,
            enable_temporal_jitter=True,
            enable_spatial_noise=True,
        )
    )

    views = generator.generate(window)

    assert len(views) == 5
    assert all(view.shape == window.shape for view in views)


def test_tta_mirror_swaps_hands_and_flips_x_axis() -> None:
    """Mirror view should swap left/right hand blocks and invert X coordinates."""
    window = np.zeros((8, 469), dtype=np.float32)
    window[:, 0] = 1.0       # left hand x
    window[:, 63] = 2.0      # right hand x
    window[:, 126] = 3.0     # pose x

    generator = TTAGenerator(TTAConfig(num_views=2, enable_mirror=True, enable_temporal_jitter=False, enable_spatial_noise=False))
    mirrored = generator._mirror_view(window)

    assert np.allclose(mirrored[:, 0], -2.0)
    assert np.allclose(mirrored[:, 63], -1.0)
    assert np.allclose(mirrored[:, 126], -3.0)

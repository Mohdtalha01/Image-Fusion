"""Unit tests for color conversions and chroma preservation."""

import numpy as np
import pytest
from fusion.fuse import (
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    rgb_to_hsv,
    hsv_to_rgb,
    extract_luminance_and_chroma,
)


def test_ycbcr_round_trip():
    """Verify RGB -> YCbCr -> RGB round-trip preserves pixel values."""
    rng = np.random.RandomState(42)
    # Generate diverse RGB test patch
    rgb = rng.rand(64, 64, 3).astype(np.float32)

    y, cb, cr = rgb_to_ycbcr(rgb)
    assert y.shape == (64, 64)
    assert cb.shape == (64, 64)
    assert cr.shape == (64, 64)

    reconstructed_rgb = ycbcr_to_rgb(y, cb, cr)
    # ITU-R BT.601 round-trip numerical accuracy
    np.testing.assert_allclose(
        reconstructed_rgb,
        rgb,
        atol=1e-4,
        err_msg="YCbCr round-trip exceeded numerical tolerance",
    )


def test_hsv_round_trip():
    """Verify RGB -> HSV -> RGB round-trip preserves pixel values."""
    # Test primary colors and ramps
    test_colors = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 1.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
    ], dtype=np.float32)

    h, s, v = rgb_to_hsv(test_colors)
    reconstructed_rgb = hsv_to_rgb(h, s, v)

    # 8-bit quantization tolerance for OpenCV HSV conversion (~ 2/255)
    np.testing.assert_allclose(reconstructed_rgb, test_colors, atol=1.5 / 255.0)


def test_extract_luminance_and_chroma():
    """Verify luminance extraction handles 2D grayscale, 3D single-channel, and 3D RGB."""
    gray_2d = np.ones((30, 30), dtype=np.float32) * 0.5
    lum, chroma, is_col = extract_luminance_and_chroma(gray_2d)
    assert not is_col
    assert chroma is None
    assert lum.shape == (30, 30)

    rgb = np.zeros((30, 30, 3), dtype=np.float32)
    rgb[..., 0] = 1.0  # Pure red
    lum, chroma, is_col = extract_luminance_and_chroma(rgb, color_space="YCbCr")
    assert is_col
    assert chroma is not None
    assert len(chroma) == 2
    # For pure red (1, 0, 0): Y = 0.299
    np.testing.assert_allclose(lum[0, 0], 0.299, atol=1e-3)

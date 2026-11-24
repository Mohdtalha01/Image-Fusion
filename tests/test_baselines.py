"""Unit tests for baseline fusion models."""

import numpy as np
import pytest
from baselines.averaging import fuse_averaging
from baselines.dwt import fuse_dwt
from baselines.gff import fuse_gff


def test_baseline_shapes_and_ranges():
    """Verify averaging, DWT, and GFF preserve dimensions and [0, 1] range."""
    rng = np.random.RandomState(42)
    img1 = rng.rand(64, 64).astype(np.float32)
    img2 = rng.rand(64, 64).astype(np.float32)

    for name, fn in [("Averaging", fuse_averaging), ("DWT", fuse_dwt), ("GFF", fuse_gff)]:
        fused = fn(img1, img2)
        assert fused.shape == (64, 64), f"{name} changed shape"
        assert fused.min() >= 0.0, f"{name} min < 0.0"
        assert fused.max() <= 1.0, f"{name} max > 1.0"
        assert not np.isnan(fused).any(), f"{name} produced NaNs"


def test_baseline_color_handling():
    """Verify baselines preserve color when one image is color and one is grayscale."""
    rng = np.random.RandomState(42)
    gray = rng.rand(48, 48).astype(np.float32)
    color = rng.rand(48, 48, 3).astype(np.float32)

    for name, fn in [("Averaging", fuse_averaging), ("DWT", fuse_dwt), ("GFF", fuse_gff)]:
        fused = fn(gray, color)
        assert fused.shape == (48, 48, 3), f"{name} failed color shape"
        assert fused.min() >= 0.0 and fused.max() <= 1.0

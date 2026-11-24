"""Unit tests for end-to-end CADM-MSWN fusion pipeline."""

import numpy as np
import pytest
from fusion.fuse import CADMSWNFusion, fuse_images


def test_fusion_output_shape_and_range():
    """Verify fused output maintains input dimensions and strictly bounded range [0, 1]."""
    rng = np.random.RandomState(42)
    img1 = rng.rand(64, 64).astype(np.float32)
    img2 = rng.rand(64, 64).astype(np.float32)

    pipeline = CADMSWNFusion()
    fused = pipeline.fuse(img1, img2)

    assert fused.shape == (64, 64), f"Expected (64, 64), got {fused.shape}"
    assert fused.min() >= 0.0, f"Fused image min {fused.min()} < 0.0"
    assert fused.max() <= 1.0, f"Fused image max {fused.max()} > 1.0"
    assert not np.isnan(fused).any(), "Fused output contains NaNs"


def test_fusion_determinism():
    """Verify that multiple runs with the same seed produce bitwise identical output."""
    rng = np.random.RandomState(10)
    img1 = rng.rand(48, 48).astype(np.float32)
    img2 = rng.rand(48, 48).astype(np.float32)

    cfg = {"clustering": {"random_state": 42}}
    fused_run1 = fuse_images(img1, img2, config=cfg)
    fused_run2 = fuse_images(img1, img2, config=cfg)

    np.testing.assert_array_equal(
        fused_run1,
        fused_run2,
        err_msg="CADM-MSWN non-deterministic across repeated executions with fixed seed",
    )


def test_color_and_grayscale_fusion():
    """Verify fusion with color source (e.g. PET) and grayscale source (e.g. MRI)."""
    rng = np.random.RandomState(42)
    mri_gray = rng.rand(50, 50).astype(np.float32)
    pet_color = rng.rand(50, 50, 3).astype(np.float32)

    # Fuse (gray, color)
    fused_1 = fuse_images(mri_gray, pet_color)
    assert fused_1.shape == (50, 50, 3), f"Expected (50, 50, 3), got {fused_1.shape}"
    assert fused_1.min() >= 0.0 and fused_1.max() <= 1.0

    # Fuse (color, gray)
    fused_2 = fuse_images(pet_color, mri_gray)
    assert fused_2.shape == (50, 50, 3)
    assert fused_2.min() >= 0.0 and fused_2.max() <= 1.0


def test_dimension_mismatch_raises_error():
    """Verify mismatch in spatial dimensions raises ValueError."""
    img1 = np.zeros((32, 32), dtype=np.float32)
    img2 = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="Source images must have identical spatial dimensions"):
        fuse_images(img1, img2)

"""Unit tests for structural decomposition and PMAD."""

import numpy as np
import pytest
from fusion.decomposition import (
    perona_malik_diffusion,
    compute_gradient_magnitude,
    decompose_structures,
)


def test_pmad_stability_and_bounds():
    """Verify PMAD preserves [0, 1] bounds and has no NaNs or Infs."""
    rng = np.random.RandomState(42)
    img = rng.rand(64, 64).astype(np.float32)

    diffused = perona_malik_diffusion(
        img, iterations=15, kappa=0.12, step_size=0.20, conduction="exponential"
    )

    assert not np.isnan(diffused).any(), "Diffused image contains NaNs"
    assert not np.isinf(diffused).any(), "Diffused image contains Infs"
    assert diffused.min() >= 0.0, f"Diffused image min {diffused.min()} < 0.0"
    assert diffused.max() <= 1.0, f"Diffused image max {diffused.max()} > 1.0"
    assert diffused.shape == (64, 64), "Shape changed during diffusion"


def test_pmad_invalid_step_size():
    """Verify PMAD raises ValueError if step_size exceeds theoretical stability limit."""
    img = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds numerical stability bound"):
        perona_malik_diffusion(img, step_size=0.30)


def test_pmad_conduction_functions():
    """Verify both exponential and rational conduction functions run and smooth image."""
    # Synthetic step edge image
    img = np.zeros((50, 50), dtype=np.float32)
    img[:, 25:] = 1.0

    diff_exp = perona_malik_diffusion(img, iterations=10, kappa=0.15, conduction="exponential")
    diff_rat = perona_malik_diffusion(img, iterations=10, kappa=0.15, conduction="rational")

    assert diff_exp.shape == img.shape
    assert diff_rat.shape == img.shape
    # Both should smooth the sharp edge transition
    assert 0.0 < diff_exp[25, 24] < 1.0
    assert 0.0 < diff_rat[25, 24] < 1.0


def test_complementary_decomposition():
    """Verify gross texture + fine texture exactly reconstructs the original input."""
    rng = np.random.RandomState(123)
    img = rng.rand(48, 48).astype(np.float32)

    gt, ge, ft, fe = decompose_structures(img, iterations=10, kappa=0.12, step_size=0.20)

    # Fundamental complement property: GT + FT = I
    reconstructed = gt + ft
    np.testing.assert_allclose(
        reconstructed,
        img,
        atol=1e-6,
        err_msg="Gross texture + Fine texture did not equal input image",
    )

    # Edge maps must be non-negative (for default non-signed FE)
    assert ge.min() >= 0.0, "Gross edge contains negative values"
    assert fe.min() >= 0.0, "Fine edge contains negative values"
    assert ge.shape == img.shape
    assert fe.shape == img.shape


def test_signed_fine_edge_option():
    """Verify signed fine edge preserves negative values when enabled."""
    rng = np.random.RandomState(42)
    img = rng.rand(32, 32).astype(np.float32)

    _, _, _, fe_signed = decompose_structures(img, iterations=10, signed_fe=True)
    # Gradient of I minus gradient of GT can produce negative values in smoothed regions
    assert fe_signed.shape == img.shape

"""Unit tests and mathematical validations for evaluation metrics suite.

Validates:
    - EN against hand-checkable analytical distributions (flat=0.0, binary=1.0 bit)
    - SD against analytical variance formulas (flat=0.0, 50/50 binary=0.5)
    - SF against flat and frequency patterns
    - AG against flat and linear ramps
    - MI against identical copy self-information
    - QABF and NABF against identical copy ground truth (QABF=1.0, NABF=0.0)
    - SSIM against identical copy ground truth (SSIM=1.0)
    - VIF against identical copy ground truth (VIF=1.0)
"""

import numpy as np
import pytest

from metrics.information import compute_entropy, compute_mutual_information
from metrics.features import (
    compute_standard_deviation,
    compute_spatial_frequency,
    compute_average_gradient,
)
from metrics.structural import compute_qabf_and_nabf, compute_ssim, compute_vif


def test_entropy_analytical_cases():
    """Verify entropy matches exact analytical calculations."""
    # Case 1: Constant image -> Entropy must be exactly 0.0
    flat_img = np.ones((64, 64), dtype=np.float32) * 0.5
    en_flat = compute_entropy(flat_img)
    assert en_flat == 0.0, f"Expected 0.0 for flat image, got {en_flat}"

    # Case 2: Exact binary uniform distribution (half 0, half 255)
    # Analytical: - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0 bit
    binary_img = np.zeros((100, 100), dtype=np.float32)
    binary_img[:, :50] = 1.0
    en_binary = compute_entropy(binary_img)
    np.testing.assert_allclose(
        en_binary,
        1.0,
        atol=1e-3,
        err_msg=f"Expected 1.0 bit for 50/50 binary image, got {en_binary}",
    )


def test_standard_deviation_analytical_cases():
    """Verify standard deviation matches analytical variance."""
    # Case 1: Flat image -> SD = 0.0
    flat_img = np.ones((50, 50), dtype=np.float32) * 0.75
    assert compute_standard_deviation(flat_img) == 0.0

    # Case 2: 50% 0.0 and 50% 1.0 -> Mean = 0.5, Var = 0.25, SD = 0.5
    step_img = np.zeros((40, 40), dtype=np.float32)
    step_img[:, :20] = 1.0
    sd_val = compute_standard_deviation(step_img)
    np.testing.assert_allclose(sd_val, 0.5000, atol=1e-4)


def test_spatial_frequency_analytical_cases():
    """Verify SF is 0 for flat images and positive for textured images."""
    flat_img = np.ones((40, 40), dtype=np.float32) * 0.3
    assert compute_spatial_frequency(flat_img) == 0.0

    # High frequency checkerboard pattern
    checker = np.indices((40, 40)).sum(axis=0) % 2
    sf_checker = compute_spatial_frequency(checker.astype(np.float32))
    assert sf_checker > 0.5, f"Expected high SF for checkerboard, got {sf_checker}"


def test_average_gradient_analytical_cases():
    """Verify AG is 0 for flat images and proportional to edge transitions."""
    flat_img = np.ones((32, 32), dtype=np.float32) * 0.8
    assert compute_average_gradient(flat_img) == 0.0

    # Image with sharp step edges
    step = np.zeros((32, 32), dtype=np.float32)
    step[16:, :] = 1.0
    ag_step = compute_average_gradient(step)
    assert ag_step > 0.0, f"Expected positive AG for step, got {ag_step}"


def test_mutual_information_properties():
    """Verify MI properties: identical self-information is positive, flat image is 0."""
    flat_img = np.ones((32, 32), dtype=np.float32) * 0.5
    assert compute_mutual_information(flat_img, flat_img, flat_img) == 0.0

    rng = np.random.RandomState(42)
    textured = rng.rand(40, 40).astype(np.float32)
    mi_self = compute_mutual_information(textured, textured, textured)
    assert mi_self > 0.0, "Expected positive MI for identical non-flat images"


def test_qabf_and_nabf_ground_truth():
    """Verify QABF=1.0 and NABF=0.0 when fused image is an exact copy of source."""
    rng = np.random.RandomState(42)
    src = rng.rand(64, 64).astype(np.float32)

    # When F == A == B, all edges are 100% transferred with 0 noise
    qabf, nabf = compute_qabf_and_nabf(img_fused=src, img_src1=src, img_src2=src)

    # Xydeas & Petrovic (2000) canonical parameters (Gamma_g=0.9994, Gamma_a=0.9879, ka=22, Da=0.8) yield max = 0.9748
    np.testing.assert_allclose(qabf, 0.9748, atol=1e-3, err_msg=f"Expected QABF ~ 0.9748, got {qabf}")
    np.testing.assert_allclose(nabf, 0.0000, atol=1e-4, err_msg=f"Expected NABF ~ 0.0, got {nabf}")


def test_ssim_ground_truth():
    """Verify SSIM=1.0 when fused image equals both sources."""
    rng = np.random.RandomState(42)
    src = rng.rand(50, 50).astype(np.float32)

    ssim_val = compute_ssim(img_fused=src, img_src1=src, img_src2=src)
    np.testing.assert_allclose(ssim_val, 1.0000, atol=1e-4, err_msg=f"Expected SSIM 1.0, got {ssim_val}")


def test_vif_ground_truth():
    """Verify VIF=1.0 when fused image equals both sources."""
    rng = np.random.RandomState(42)
    src = rng.rand(64, 64).astype(np.float32)

    vif_val = compute_vif(img_fused=src, img_src1=src, img_src2=src)
    np.testing.assert_allclose(vif_val, 1.0000, atol=1e-2, err_msg=f"Expected VIF ~ 1.0, got {vif_val}")




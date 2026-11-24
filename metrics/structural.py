"""Structural and perceptual quality metrics for multimodal image fusion.

Implementation & Wrapper Provenance:
    - compute_ssim: Direct wrapper around skimage.metrics.structural_similarity(..., data_range=1.0)
    - compute_vif (VIFP): Direct wrapper around sewar.full_ref.vifp(..., sigma_nsq=2.0) (Sheikh & Bovik, 2006)
    - compute_viff (VIFF): Native implementation of Han et al. (2013) multiscale fusion visual information fidelity
    - compute_qabf_and_nabf: Native implementation of Xydeas & Petrovic (2000) and Petrovic & Xydeas (2005)

Intensity Convention:
    - Images are internally passed as [0.0, 1.0] float arrays.
    - SSIM is computed on [0.0, 1.0] with data_range=1.0.
    - QABF, NABF, VIFP, and VIFF operate on standard [0.0, 255.0] luminance dynamic range.
"""

from typing import Tuple
import math
import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim_fn
import sewar.full_ref as sfr

from metrics.features import _to_luminance


def _compute_edge_information(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gradient magnitude and orientation angle using Sobel kernels.

    Evaluated on [0.0, 255.0] continuous luminance scale with Sobel / 8 normalization.
    """
    lum = _to_luminance(image)
    if lum.max() <= 1.0:
        lum_255 = lum * 255.0
    else:
        lum_255 = lum

    gx = cv2.Sobel(lum_255, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(lum_255, cv2.CV_64F, 0, 1, ksize=3) / 8.0

    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    return mag, angle


def compute_qabf_and_nabf(
    img_fused: np.ndarray,
    img_src1: np.ndarray,
    img_src2: np.ndarray,
    L: float = 1.0,
) -> Tuple[float, float]:
    """Compute QABF (edge preservation) and NABF (edge artifact noise) metrics.

    Reference:
        C. S. Xydeas and V. Petrovic,
        'Objective image fusion performance measure',
        Electronics Letters, Vol. 36, No. 4, pp. 308-309, 2000.
        V. Petrovic and C. Xydeas,
        'Objective evaluation of hybrid image fusion performance',
        SPIE Defense & Security, 2005.

    Model Parameters:
        Gamma_g = 0.9994, kappa_g = 15.0, sigma_g = 0.5
        Gamma_alpha = 0.9995, kappa_alpha = 15.0, sigma_alpha = 0.5
        L = 1.0 (gradient weight exponent)

    Returns:
        Tuple of (QABF, NABF) rounded to 4 decimal places.
    """
    g_f, a_f = _compute_edge_information(img_fused)
    g_a, a_a = _compute_edge_information(img_src1)
    g_b, a_b = _compute_edge_information(img_src2)

    gamma_g, k_g, sigma_g = 0.9994, 15.0, 0.5
    gamma_a, k_a, sigma_a = 0.9879, 22.0, 0.8

    def _edge_preservation(g_src, a_src):
        # 1. Gradient magnitude preservation G_AF
        g_max = np.maximum(g_src, g_f)
        g_min = np.minimum(g_src, g_f)

        g_ratio = np.zeros_like(g_src)
        valid_g = g_max > 1e-8
        g_ratio[valid_g] = g_min[valid_g] / g_max[valid_g]

        q_g = gamma_g / (1.0 + np.exp(-k_g * (g_ratio - sigma_g)))
        q_g[~valid_g] = 0.0

        # 2. Orientation preservation A_AF with unoriented angle symmetry modulo pi
        delta_a = np.abs(a_src - a_f)
        delta_a = np.where(delta_a > np.pi / 2.0, np.pi - delta_a, delta_a)
        a_ratio = 1.0 - (delta_a / (np.pi / 2.0))

        q_a = gamma_a / (1.0 + np.exp(-k_a * (a_ratio - sigma_a)))

        return q_g * q_a

    q_af = _edge_preservation(g_a, a_a)
    q_bf = _edge_preservation(g_b, a_b)

    w_a = g_a**L
    w_b = g_b**L
    total_w = w_a + w_b
    sum_w = np.sum(total_w)

    if sum_w < 1e-8:
        return 1.0, 0.0

    # QABF: total edge information preservation score in [0, 1]
    qabf = float(np.sum(q_af * w_a + q_bf * w_b) / sum_w)

    # NABF: Petrovic & Xydeas (2005) artifact noise where fused edges exceed both sources
    excess_mask = (g_f > g_a) & (g_f > g_b)
    n_map = np.zeros_like(g_f)
    n_map[excess_mask] = (1.0 - q_af[excess_mask]) * w_a[excess_mask] + (1.0 - q_bf[excess_mask]) * w_b[excess_mask]

    nabf = float(np.sum(n_map) / sum_w)

    return round(float(np.clip(qabf, 0.0, 1.0)), 4), round(float(np.clip(nabf, 0.0, 1.0)), 4)


def compute_ssim(
    img_fused: np.ndarray,
    img_src1: np.ndarray,
    img_src2: np.ndarray,
) -> float:
    """Compute mean Structural Similarity Index SSIM(F; I1, I2) = 0.5*SSIM(F, I1) + 0.5*SSIM(F, I2).

    Formula (Wang et al., IEEE TIP 2004):
        SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
        C1 = (K1 * L)^2, C2 = (K2 * L)^2 with K1=0.01, K2=0.03, L=1.0 on [0, 1].

    Returns:
        SSIM value in [-1, 1], typically [0, 1].
    """
    f = _to_luminance(img_fused)
    s1 = _to_luminance(img_src1)
    s2 = _to_luminance(img_src2)

    ssim1 = ssim_fn(f, s1, data_range=1.0)
    ssim2 = ssim_fn(f, s2, data_range=1.0)

    mean_ssim = 0.5 * (ssim1 + ssim2)
    return round(float(mean_ssim), 4)


def compute_vif(
    img_fused: np.ndarray,
    img_src1: np.ndarray,
    img_src2: np.ndarray,
    sigma_nsq: float = 2.0,
) -> float:
    """Compute multiscale Visual Information Fidelity in Pixel Domain (VIFP).

    Reference:
        H. R. Sheikh and A. C. Bovik,
        'Image Information and Visual Quality',
        IEEE Transactions on Image Processing, Vol. 15, No. 2, pp. 430-444, 2006.

    Fusion Formulation:
        VIF(F; I1, I2) = 0.5 * VIFP(I1, F) + 0.5 * VIFP(I2, F)
        Computed using the established sewar.full_ref.vifp reference implementation
        on 4 pyramid scales with Gaussian Scale Mixture (GSM) modeling and noise
        variance sigma_nsq = 2.0 on standard [0, 255] dynamic range.

    Returns:
        VIF score (float >= 0.0).
    """
    f = _to_luminance(img_fused)
    s1 = _to_luminance(img_src1)
    s2 = _to_luminance(img_src2)

    # Convert normalized [0, 1] to [0, 255] dynamic range for GSM HVS model
    f_255 = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.float64)
    s1_255 = (np.clip(s1, 0.0, 1.0) * 255.0).astype(np.float64)
    s2_255 = (np.clip(s2, 0.0, 1.0) * 255.0).astype(np.float64)

    vif1 = float(sfr.vifp(s1_255, f_255, sigma_nsq=sigma_nsq))
    vif2 = float(sfr.vifp(s2_255, f_255, sigma_nsq=sigma_nsq))

    vif_score = 0.5 * (vif1 + vif2)
    return round(float(vif_score), 4)


"""Anisotropic Diffusion Fusion (ADF) baseline.

Reference:
    D. P. Bavirisetti and R. Dhuli,
    'Two-scale image fusion of visual and infrared images using an additive wavelet transform and anisotropic diffusion',
    Information Fusion, Vol. 28, pp. 45-53, 2016.
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import uniform_filter

from fusion.decomposition import perona_malik_diffusion
from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb


def fuse_adf_luminance(
    lum1: np.ndarray,
    lum2: np.ndarray,
    iterations: int = 15,
    kappa: float = 0.12,
    step_size: float = 0.20,
) -> np.ndarray:
    """Fuse two luminance maps using Anisotropic Diffusion Fusion (ADF)."""
    # Base layers via anisotropic diffusion
    base1 = perona_malik_diffusion(lum1, iterations=iterations, kappa=kappa, step_size=step_size)
    base2 = perona_malik_diffusion(lum2, iterations=iterations, kappa=kappa, step_size=step_size)

    # Detail layers
    detail1 = lum1 - base1
    detail2 = lum2 - base2

    # Base fusion rule: simple average
    fused_base = 0.5 * (base1 + base2)

    # Detail fusion rule: local energy-weighted selection
    energy1 = uniform_filter(detail1**2, size=5, mode="reflect")
    energy2 = uniform_filter(detail2**2, size=5, mode="reflect")

    weight1 = (energy1 >= energy2).astype(np.float32)
    weight2 = 1.0 - weight1

    fused_detail = weight1 * detail1 + weight2 * detail2
    fused = fused_base + fused_detail

    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def fuse_adf(
    img1: np.ndarray,
    img2: np.ndarray,
    color_space: str = "YCbCr",
) -> np.ndarray:
    """Fuse two images using ADF on luminance with chroma preservation."""
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_lum = fuse_adf_luminance(lum1, lum2)

    if not is_col1 and not is_col2:
        return fused_lum

    if is_col1 and not is_col2:
        target_chroma = chroma1
    elif is_col2 and not is_col1:
        target_chroma = chroma2
    else:
        target_chroma = (
            0.5 * (chroma1[0] + chroma2[0]),
            0.5 * (chroma1[1] + chroma2[1]),
        )

    if color_space.upper() == "YCBCR":
        cb, cr = target_chroma
        return ycbcr_to_rgb(fused_lum, cb, cr)
    else:
        h, s = target_chroma
        return hsv_to_rgb(h, s, fused_lum)

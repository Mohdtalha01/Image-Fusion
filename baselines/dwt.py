"""Discrete Wavelet Transform (DWT) baseline for multimodal image fusion.

Uses PyWavelets with Daubechies (db2) wavelets.
Fusion rule:
    - Low-frequency approximation: average
    - High-frequency detail subbands (LH, HL, HH): choose-max absolute value
"""

from typing import Tuple, Optional
import numpy as np
import pywt
from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb


def fuse_dwt_luminance(
    lum1: np.ndarray,
    lum2: np.ndarray,
    wavelet: str = "db2",
    level: int = 2,
) -> np.ndarray:
    """Fuse two 2D luminance maps via multiscale DWT.

    Args:
        lum1: First luminance map in [0, 1].
        lum2: Second luminance map in [0, 1].
        wavelet: Wavelet family name (default: 'db2').
        level: Decomposition level (default: 2).

    Returns:
        Fused luminance map in [0, 1].
    """
    coeffs1 = pywt.wavedec2(lum1, wavelet=wavelet, level=level)
    coeffs2 = pywt.wavedec2(lum2, wavelet=wavelet, level=level)

    # Low frequency approximation: average
    fused_approx = 0.5 * (coeffs1[0] + coeffs2[0])

    # High frequency details: max-absolute selection
    fused_details = []
    for d1, d2 in zip(coeffs1[1:], coeffs2[1:]):
        # Each level contains (LH, HL, HH)
        fused_subbands = []
        for s1, s2 in zip(d1, d2):
            mask = np.abs(s1) >= np.abs(s2)
            fused_s = np.where(mask, s1, s2)
            fused_subbands.append(fused_s)
        fused_details.append(tuple(fused_subbands))

    fused_coeffs = [fused_approx] + fused_details
    reconstructed = pywt.waverec2(fused_coeffs, wavelet=wavelet)

    # Trim to original dimensions if padding occurred
    h, w = lum1.shape
    reconstructed = reconstructed[:h, :w]
    return np.clip(reconstructed, 0.0, 1.0).astype(np.float32)


def fuse_dwt(
    img1: np.ndarray,
    img2: np.ndarray,
    wavelet: str = "db2",
    level: int = 2,
    color_space: str = "YCbCr",
) -> np.ndarray:
    """Fuse two images using DWT on luminance with chroma preservation."""
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_lum = fuse_dwt_luminance(lum1, lum2, wavelet=wavelet, level=level)

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

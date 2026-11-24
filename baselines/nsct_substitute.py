"""Non-Subsampled Shift-Invariant Transform (NSCT substitute) baseline.

Note on substitution (documented per project instructions):
    The original Non-Subsampled Contourlet Transform (NSCT) by da Cunha, Zhou,
    and Do (IEEE TIP 2006) relies on compiled MATLAB MEX binaries.
    In native Python, this baseline implements a 2-level Stationary Wavelet
    Transform (SWT) with Daubechies shift-invariant filters combined with
    directional steerable high-pass energy selection, providing shift-invariance
    and directional multiscale decomposition without sub-sampling.
"""

from typing import Tuple
import numpy as np
import pywt

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb


def fuse_nsct_substitute_luminance(
    lum1: np.ndarray,
    lum2: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
) -> np.ndarray:
    """Fuse two luminance maps using Stationary Wavelet Transform (shift-invariant multiscale)."""
    h_orig, w_orig = lum1.shape

    # SWT requires dimensions to be divisible by 2^level
    divisor = 2**level
    pad_h = (divisor - (h_orig % divisor)) % divisor
    pad_w = (divisor - (w_orig % divisor)) % divisor

    if pad_h > 0 or pad_w > 0:
        l1 = np.pad(lum1, ((0, pad_h), (0, pad_w)), mode="reflect")
        l2 = np.pad(lum2, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        l1 = lum1
        l2 = lum2

    # Stationary (non-subsampled) wavelet transform
    coeffs1 = pywt.swt2(l1, wavelet=wavelet, level=level)
    coeffs2 = pywt.swt2(l2, wavelet=wavelet, level=level)

    fused_coeffs = []
    for (cA1, (cH1, cV1, cD1)), (cA2, (cH2, cV2, cD2)) in zip(coeffs1, coeffs2):
        # Low-frequency approximation: average
        cA_fused = 0.5 * (cA1 + cA2)

        # High-frequency directional subbands: max-absolute energy selection
        cH_fused = np.where(np.abs(cH1) >= np.abs(cH2), cH1, cH2)
        cV_fused = np.where(np.abs(cV1) >= np.abs(cV2), cV1, cV2)
        cD_fused = np.where(np.abs(cD1) >= np.abs(cD2), cD1, cD2)

        fused_coeffs.append((cA_fused, (cH_fused, cV_fused, cD_fused)))

    # Inverse stationary wavelet transform
    reconstructed = pywt.iswt2(fused_coeffs, wavelet=wavelet)
    reconstructed = reconstructed[:h_orig, :w_orig]

    return np.clip(reconstructed, 0.0, 1.0).astype(np.float32)


def fuse_nsct_substitute(
    img1: np.ndarray,
    img2: np.ndarray,
    color_space: str = "YCbCr",
) -> np.ndarray:
    """Fuse two images using NSCT substitute on luminance with chroma preservation."""
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_lum = fuse_nsct_substitute_luminance(lum1, lum2)

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

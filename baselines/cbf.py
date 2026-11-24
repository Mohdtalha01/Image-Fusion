"""Cross Bilateral Filter (CBF) baseline.

Reference:
    B. K. Shreyamsha Kumar,
    'Image fusion based on cross bilateral filter',
    IEEE Signal Processing Letters, Vol. 22, No. 7, pp. 886-890, 2015.
"""

from typing import Tuple
import numpy as np
import cv2
from scipy.ndimage import uniform_filter

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb


def fast_cross_bilateral_filter(
    image: np.ndarray,
    guide: np.ndarray,
    d: int = 7,
    sigma_color: float = 0.1,
    sigma_space: float = 3.0,
) -> np.ndarray:
    """Joint/cross bilateral filter using guide to filter image."""
    img_8u = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    guide_8u = (np.clip(guide, 0.0, 1.0) * 255.0).astype(np.uint8)

    # OpenCV provides ximgproc.jointBilateralFilter or guidedFilter as standard fast cross filter
    filtered = cv2.ximgproc.jointBilateralFilter(
        guide_8u, img_8u, d=d, sigmaColor=sigma_color * 255.0, sigmaSpace=sigma_space
    ) if hasattr(cv2, "ximgproc") else None

    if filtered is None:
        # Fallback to bilateral-guided convolution
        from fusion.weighting import fast_guided_filter
        filtered_f = fast_guided_filter(guide, image, radius=d // 2, eps=(sigma_color**2))
        return filtered_f

    return filtered.astype(np.float32) / 255.0


def fuse_cbf_luminance(
    lum1: np.ndarray,
    lum2: np.ndarray,
) -> np.ndarray:
    """Fuse two luminance maps using Cross Bilateral Filter (CBF)."""
    # Cross filtering: lum1 filtered by lum2, lum2 filtered by lum1
    cbf1 = fast_cross_bilateral_filter(lum1, guide=lum2)
    cbf2 = fast_cross_bilateral_filter(lum2, guide=lum1)

    # Detail maps
    detail1 = lum1 - cbf1
    detail2 = lum2 - cbf2

    # Saliency weights from local variance
    mean_d1 = uniform_filter(detail1, size=5, mode="reflect")
    var_d1 = uniform_filter(detail1**2, size=5, mode="reflect") - mean_d1**2

    mean_d2 = uniform_filter(detail2, size=5, mode="reflect")
    var_d2 = uniform_filter(detail2**2, size=5, mode="reflect") - mean_d2**2

    w1 = (var_d1 >= var_d2).astype(np.float32)
    w2 = 1.0 - w1

    fused_detail = w1 * detail1 + w2 * detail2
    fused_base = 0.5 * (cbf1 + cbf2)

    return np.clip(fused_base + fused_detail, 0.0, 1.0).astype(np.float32)


def fuse_cbf(
    img1: np.ndarray,
    img2: np.ndarray,
    color_space: str = "YCbCr",
) -> np.ndarray:
    """Fuse two images using CBF on luminance with chroma preservation."""
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_lum = fuse_cbf_luminance(lum1, lum2)

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

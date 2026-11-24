"""Guided Filtering Fusion (GFF) baseline.

Reference:
    Shutao Li, Xudong Kang, Jianwen Hu,
    'Image Fusion With Guided Filtering',
    IEEE Transactions on Image Processing, Vol. 22, No. 7, pp. 2864-2875, 2013.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import uniform_filter
import cv2

from fusion.weighting import fast_guided_filter
from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb


def fuse_gff_luminance(
    lum1: np.ndarray,
    lum2: np.ndarray,
    r1: int = 15,
    eps1: float = 0.3,
    r2: int = 5,
    eps2: float = 1e-6,
) -> np.ndarray:
    """Execute Guided Filtering Fusion on two 2D luminance maps.

    Args:
        lum1: First luminance array in [0, 1].
        lum2: Second luminance array in [0, 1].
        r1: Window radius for base layer guided filter.
        eps1: Epsilon for base layer guided filter.
        r2: Window radius for detail layer guided filter.
        eps2: Epsilon for detail layer guided filter.

    Returns:
        Fused luminance array in [0, 1].
    """
    # 1. Two-scale decomposition using average filter (size 31x31 default, radius 15)
    filter_size = 2 * r1 + 1
    base1 = uniform_filter(lum1.astype(np.float64), size=filter_size, mode="reflect")
    base2 = uniform_filter(lum2.astype(np.float64), size=filter_size, mode="reflect")

    detail1 = lum1 - base1
    detail2 = lum2 - base2

    # 2. Saliency detection using Laplacian filtering
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h1 = cv2.filter2D(lum1.astype(np.float64), -1, laplacian_kernel)
    h2 = cv2.filter2D(lum2.astype(np.float64), -1, laplacian_kernel)

    # Gaussian or box smoothing of absolute Laplacian response
    saliency1 = cv2.GaussianBlur(np.abs(h1), (filter_size, filter_size), 5.0)
    saliency2 = cv2.GaussianBlur(np.abs(h2), (filter_size, filter_size), 5.0)

    # 3. Initial binary decision map
    p1 = (saliency1 >= saliency2).astype(np.float64)
    p2 = 1.0 - p1

    # 4. Weight map construction via Guided Filtering
    # Base weights
    w_b1 = fast_guided_filter(guide=lum1, src=p1, radius=r1, eps=eps1)
    w_b2 = fast_guided_filter(guide=lum2, src=p2, radius=r1, eps=eps1)
    sum_wb = w_b1 + w_b2 + 1e-8
    w_b1 = np.maximum(0.0, w_b1) / sum_wb
    w_b2 = np.maximum(0.0, w_b2) / sum_wb

    # Detail weights
    w_d1 = fast_guided_filter(guide=lum1, src=p1, radius=r2, eps=eps2)
    w_d2 = fast_guided_filter(guide=lum2, src=p2, radius=r2, eps=eps2)
    sum_wd = w_d1 + w_d2 + 1e-8
    w_d1 = np.maximum(0.0, w_d1) / sum_wd
    w_d2 = np.maximum(0.0, w_d2) / sum_wd

    # 5. Two-scale combination
    fused_base = w_b1 * base1 + w_b2 * base2
    fused_detail = w_d1 * detail1 + w_d2 * detail2
    fused = fused_base + fused_detail

    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def fuse_gff(
    img1: np.ndarray,
    img2: np.ndarray,
    color_space: str = "YCbCr",
) -> np.ndarray:
    """Fuse two images using GFF on luminance with chroma preservation."""
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_lum = fuse_gff_luminance(lum1, lum2)

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

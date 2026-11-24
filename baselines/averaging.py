"""Pixel-wise averaging baseline for multimodal image fusion."""

from typing import Tuple, Optional
import numpy as np
from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb


def fuse_averaging(
    img1: np.ndarray,
    img2: np.ndarray,
    color_space: str = "YCbCr",
) -> np.ndarray:
    """Fuse two images using pixel-wise averaging on luminance.

    Args:
        img1: First input image.
        img2: Second input image.
        color_space: 'YCbCr' or 'HSV'.

    Returns:
        Fused image in [0, 1].
    """
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_lum = np.clip(0.5 * (lum1 + lum2), 0.0, 1.0)

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

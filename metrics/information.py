"""Information-theoretic evaluation metrics for multimodal image fusion.

Implements:
    - EN: Shannon Entropy (bits)
    - MI: Mutual Information between fused image and source images
"""

from typing import Tuple, Union
import numpy as np


def compute_entropy(image: np.ndarray, bins: int = 256) -> float:
    """Compute Shannon Entropy of an image in bits.

    Formula:
        EN = - sum_{i=0}^{L-1} p_i * log2(p_i)
    where p_i is the normalized probability distribution of pixel intensities.

    Args:
        image: 2D array of shape (H, W) in [0, 1] or [0, 255].
        bins: Number of histogram bins (default: 256).

    Returns:
        Entropy value in bits (float).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        # Convert to luminance
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    if img.max() <= 1.0:
        img_8u = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img_8u = np.clip(img, 0, 255).astype(np.uint8)

    hist, _ = np.histogram(img_8u, bins=bins, range=(0, 256), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0

    entropy = float(-np.sum(hist * np.log2(hist)))
    return round(entropy, 4)


def compute_mutual_information(
    img_fused: np.ndarray,
    img_src1: np.ndarray,
    img_src2: np.ndarray,
    bins: int = 256,
) -> float:
    """Compute Mutual Information MI(F; I1, I2) = MI(F, I1) + MI(F, I2).

    Formula:
        MI(F, A) = sum_{f, a} p(f, a) * log2( p(f, a) / (p(f) * p(a)) )

    Args:
        img_fused: Fused image.
        img_src1: First source image.
        img_src2: Second source image.
        bins: Histogram bins (default: 256).

    Returns:
        Total mutual information value (float).
    """
    def _to_luminance_8u(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim == 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        if arr.max() <= 1.0:
            return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

    f = _to_luminance_8u(img_fused).flatten()
    s1 = _to_luminance_8u(img_src1).flatten()
    s2 = _to_luminance_8u(img_src2).flatten()

    def _pairwise_mi(x: np.ndarray, y: np.ndarray) -> float:
        # 2D joint histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 256], [0, 256]])
        pxy = hist_2d / float(np.sum(hist_2d))

        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        px_py = px[:, None] * py[None, :]
        nz = pxy > 0

        mi = np.sum(pxy[nz] * np.log2(pxy[nz] / px_py[nz]))
        return float(mi)

    mi_f_s1 = _pairwise_mi(f, s1)
    mi_f_s2 = _pairwise_mi(f, s2)
    return round(float(mi_f_s1 + mi_f_s2), 4)

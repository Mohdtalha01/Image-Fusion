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
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr_f = np.asarray(arr, dtype=np.float64)
        arr = 0.299 * arr_f[..., 0] + 0.587 * arr_f[..., 1] + 0.114 * arr_f[..., 2]

    if arr.dtype == np.uint8:
        img_8u = arr
    else:
        arr_f = np.asarray(arr, dtype=np.float64)
        if arr_f.max() <= 1.0:
            img_8u = (np.clip(arr_f, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_8u = np.clip(arr_f, 0, 255).astype(np.uint8)

    if bins == 256:
        counts = np.bincount(img_8u.ravel(), minlength=256)
        nz = counts > 0
        if not np.any(nz):
            return 0.0
        p = counts[nz] / float(img_8u.size)
    else:
        p, _ = np.histogram(img_8u, bins=bins, range=(0, 256), density=True)
        p = p[p > 0]
        if len(p) == 0:
            return 0.0

    entropy = float(-np.sum(p * np.log2(p)))
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
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr_f = np.asarray(arr, dtype=np.float64)
            arr = 0.299 * arr_f[..., 0] + 0.587 * arr_f[..., 1] + 0.114 * arr_f[..., 2]
        if arr.dtype == np.uint8:
            return arr.ravel()
        arr_f = np.asarray(arr, dtype=np.float64)
        if arr_f.max() <= 1.0:
            return (np.clip(arr_f, 0.0, 1.0) * 255.0).astype(np.uint8).ravel()
        return np.clip(arr_f, 0, 255).astype(np.uint8).ravel()

    f = _to_luminance_8u(img_fused)
    s1 = _to_luminance_8u(img_src1)
    s2 = _to_luminance_8u(img_src2)

    if bins == 256:
        f_shifted = f.astype(np.int32) * 256
        s1_i = s1.astype(np.int32)
        s2_i = s2.astype(np.int32)

        def _pairwise_mi_256(x_shifted: np.ndarray, y_int: np.ndarray) -> float:
            hist_2d = np.bincount(x_shifted + y_int, minlength=65536).reshape(256, 256)
            pxy = hist_2d / float(y_int.size)
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            r, c = np.nonzero(pxy)
            vals = pxy[r, c]
            return float(np.sum(vals * np.log2(vals / (px[r] * py[c]))))

        return round(_pairwise_mi_256(f_shifted, s1_i) + _pairwise_mi_256(f_shifted, s2_i), 4)

    def _pairwise_mi(x: np.ndarray, y: np.ndarray) -> float:
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 256], [0, 256]])
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nz = pxy > 0
        return float(np.sum(pxy[nz] * np.log2(pxy[nz] / px_py[nz])))

    return round(float(_pairwise_mi(f, s1) + _pairwise_mi(f, s2)), 4)

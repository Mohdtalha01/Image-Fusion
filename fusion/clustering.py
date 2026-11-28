"""Clustering module for CADM-MSWN image fusion.

Constructs per-pixel normalized feature vectors (intensity, local mean,
local standard deviation, gradient magnitude) and partitions the pixel manifold
using K-Means or Fuzzy C-Means into hard masks satisfying the partition of unity.
Supports both per-source and joint clustering modes.
"""

from typing import Tuple, Literal
import numpy as np
import cv2
from sklearn.cluster import KMeans

from fusion.decomposition import compute_gradient_magnitude


def extract_pixel_features(
    image: np.ndarray,
    window_size: int = 7,
) -> np.ndarray:
    """Extract a 4-dimensional normalized feature vector per pixel.

    Features:
        1. Intensity: I(x, y)
        2. Local Mean: mu_w(x, y)
        3. Local Standard Deviation: sigma_w(x, y)
        4. Gradient Magnitude: |grad I(x, y)|

    Each feature is Min-Max normalized across the image to [0, 1].
    Vectorized in float32 using fast cv2.boxFilter.

    Args:
        image: 2D float array (H, W) in [0, 1].
        window_size: Size of local rectangular window w x w.

    Returns:
        Feature array of shape (H, W, 4) in [0, 1].
    """
    h, w = image.shape
    img_32 = np.asarray(image, dtype=np.float32)

    # 1. Intensity
    feat_i = img_32

    # 2. Local Mean (fast cv2.boxFilter)
    ksize = (window_size, window_size)
    feat_mean = cv2.boxFilter(img_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    # 3. Local Standard Deviation
    img_sq_mean = cv2.boxFilter(img_32 * img_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(0.0, img_sq_mean - feat_mean * feat_mean)
    feat_std = np.sqrt(variance)

    # 4. Gradient Magnitude
    feat_grad = compute_gradient_magnitude(img_32)

    out = np.empty((h, w, 4), dtype=np.float32)
    for idx, f in enumerate((feat_i, feat_mean, feat_std, feat_grad)):
        min_v = float(np.min(f))
        max_v = float(np.max(f))
        diff = max_v - min_v
        if diff > 1e-8:
            np.clip(
                (f - np.float32(min_v)) * np.float32(1.0 / diff),
                0.0,
                1.0,
                out=out[..., idx],
            )
        else:
            out[..., idx] = 0.0

    return out


def fuzzy_c_means(
    data: np.ndarray,
    n_clusters: int = 3,
    m: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    random_state: int = 42,
) -> np.ndarray:
    """Vectorized Fuzzy C-Means clustering.

    Args:
        data: Array of shape (N, D).
        n_clusters: Number of clusters K.
        m: Fuzziness exponent (m > 1).
        max_iter: Maximum iterations.
        tol: Convergence tolerance for membership matrix.
        random_state: Random seed for initialization.

    Returns:
        Hard cluster assignment labels of shape (N,).
    """
    rng = np.random.RandomState(random_state)
    n_samples, _ = data.shape
    data_64 = np.asarray(data, dtype=np.float64)
    data_norm_sq = np.sum(data_64 * data_64, axis=1)  # (N,)
    inv_m_minus_1 = -1.0 / (m - 1.0)

    # Initialize random membership matrix U
    u = rng.rand(n_clusters, n_samples)
    u /= np.sum(u, axis=0, keepdims=True)

    for _ in range(max_iter):
        u_prev = u
        um = u**m

        # Compute cluster centers: shape (K, D)
        centers = (um @ data_64) / np.sum(um, axis=1, keepdims=True)
        centers_norm_sq = np.sum(centers * centers, axis=1)[:, None]  # (K, 1)

        # Vectorized squared distances ||x_j - c_i||^2 = ||x_j||^2 + ||c_i||^2 - 2 c_i . x_j
        dist_sq = centers_norm_sq + data_norm_sq[None, :] - 2.0 * (centers @ data_64.T)
        np.maximum(dist_sq, 1e-20, out=dist_sq)

        # Update membership matrix U using dist_sq^(-1/(m-1)) == dist^(-2/(m-1))
        inv_dist = dist_sq**inv_m_minus_1
        u = inv_dist / np.sum(inv_dist, axis=0, keepdims=True)

        if np.max(np.abs(u - u_prev)) < tol:
            break

    # Return hard cluster assignments
    return np.argmax(u, axis=0)


def generate_cluster_masks(
    features: np.ndarray,
    n_clusters: int = 3,
    method: Literal["kmeans", "fcm"] = "kmeans",
    random_state: int = 42,
) -> np.ndarray:
    """Partition feature space into K hard cluster masks.

    Ensures the partition of unity:
        sum_{k=0}^{K-1} M_k(x, y) = 1  for all (x, y).

    Args:
        features: Array of shape (H, W, D).
        n_clusters: Number of clusters K.
        method: 'kmeans' or 'fcm'.
        random_state: Seed for reproducibility.

    Returns:
        Binary masks array of shape (K, H, W) of type float32.
    """
    if n_clusters == 1:
        h, w, _ = features.shape
        return np.ones((1, h, w), dtype=np.float32)

    h, w, d = features.shape
    flat_data = features.reshape(-1, d)

    if method == "kmeans":
        cv2.setRNGSeed(random_state)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels_cv, _ = cv2.kmeans(
            flat_data.astype(np.float32),
            n_clusters,
            None,
            criteria,
            1,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels_cv.flatten()
    elif method == "fcm":
        labels = fuzzy_c_means(
            flat_data,
            n_clusters=n_clusters,
            m=2.0,
            max_iter=100,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Build hard one-hot masks in one vectorized scatter
    masks = np.zeros((n_clusters, h * w), dtype=np.float32)
    masks[labels, np.arange(h * w)] = 1.0

    return masks.reshape(n_clusters, h, w)


def cluster_source_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    n_clusters: int = 3,
    method: Literal["kmeans", "fcm"] = "kmeans",
    mode: Literal["per_source", "joint"] = "per_source",
    window_size: int = 7,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform clustering on a pair of source images.

    In 'per_source' mode:
        Computes separate cluster masks M1 and M2 for img1 and img2.
    In 'joint' mode:
        Concatenates pixel feature vectors from both images and clusters jointly,
        returning identical partition masks (M_joint, M_joint).

    Args:
        img1: First source luminance array (H, W) in [0, 1].
        img2: Second source luminance array (H, W) in [0, 1].
        n_clusters: Number of clusters K.
        method: 'kmeans' or 'fcm'.
        mode: 'per_source' or 'joint'.
        window_size: Local window size.
        random_state: Random seed.

    Returns:
        Tuple of (masks1, masks2), each of shape (K, H, W).
    """
    if n_clusters == 1:
        h, w = img1.shape
        ones = np.ones((1, h, w), dtype=np.float32)
        return ones, ones.copy()

    feat1 = extract_pixel_features(img1, window_size=window_size)
    feat2 = extract_pixel_features(img2, window_size=window_size)

    if mode == "joint":
        joint_feat = np.concatenate([feat1, feat2], axis=-1)
        masks = generate_cluster_masks(
            joint_feat,
            n_clusters=n_clusters,
            method=method,
            random_state=random_state,
        )
        return masks, masks
    elif mode == "per_source":
        masks1 = generate_cluster_masks(
            feat1,
            n_clusters=n_clusters,
            method=method,
            random_state=random_state,
        )
        masks2 = generate_cluster_masks(
            feat2,
            n_clusters=n_clusters,
            method=method,
            random_state=random_state,
        )
        return masks1, masks2
    else:
        raise ValueError(f"Unknown clustering mode: {mode}")

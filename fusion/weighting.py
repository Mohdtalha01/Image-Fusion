"""Weighting and Fusion module for CADM-MSWN.

Implements multi-structured statistic-driven weighting networks (unlearned,
deterministic, layered structure) using local energy, contrast, edge strength,
variance, spatial frequency, and gradient energy.
Includes fast O(1) guided filter smoothing and composite reconstruction.
"""

from typing import List, Tuple, Literal, Optional
import numpy as np
import cv2


def fast_guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 5,
    eps: float = 1e-4,
) -> np.ndarray:
    """Fast Guided Image Filter with O(1) box filtering.

    Paper: Kaiming He, Jian Sun, Xiaoou Tang, 'Guided Image Filtering', IEEE TPAMI 2013.
    Vectorized in float32 using optimized cv2.boxFilter.

    Args:
        guide: Guidance image of shape (H, W) in [0, 1].
        src: Source/filtering input image of shape (H, W).
        radius: Filter window radius r (filter size is 2r + 1).
        eps: Regularization parameter.

    Returns:
        Filtered image of shape (H, W).
    """
    ksize = (2 * radius + 1, 2 * radius + 1)
    guide_32 = guide.astype(np.float32)
    src_32 = src.astype(np.float32)

    mean_i = cv2.boxFilter(guide_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide_32 * guide_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide_32 * src_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    q = mean_a * guide_32 + mean_b
    return q.astype(src.dtype)


def compute_local_spatial_frequency(
    image: np.ndarray,
    window_size: int = 7,
) -> np.ndarray:
    """Compute local Spatial Frequency (SF = sqrt(RF^2 + CF^2)) in window w x w.

    Args:
        image: 2D array (H, W).
        window_size: Window dimension.

    Returns:
        Local spatial frequency map (H, W).
    """
    img = image.astype(np.float32)
    diff_row = np.zeros_like(img)
    diff_row[1:, :] = (img[1:, :] - img[:-1, :]) ** 2

    diff_col = np.zeros_like(img)
    diff_col[:, 1:] = (img[:, 1:] - img[:, :-1]) ** 2

    ksize = (window_size, window_size)
    rf_sq = cv2.boxFilter(diff_row, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    cf_sq = cv2.boxFilter(diff_col, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    sf = np.sqrt(np.maximum(0.0, rf_sq + cf_sq))
    return sf.astype(image.dtype)


def compute_component_saliency(
    component_type: Literal["GT", "GE", "FT", "FE"],
    component: np.ndarray,
    window_size: int = 7,
) -> np.ndarray:
    """Compute local statistical saliency map for a given structural component.

    Rules:
        - GT: Local energy * (1 + local contrast)
        - GE: Local edge strength (sum/mean of gradient magnitude)
        - FT: Local variance * local spatial frequency
        - FE: Local gradient energy (mean of FE^2)

    Args:
        component_type: One of 'GT', 'GE', 'FT', 'FE'.
        component: 2D array of shape (H, W).
        window_size: Size of local window w.

    Returns:
        Saliency map of shape (H, W).
    """
    c = component.astype(np.float32)
    ksize = (window_size, window_size)

    if component_type == "GT":
        mean_c = cv2.boxFilter(c, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
        mean_c2 = cv2.boxFilter(c**2, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
        local_energy = mean_c2
        local_var = np.maximum(0.0, mean_c2 - mean_c**2)
        local_contrast = np.sqrt(local_var)
        saliency = local_energy * (1.0 + local_contrast)

    elif component_type == "GE":
        saliency = cv2.boxFilter(c, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    elif component_type == "FT":
        mean_c = cv2.boxFilter(c, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
        mean_c2 = cv2.boxFilter(c**2, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
        local_var = np.maximum(0.0, mean_c2 - mean_c**2)
        local_sf = compute_local_spatial_frequency(c, window_size=window_size)
        saliency = local_var * (1.0 + local_sf)

    elif component_type == "FE":
        saliency = cv2.boxFilter(c**2, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    else:
        raise ValueError(f"Unknown component type: {component_type}")

    return np.maximum(0.0, saliency).astype(np.float32)


def compute_inter_source_weights(
    saliency_list: List[np.ndarray],
    normalization: Literal["linear", "softmax"] = "linear",
    temperature: float = 0.1,
    guides: Optional[List[np.ndarray]] = None,
    guided_radius: int = 5,
    guided_eps: float = 1e-4,
) -> List[np.ndarray]:
    """Compute normalized and smoothed weight maps across N source modalities.

    Args:
        saliency_list: List of N saliency maps, each of shape (H, W).
        normalization: 'linear' or 'softmax'.
        temperature: Softmax temperature parameter tau.
        guides: Optional list of guidance images (source luminance) for guided filtering.
        guided_radius: Radius for guided filtering.
        guided_eps: Regularization parameter for guided filtering.

    Returns:
        List of N weight maps, each (H, W), satisfying sum_i w_i = 1 for all pixels.
    """
    n = len(saliency_list)
    h, w = saliency_list[0].shape
    eps = 1e-8

    if normalization == "linear":
        total_saliency = np.zeros((h, w), dtype=np.float64)
        for s in saliency_list:
            total_saliency += s.astype(np.float64)
        weights = [
            (s.astype(np.float64) + eps / n) / (total_saliency + eps)
            for s in saliency_list
        ]
    elif normalization == "softmax":
        scaled = [s.astype(np.float64) / max(temperature, 1e-4) for s in saliency_list]
        max_s = np.maximum.reduce(scaled)
        exp_s = [np.exp(s - max_s) for s in scaled]
        sum_exp = np.sum(exp_s, axis=0) + eps
        weights = [e / sum_exp for e in exp_s]
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    # Apply guided filter smoothing if guides are provided
    if guides is not None and len(guides) == n and guided_radius > 0:
        smoothed_weights = []
        for w_map, guide in zip(weights, guides):
            filtered = fast_guided_filter(
                guide=guide,
                src=w_map,
                radius=guided_radius,
                eps=guided_eps,
            )
            smoothed_weights.append(np.maximum(0.0, filtered))

        # Re-normalize across sources to preserve exact partition of unity
        total_smoothed = np.sum(smoothed_weights, axis=0) + eps
        weights = [sw / total_smoothed for sw in smoothed_weights]

    return [w_map.astype(np.float32) for w_map in weights]


def fuse_cluster_components(
    components1: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    components2: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    masks1: np.ndarray,
    masks2: np.ndarray,
    guide1: np.ndarray,
    guide2: np.ndarray,
    window_size: int = 7,
    normalization: Literal["linear", "softmax"] = "linear",
    temperature: float = 0.1,
    guided_radius: int = 5,
    guided_eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse GT, GE, FT, FE across K clusters using the multi-structured weighting network.

    Args:
        components1: (GT1, GE1, FT1, FE1) for source 1.
        components2: (GT2, GE2, FT2, FE2) for source 2.
        masks1: Cluster masks for source 1 of shape (K, H, W).
        masks2: Cluster masks for source 2 of shape (K, H, W).
        guide1: Guidance luminance for source 1.
        guide2: Guidance luminance for source 2.
        window_size: Statistics window size.
        normalization: Weight normalization method.
        temperature: Softmax temperature.
        guided_radius: Guided filter radius.
        guided_eps: Guided filter epsilon.

    Returns:
        Tuple of (fused_GT, fused_GE, fused_FT, fused_FE), each of shape (H, W).
    """
    comp_names = ["GT", "GE", "FT", "FE"]
    n_clusters = masks1.shape[0]
    h, w = guide1.shape

    fused_outputs = {name: np.zeros((h, w), dtype=np.float32) for name in comp_names}

    for k in range(n_clusters):
        m1_k = masks1[k]
        m2_k = masks2[k]

        # Process each component type in this cluster
        for name, c1, c2 in zip(comp_names, components1, components2):
            # Cluster-gated components
            c1_k = c1 * m1_k
            c2_k = c2 * m2_k

            # Saliency maps
            s1_k = compute_component_saliency(name, c1_k, window_size=window_size)
            s2_k = compute_component_saliency(name, c2_k, window_size=window_size)

            # Guided weights
            w1_k, w2_k = compute_inter_source_weights(
                [s1_k, s2_k],
                normalization=normalization,
                temperature=temperature,
                guides=[guide1, guide2],
                guided_radius=guided_radius,
                guided_eps=guided_eps,
            )

            # Fused cluster component
            fused_c_k = w1_k * c1_k + w2_k * c2_k
            fused_outputs[name] += fused_c_k

    return (
        fused_outputs["GT"],
        fused_outputs["GE"],
        fused_outputs["FT"],
        fused_outputs["FE"],
    )


def reconstruct_fused_image(
    fused_gt: np.ndarray,
    fused_ge: np.ndarray,
    fused_ft: np.ndarray,
    fused_fe: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
) -> np.ndarray:
    """Reconstruct composite fused luminance from structural components.

    Reconstruction equation:
        F = clip( alpha * GT + beta * FT + gamma * sign(FT) * (GE + FE), 0.0, 1.0 )

    Args:
        fused_gt: Fused gross texture.
        fused_ge: Fused gross edge.
        fused_ft: Fused fine texture.
        fused_fe: Fused fine edge.
        alpha: Weight for gross texture.
        beta: Weight for fine texture.
        gamma: Weight for edge compensation.

    Returns:
        Reconstructed 2D float array in [0.0, 1.0].
    """
    edge_term = fused_ge + fused_fe
    edge_compensation = np.sign(fused_ft) * edge_term
    fused = alpha * fused_gt + beta * fused_ft + gamma * edge_compensation
    return np.clip(fused, 0.0, 1.0).astype(np.float32)

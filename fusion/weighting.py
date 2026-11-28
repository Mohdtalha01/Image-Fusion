"""Weighting and Fusion module for CADM-MSWN.

Implements multi-structured statistic-driven weighting networks (unlearned,
deterministic, layered structure) using local energy, contrast, edge strength,
variance, spatial frequency, and gradient energy.
Includes fast O(1) guided filter smoothing and composite reconstruction.
"""

from typing import List, Tuple, Literal, Optional
import numpy as np
import cv2


def precompute_guide_stats(
    guide: np.ndarray,
    radius: int = 5,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute guide mean and regularized variance for reuse across guided filter calls."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    guide_32 = np.asarray(guide, dtype=np.float32)
    mean_i = cv2.boxFilter(guide_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide_32 * guide_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    var_i_eps = (corr_i - mean_i * mean_i) + np.float32(eps)
    return guide_32, mean_i, var_i_eps


def fast_guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 5,
    eps: float = 1e-4,
    guide_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Fast Guided Image Filter with O(1) box filtering.

    Paper: Kaiming He, Jian Sun, Xiaoou Tang, 'Guided Image Filtering', IEEE TPAMI 2013.
    Vectorized in float32 using optimized cv2.boxFilter.

    Args:
        guide: Guidance image of shape (H, W) in [0, 1].
        src: Source/filtering input image of shape (H, W).
        radius: Filter window radius r (filter size is 2r + 1).
        eps: Regularization parameter.
        guide_stats: Optional precomputed (guide_32, mean_i, var_i + eps) tuple.

    Returns:
        Filtered image of shape (H, W).
    """
    ksize = (2 * radius + 1, 2 * radius + 1)
    if guide_stats is None:
        guide_32, mean_i, var_i_eps = precompute_guide_stats(guide, radius=radius, eps=eps)
    else:
        guide_32, mean_i, var_i_eps = guide_stats

    src_32 = np.asarray(src, dtype=np.float32)
    mean_p = cv2.boxFilter(src_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide_32 * src_32, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / var_i_eps
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    q = mean_a * guide_32 + mean_b
    return q if src.dtype == np.float32 else q.astype(src.dtype)


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
    img = np.asarray(image, dtype=np.float32)
    diff_row = np.zeros_like(img)
    dr = img[1:, :] - img[:-1, :]
    np.multiply(dr, dr, out=diff_row[1:, :])

    diff_col = np.zeros_like(img)
    dc = img[:, 1:] - img[:, :-1]
    np.multiply(dc, dc, out=diff_col[:, 1:])

    ksize = (window_size, window_size)
    rf_sq = cv2.boxFilter(diff_row, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    cf_sq = cv2.boxFilter(diff_col, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    sf = np.sqrt(np.maximum(np.float32(0.0), rf_sq + cf_sq))
    return sf if image.dtype == np.float32 else sf.astype(image.dtype)


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
    c = np.asarray(component, dtype=np.float32)
    ksize = (window_size, window_size)

    if component_type == "GT":
        mean_c = cv2.boxFilter(c, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
        mean_c2 = cv2.boxFilter(c * c, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
        local_var = np.maximum(np.float32(0.0), mean_c2 - mean_c * mean_c)
        saliency = mean_c2 * (np.float32(1.0) + np.sqrt(local_var))

    elif component_type == "GE":
        saliency = cv2.boxFilter(c, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    elif component_type == "FT":
        mean_c = cv2.boxFilter(c, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
        mean_c2 = cv2.boxFilter(c * c, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
        local_var = np.maximum(np.float32(0.0), mean_c2 - mean_c * mean_c)
        local_sf = compute_local_spatial_frequency(c, window_size=window_size)
        saliency = local_var * (np.float32(1.0) + local_sf)

    elif component_type == "FE":
        saliency = cv2.boxFilter(c * c, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    else:
        raise ValueError(f"Unknown component type: {component_type}")

    np.maximum(saliency, np.float32(0.0), out=saliency)
    return saliency


def compute_inter_source_weights(
    saliency_list: List[np.ndarray],
    normalization: Literal["linear", "softmax"] = "linear",
    temperature: float = 0.1,
    guides: Optional[List[np.ndarray]] = None,
    guided_radius: int = 5,
    guided_eps: float = 1e-4,
    guide_stats_list: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
) -> List[np.ndarray]:
    """Compute normalized and smoothed weight maps across N source modalities.

    Args:
        saliency_list: List of N saliency maps, each of shape (H, W).
        normalization: 'linear' or 'softmax'.
        temperature: Softmax temperature parameter tau.
        guides: Optional list of guidance images (source luminance) for guided filtering.
        guided_radius: Radius for guided filtering.
        guided_eps: Regularization parameter for guided filtering.
        guide_stats_list: Optional precomputed guide stats matching guides.

    Returns:
        List of N weight maps, each (H, W), satisfying sum_i w_i = 1 for all pixels.
    """
    n = len(saliency_list)
    eps = np.float32(1e-8)

    if normalization == "linear":
        total_saliency = np.asarray(saliency_list[0], dtype=np.float32).copy()
        for s in saliency_list[1:]:
            total_saliency += np.asarray(s, dtype=np.float32)
        inv_denom = np.float32(1.0) / (total_saliency + eps)
        eps_term = np.float32(1e-8 / n)
        weights = [
            (np.asarray(s, dtype=np.float32) + eps_term) * inv_denom
            for s in saliency_list
        ]
    elif normalization == "softmax":
        inv_temp = np.float32(1.0 / max(temperature, 1e-4))
        scaled = [np.asarray(s, dtype=np.float32) * inv_temp for s in saliency_list]
        max_s = np.maximum.reduce(scaled)
        exp_s = [np.exp(s - max_s) for s in scaled]
        sum_exp = exp_s[0].copy()
        for e in exp_s[1:]:
            sum_exp += e
        inv_sum_exp = np.float32(1.0) / (sum_exp + eps)
        weights = [e * inv_sum_exp for e in exp_s]
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    # Apply guided filter smoothing if guides are provided
    if guides is not None and len(guides) == n and guided_radius > 0:
        smoothed_weights = []
        for idx, (w_map, guide) in enumerate(zip(weights, guides)):
            g_stats = guide_stats_list[idx] if guide_stats_list is not None else None
            filtered = fast_guided_filter(
                guide=guide,
                src=w_map,
                radius=guided_radius,
                eps=guided_eps,
                guide_stats=g_stats,
            )
            np.maximum(filtered, np.float32(0.0), out=filtered)
            smoothed_weights.append(filtered)

        # Re-normalize across sources to preserve exact partition of unity
        total_smoothed = smoothed_weights[0].copy()
        for sw in smoothed_weights[1:]:
            total_smoothed += sw
        inv_total = np.float32(1.0) / (total_smoothed + eps)
        weights = [sw * inv_total for sw in smoothed_weights]

    return weights


def fuse_cluster_components(
    components1: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    components2: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    masks1: Optional[np.ndarray],
    masks2: Optional[np.ndarray],
    guide1: np.ndarray,
    guide2: np.ndarray,
    window_size: int = 7,
    normalization: Literal["linear", "softmax"] = "linear",
    temperature: float = 0.1,
    guided_radius: int = 5,
    guided_eps: float = 1e-4,
    active_components: Tuple[str, ...] = ("GT", "GE", "FT", "FE"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse GT, GE, FT, FE across K clusters using the multi-structured weighting network.

    Args:
        components1: (GT1, GE1, FT1, FE1) for source 1.
        components2: (GT2, GE2, FT2, FE2) for source 2.
        masks1: Cluster masks for source 1 of shape (K, H, W), or None for unclustered (K=1).
        masks2: Cluster masks for source 2 of shape (K, H, W), or None for unclustered (K=1).
        guide1: Guidance luminance for source 1.
        guide2: Guidance luminance for source 2.
        window_size: Statistics window size.
        normalization: Weight normalization method.
        temperature: Softmax temperature.
        guided_radius: Guided filter radius.
        guided_eps: Guided filter epsilon.
        active_components: Subset of ('GT', 'GE', 'FT', 'FE') to fuse; inactive ones return zeros.

    Returns:
        Tuple of (fused_GT, fused_GE, fused_FT, fused_FE), each of shape (H, W).
    """
    comp_names = ("GT", "GE", "FT", "FE")
    active_set = set(active_components)
    h, w = guide1.shape
    unclustered = masks1 is None or masks2 is None
    n_clusters = 1 if unclustered else masks1.shape[0]

    guide_stats_list = None
    if guided_radius > 0:
        guide_stats_list = [
            precompute_guide_stats(guide1, radius=guided_radius, eps=guided_eps),
            precompute_guide_stats(guide2, radius=guided_radius, eps=guided_eps),
        ]

    fused_outputs = {name: np.zeros((h, w), dtype=np.float32) for name in comp_names}

    for k in range(n_clusters):
        m1_k = None if unclustered else masks1[k]
        m2_k = None if unclustered else masks2[k]

        # Process each active component type in this cluster
        for name, c1, c2 in zip(comp_names, components1, components2):
            if name not in active_set:
                continue

            if unclustered:
                c1_k = c1
                c2_k = c2
            else:
                c1_k = c1 * m1_k
                c2_k = c2 * m2_k

            # Saliency maps
            s1_k = compute_component_saliency(name, c1_k, window_size=window_size)
            s2_k = compute_component_saliency(name, c2_k, window_size=window_size)

            # Guided weights reusing precomputed guide statistics
            w1_k, w2_k = compute_inter_source_weights(
                [s1_k, s2_k],
                normalization=normalization,
                temperature=temperature,
                guides=[guide1, guide2],
                guided_radius=guided_radius,
                guided_eps=guided_eps,
                guide_stats_list=guide_stats_list,
            )

            # Fused cluster component
            fused_c_k = w1_k * c1_k + w2_k * c2_k
            if unclustered:
                fused_outputs[name] = fused_c_k
            else:
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
    fused = np.float32(alpha) * fused_gt + np.float32(beta) * fused_ft
    if gamma != 0.0:
        edge_term = fused_ge + fused_fe
        fused += np.float32(gamma) * np.sign(fused_ft) * edge_term
    np.clip(fused, 0.0, 1.0, out=fused)
    return fused.astype(np.float32, copy=False)

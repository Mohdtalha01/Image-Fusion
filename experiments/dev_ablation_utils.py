"""Utility and worker functions for parallelized, cached dev-set ablation sweep.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb
from fusion.decomposition import compute_gradient_magnitude
from fusion.clustering import cluster_source_pair
from fusion.weighting import (
    compute_component_saliency,
    compute_inter_source_weights,
    fuse_cluster_components,
    reconstruct_fused_image,
)
from baselines.averaging import fuse_averaging
from baselines.gff import fuse_gff
from metrics.information import compute_entropy, compute_mutual_information
from metrics.features import compute_standard_deviation, compute_spatial_frequency, compute_average_gradient
from metrics.structural import compute_qabf_and_nabf, compute_ssim, compute_vif


def init_worker():
    """Seed each worker process deterministically."""
    np.random.seed(42)
    cv2.setRNGSeed(42)


def build_pair_cache(pair_data: Dict[str, Any], cache_dir: str) -> str:
    """Precompute and cache all intermediate structural and clustering representations.

    A single pass of 20 diffusion iterations captures states at T=5, 10, 20.
    Cluster masks for K=1, 2, 3 and base fused components for (T=20, K=3, linear, r=5)
    are stored on disk.
    """
    pair_id = pair_data["pair_id"]
    out_file = Path(cache_dir) / f"{pair_id}.npz"
    if out_file.exists():
        return pair_id

    # 1. Load source images
    im1 = cv2.imread(pair_data["path1"], cv2.IMREAD_UNCHANGED)
    im2 = cv2.imread(pair_data["path2"], cv2.IMREAD_UNCHANGED)
    if im1 is None or im2 is None:
        raise FileNotFoundError(f"Failed to read images for pair: {pair_data}")

    if im1.ndim == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    if im2.ndim == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    im1 = im1.astype(np.float32) / 255.0
    im2 = im2.astype(np.float32) / 255.0

    if im1.shape[:2] != im2.shape[:2]:
        h = min(im1.shape[0], im2.shape[0])
        w = min(im1.shape[1], im2.shape[1])
        im1 = cv2.resize(im1, (w, h))
        im2 = cv2.resize(im2, (w, h))

    # 2. Extract luminance and chroma
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(im1, "YCbCr")
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(im2, "YCbCr")

    if is_col1 and not is_col2:
        target_cb, target_cr = chroma1[0], chroma1[1]
        is_color = True
    elif is_col2 and not is_col1:
        target_cb, target_cr = chroma2[0], chroma2[1]
        is_color = True
    elif is_col1 and is_col2:
        target_cb = 0.5 * (chroma1[0] + chroma2[0])
        target_cr = 0.5 * (chroma1[1] + chroma2[1])
        is_color = True
    else:
        target_cb = np.zeros_like(lum1)
        target_cr = np.zeros_like(lum1)
        is_color = False

    # 3. Single-pass Perona-Malik Diffusion up to 20 iterations
    u1 = lum1.astype(np.float32, copy=True)
    u2 = lum2.astype(np.float32, copy=True)
    u1_states = {}
    u2_states = {}
    inv_kappa = np.float32(1.0 / 0.1176)
    step = np.float32(0.20)

    for t in range(1, 21):
        for u in (u1, u2):
            dn = np.zeros_like(u); dn[:-1, :] = u[1:, :] - u[:-1, :]
            ds = np.zeros_like(u); ds[1:, :] = u[:-1, :] - u[1:, :]
            de = np.zeros_like(u); de[:, :-1] = u[:, 1:] - u[:, :-1]
            dw = np.zeros_like(u); dw[:, 1:] = u[:, :-1] - u[:, 1:]
            cn = np.exp(-((dn * inv_kappa) ** 2))
            cs = np.exp(-((ds * inv_kappa) ** 2))
            ce = np.exp(-((de * inv_kappa) ** 2))
            cw = np.exp(-((dw * inv_kappa) ** 2))
            u += step * (cn * dn + cs * ds + ce * de + cw * dw)

        if t in (5, 10, 20):
            u1_states[t] = np.clip(u1, 0.0, 1.0).astype(lum1.dtype, copy=True)
            u2_states[t] = np.clip(u2, 0.0, 1.0).astype(lum2.dtype, copy=True)

    # 4. Decompose structural components for T in {5, 10, 20}
    comp_dict = {}
    grad_lum1 = compute_gradient_magnitude(lum1)
    grad_lum2 = compute_gradient_magnitude(lum2)

    for T in (5, 10, 20):
        gt1 = u1_states[T]
        ge1 = compute_gradient_magnitude(gt1)
        ft1 = lum1 - gt1
        fe1 = np.maximum(0.0, grad_lum1 - ge1)

        gt2 = u2_states[T]
        ge2 = compute_gradient_magnitude(gt2)
        ft2 = lum2 - gt2
        fe2 = np.maximum(0.0, grad_lum2 - ge2)

        comp_dict[f"gt1_{T}"] = gt1
        comp_dict[f"ge1_{T}"] = ge1
        comp_dict[f"ft1_{T}"] = ft1
        comp_dict[f"fe1_{T}"] = fe1

        comp_dict[f"gt2_{T}"] = gt2
        comp_dict[f"ge2_{T}"] = ge2
        comp_dict[f"ft2_{T}"] = ft2
        comp_dict[f"fe2_{T}"] = fe2

    # 5. Cluster masks for K in {1, 2, 3}
    masks_dict = {}
    masks_dict["m1_k1"] = np.ones((1, *lum1.shape), dtype=np.float32)
    masks_dict["m2_k1"] = np.ones((1, *lum2.shape), dtype=np.float32)

    m1_k2, m2_k2 = cluster_source_pair(lum1, lum2, n_clusters=2, random_state=42)
    masks_dict["m1_k2"] = m1_k2
    masks_dict["m2_k2"] = m2_k2

    m1_k3, m2_k3 = cluster_source_pair(lum1, lum2, n_clusters=3, random_state=42)
    masks_dict["m1_k3"] = m1_k3
    masks_dict["m2_k3"] = m2_k3

    # 6. Precompute base fused components for (T=20, K=3, linear, r=5)
    f_gt_base, f_ge_base, f_ft_base, f_fe_base = fuse_cluster_components(
        (comp_dict["gt1_20"], comp_dict["ge1_20"], comp_dict["ft1_20"], comp_dict["fe1_20"]),
        (comp_dict["gt2_20"], comp_dict["ge2_20"], comp_dict["ft2_20"], comp_dict["fe2_20"]),
        m1_k3, m2_k3, lum1, lum2,
        window_size=7, normalization="linear", temperature=0.1,
        guided_radius=5, guided_eps=0.0001
    )

    # Save to disk
    np.savez_compressed(
        out_file,
        im1=im1,
        im2=im2,
        lum1=lum1,
        lum2=lum2,
        is_color=is_color,
        target_cb=target_cb,
        target_cr=target_cr,
        fused_gt_base=f_gt_base,
        fused_ge_base=f_ge_base,
        fused_ft_base=f_ft_base,
        fused_fe_base=f_fe_base,
        **comp_dict,
        **masks_dict,
    )
    return pair_id


def evaluate_pair_worker(args: Tuple[Dict[str, Any], Dict[str, Any], str]) -> Dict[str, Any]:
    """Evaluate one configuration on one image pair using cached intermediates.

    Args:
        args: Tuple of (cfg_dict, pair_data, cache_dir).

    Returns:
        Dictionary of per-pair evaluation metrics and latency.
    """
    cfg_dict, pair_data, cache_dir = args
    cfg_name = cfg_dict["name"]
    cat = pair_data["category"]
    p_id = pair_data["pair_id"]

    cache_file = Path(cache_dir) / f"{p_id}.npz"
    cache = np.load(cache_file)

    im1 = cache["im1"]
    im2 = cache["im2"]
    lum1 = cache["lum1"]
    lum2 = cache["lum2"]
    is_color = bool(cache["is_color"])
    target_cb = cache["target_cb"]
    target_cr = cache["target_cr"]

    t0 = time.perf_counter()

    # Branch 1: Reference baselines
    if cfg_dict["type"] == "baseline_avg":
        fused = fuse_averaging(im1, im2)
    elif cfg_dict["type"] == "baseline_gff":
        fused = fuse_gff(im1, im2)

    # Branch 2: CADM-MSWN configurations
    else:
        # Optimization 1: Gamma sweep (gamma in {0.5, 0.25, 0.0}) directly reuses base fused components!
        if cfg_name in ("CADM_v1_prototype (gamma=0.5)", "gamma_0.25", "gamma_0.0_no_edge_comp"):
            f_lum = reconstruct_fused_image(
                cache["fused_gt_base"],
                cache["fused_ge_base"],
                cache["fused_ft_base"],
                cache["fused_fe_base"],
                alpha=1.0,
                beta=1.0,
                gamma=cfg_dict["gamma"],
            )

        # Optimization 2: Tau sweep (softmax normalization, gamma=0, T=20, K=3, r=5)
        elif cfg_name.startswith("softmax_tau_"):
            c1 = (cache["gt1_20"], cache["ge1_20"], cache["ft1_20"], cache["fe1_20"])
            c2 = (cache["gt2_20"], cache["ge2_20"], cache["ft2_20"], cache["fe2_20"])
            m1 = cache["m1_k3"]
            m2 = cache["m2_k3"]
            f_gt, f_ge, f_ft, f_fe = fuse_cluster_components(
                c1, c2, m1, m2, lum1, lum2,
                window_size=7, normalization="softmax", temperature=cfg_dict["temp"],
                guided_radius=5, guided_eps=0.0001
            )
            f_lum = reconstruct_fused_image(
                f_gt, f_ge, f_ft, f_fe,
                alpha=1.0, beta=1.0, gamma=0.0
            )

        # Optimization 3: Spatial radius sweep (r in {9, 13}, gamma=0, T=20, K=3, linear)
        elif cfg_name.startswith("guided_radius_"):
            c1 = (cache["gt1_20"], cache["ge1_20"], cache["ft1_20"], cache["fe1_20"])
            c2 = (cache["gt2_20"], cache["ge2_20"], cache["ft2_20"], cache["fe2_20"])
            m1 = cache["m1_k3"]
            m2 = cache["m2_k3"]
            f_gt, f_ge, f_ft, f_fe = fuse_cluster_components(
                c1, c2, m1, m2, lum1, lum2,
                window_size=7, normalization="linear", temperature=0.1,
                guided_radius=cfg_dict["radius"], guided_eps=0.0001
            )
            f_lum = reconstruct_fused_image(
                f_gt, f_ge, f_ft, f_fe,
                alpha=1.0, beta=1.0, gamma=0.0
            )

        # Optimization 4: Diffusion iterations sweep (T in {5, 10}, gamma=0, K=3, linear, r=5)
        elif cfg_name.startswith("T_"):
            T = cfg_dict["T"]
            c1 = (cache[f"gt1_{T}"], cache[f"ge1_{T}"], cache[f"ft1_{T}"], cache[f"fe1_{T}"])
            c2 = (cache[f"gt2_{T}"], cache[f"ge2_{T}"], cache[f"ft2_{T}"], cache[f"fe2_{T}"])
            m1 = cache["m1_k3"]
            m2 = cache["m2_k3"]
            f_gt, f_ge, f_ft, f_fe = fuse_cluster_components(
                c1, c2, m1, m2, lum1, lum2,
                window_size=7, normalization="linear", temperature=0.1,
                guided_radius=5, guided_eps=0.0001
            )
            f_lum = reconstruct_fused_image(
                f_gt, f_ge, f_ft, f_fe,
                alpha=1.0, beta=1.0, gamma=0.0
            )

        # Optimization 5: Cluster count sweep (K in {1, 2}, gamma=0, T=20, linear, r=5)
        elif cfg_name.startswith("K_"):
            K = cfg_dict["K"]
            c1 = (cache["gt1_20"], cache["ge1_20"], cache["ft1_20"], cache["fe1_20"])
            c2 = (cache["gt2_20"], cache["ge2_20"], cache["ft2_20"], cache["fe2_20"])
            m1 = cache[f"m1_k{K}"]
            m2 = cache[f"m2_k{K}"]
            f_gt, f_ge, f_ft, f_fe = fuse_cluster_components(
                c1, c2, m1, m2, lum1, lum2,
                window_size=7, normalization="linear", temperature=0.1,
                guided_radius=5, guided_eps=0.0001
            )
            f_lum = reconstruct_fused_image(
                f_gt, f_ge, f_ft, f_fe,
                alpha=1.0, beta=1.0, gamma=0.0
            )

        else:
            raise ValueError(f"Unhandled configuration: {cfg_name}")

        # Color Reconstitution
        if is_color:
            fused = ycbcr_to_rgb(f_lum, target_cb, target_cr)
        else:
            fused = f_lum

    lat_ms = (time.perf_counter() - t0) * 1000.0

    # Primary metrics
    qabf, nabf = compute_qabf_and_nabf(fused, im1, im2)
    mi = compute_mutual_information(fused, im1, im2)
    ssim = compute_ssim(fused, im1, im2)
    vifp = compute_vif(fused, im1, im2)

    # Secondary metrics
    en = compute_entropy(fused)
    sd = compute_standard_deviation(fused)
    sf = compute_spatial_frequency(fused)
    ag = compute_average_gradient(fused)

    return {
        "Config": cfg_name,
        "Category": cat,
        "Pair": p_id,
        "Latency_ms": round(lat_ms, 2),
        "QABF": qabf,
        "NABF": nabf,
        "MI": mi,
        "SSIM": ssim,
        "VIFP": vifp,
        "EN": en,
        "SD": sd,
        "SF": sf,
        "AG": ag,
    }

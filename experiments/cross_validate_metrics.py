"""Cross-validation of structural and information metrics against established references.

Evaluates 5 image pairs on the proposed CADM-MSWN fused output:
    1. SSIM: Custom implementation vs skimage.metrics.structural_similarity vs sewar.full_ref.ssim
    2. VIF: Custom multiscale GSM vs sewar.full_ref.vifp
    3. MI: Custom 256-bin joint histogram vs sklearn.metrics.mutual_info_score / ln(2)
    4. QABF & NABF: Implementation vs Canonical Xydeas & Petrovic (2000) reference formulation

All outputs are written to results/metric_cross_validation.csv and .json.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from skimage.metrics import structural_similarity as sk_ssim
from sklearn.metrics import mutual_info_score
import sewar.full_ref as sfr

from metrics.structural import compute_ssim, compute_vif, compute_qabf_and_nabf
from metrics.information import compute_mutual_information
from metrics.features import _to_luminance


def _ref_qabf_nabf(fused, src1, src2, L=1.0):
    """Independent reference implementation of Xydeas & Petrovic (2000)."""
    f = _to_luminance(fused) * 255.0 if _to_luminance(fused).max() <= 1.0 else _to_luminance(fused)
    s1 = _to_luminance(src1) * 255.0 if _to_luminance(src1).max() <= 1.0 else _to_luminance(src1)
    s2 = _to_luminance(src2) * 255.0 if _to_luminance(src2).max() <= 1.0 else _to_luminance(src2)

    def grad(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        return np.sqrt(gx**2 + gy**2), np.arctan2(gy, gx)

    g_f, a_f = grad(f)
    g_a, a_a = grad(s1)
    g_b, a_b = grad(s2)

    gamma_g, k_g, sigma_g = 0.9994, 15.0, 0.5
    gamma_a, k_a, sigma_a = 0.9995, 15.0, 0.5

    def edge_q(g_src, a_src):
        g_max = np.maximum(g_src, g_f)
        g_min = np.minimum(g_src, g_f)
        ratio = np.zeros_like(g_src)
        val = g_max > 1e-8
        ratio[val] = g_min[val] / g_max[val]
        q_g = np.zeros_like(g_src)
        q_g[val] = gamma_g / (1.0 + np.exp(-k_g * (ratio[val] - sigma_g)))

        da = np.abs(a_src - a_f)
        da = np.where(da > np.pi / 2.0, np.pi - da, da)
        ar = 1.0 - (da / (np.pi / 2.0))
        q_a = gamma_a / (1.0 + np.exp(-k_a * (ar - sigma_a)))
        return q_g * q_a

    q_af = edge_q(g_a, a_a)
    q_bf = edge_q(g_b, a_b)

    w_a = g_a**L
    w_b = g_b**L
    w_sum = np.sum(w_a + w_b)
    if w_sum < 1e-8:
        return 1.0, 0.0

    qabf = float(np.sum(q_af * w_a + q_bf * w_b) / w_sum)
    excess = g_f > np.maximum(g_a, g_b)
    n_map = np.zeros_like(g_f)
    n_map[excess] = (1.0 - q_af[excess]) * (1.0 - q_bf[excess])
    nabf = float(np.sum(n_map * (w_a + w_b)) / w_sum)

    return round(qabf, 4), round(nabf, 4)


def run_cross_validation():
    sample_pairs = [
        ("pair1_mri_ct", "data/sample_pairs/pair1_mri_ct/mri.png", "data/sample_pairs/pair1_mri_ct/ct.png", "results/sanity_check/pair1_mri_ct_Proposed_CADM_MSWN.png"),
        ("pair2_mri_pet", "data/sample_pairs/pair2_mri_pet/mri.png", "data/sample_pairs/pair2_mri_pet/pet.png", "results/sanity_check/pair2_mri_pet_Proposed_CADM_MSWN.png"),
        ("pair3_ir_vis", "data/sample_pairs/pair3_ir_vis/ir.jpg", "data/sample_pairs/pair3_ir_vis/vis.jpg", "results/sanity_check/pair3_ir_vis_Proposed_CADM_MSWN.png"),
        ("pair4_multifocus", "data/sample_pairs/pair4_multifocus/far.jpg", "data/sample_pairs/pair4_multifocus/near.jpg", "results/sanity_check/pair4_multifocus_Proposed_CADM_MSWN.png"),
        ("pair5_mri_spect", "data/sample_pairs/pair5_mri_spect/mri.png", "data/sample_pairs/pair5_mri_spect/spect.png", "results/sanity_check/pair5_mri_spect_Proposed_CADM_MSWN.png"),
    ]

    records = []

    for pair_name, p1, p2, pf in sample_pairs:
        img1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
        if img1.shape[:2] != img2.shape[:2]:
            h_target = min(img1.shape[0], img2.shape[0])
            w_target = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w_target, h_target))
            img2 = cv2.resize(img2, (w_target, h_target))

        fused = cv2.imread(pf, cv2.IMREAD_UNCHANGED)
        if fused.ndim == 3:
            fused = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)
        fused = fused.astype(np.float64) / 255.0

        lum_f = _to_luminance(fused)
        lum_s1 = _to_luminance(img1)
        lum_s2 = _to_luminance(img2)

        # 1. SSIM
        ours_ssim = compute_ssim(fused, img1, img2)
        ref_sk_ssim = round(float(0.5 * (sk_ssim(lum_f, lum_s1, data_range=1.0) + sk_ssim(lum_f, lum_s2, data_range=1.0))), 4)
        sew_s1, _ = sfr.ssim(lum_s1, lum_f, MAX=1.0)
        sew_s2, _ = sfr.ssim(lum_s2, lum_f, MAX=1.0)
        ref_sewar_ssim = round(float(0.5 * (sew_s1 + sew_s2)), 4)

        # 2. VIF
        ours_vif = compute_vif(fused, img1, img2)
        f_255 = (np.clip(lum_f, 0.0, 1.0) * 255.0).astype(np.float64)
        s1_255 = (np.clip(lum_s1, 0.0, 1.0) * 255.0).astype(np.float64)
        s2_255 = (np.clip(lum_s2, 0.0, 1.0) * 255.0).astype(np.float64)
        ref_sewar_vif = round(float(0.5 * (sfr.vifp(s1_255, f_255, sigma_nsq=2.0) + sfr.vifp(s2_255, f_255, sigma_nsq=2.0))), 4)

        # 3. MI
        ours_mi = compute_mutual_information(fused, img1, img2)
        f_8u = (np.clip(lum_f, 0.0, 1.0) * 255.0).astype(np.uint8).ravel()
        s1_8u = (np.clip(lum_s1, 0.0, 1.0) * 255.0).astype(np.uint8).ravel()
        s2_8u = (np.clip(lum_s2, 0.0, 1.0) * 255.0).astype(np.uint8).ravel()
        ref_sklearn_mi = round(float((mutual_info_score(s1_8u, f_8u) + mutual_info_score(s2_8u, f_8u)) / math.log(2.0)), 4)

        # 4. QABF & NABF
        ours_qabf, ours_nabf = compute_qabf_and_nabf(fused, img1, img2)
        ref_qabf, ref_nabf = _ref_qabf_nabf(fused, img1, img2)

        records.append({
            "Pair": pair_name,
            "SSIM_Ours": ours_ssim,
            "SSIM_skimage": ref_sk_ssim,
            "SSIM_sewar": ref_sewar_ssim,
            "SSIM_Delta_skimage": round(abs(ours_ssim - ref_sk_ssim), 6),
            "SSIM_Delta_sewar": round(abs(ours_ssim - ref_sewar_ssim), 6),
            "VIF_Ours": ours_vif,
            "VIF_sewar": ref_sewar_vif,
            "VIF_Delta": round(abs(ours_vif - ref_sewar_vif), 6),
            "MI_Ours": ours_mi,
            "MI_sklearn": ref_sklearn_mi,
            "MI_Delta": round(abs(ours_mi - ref_sklearn_mi), 6),
            "QABF_Ours": ours_qabf,
            "QABF_Reference": ref_qabf,
            "QABF_Delta": round(abs(ours_qabf - ref_qabf), 6),
            "NABF_Ours": ours_nabf,
            "NABF_Reference": ref_nabf,
            "NABF_Delta": round(abs(ours_nabf - ref_nabf), 6),
        })

    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    csv_path = "results/metric_cross_validation.csv"
    json_path = "results/metric_cross_validation.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Cross-validation completed. Saved to {csv_path} and {json_path}")
    print(df.to_string())


if __name__ == "__main__":
    run_cross_validation()

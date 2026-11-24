"""Run deep learning baselines (DenseFuse, U2Fusion, SwinFusion) on the 5 sample pairs.

Uses official task-matched checkpoints and identical YCbCr luminance fusion with chroma preservation.
Computes all 8 metrics: EN, SD, SF, AG, QABF, NABF, SSIM, VIF, latency, and process RSS memory.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.densefuse import fuse_densefuse
from baselines.u2fusion import fuse_u2fusion
from baselines.swinfusion import fuse_swinfusion
from metrics.information import compute_entropy, compute_mutual_information
from metrics.features import compute_standard_deviation, compute_spatial_frequency, compute_average_gradient
from metrics.structural import compute_ssim, compute_vif, compute_qabf_and_nabf
from metrics.profiler import profile_execution, get_hardware_info


def compute_all_metrics(fused: np.ndarray, src1: np.ndarray, src2: np.ndarray) -> dict:
    en = compute_entropy(fused)
    sd = compute_standard_deviation(fused)
    sf = compute_spatial_frequency(fused)
    ag = compute_average_gradient(fused)
    qabf, nabf = compute_qabf_and_nabf(fused, src1, src2)
    ssim = compute_ssim(fused, src1, src2)
    vif = compute_vif(fused, src1, src2)
    return {
        "EN": en,
        "SD": sd,
        "SF": sf,
        "AG": ag,
        "QABF": qabf,
        "NABF": nabf,
        "SSIM": ssim,
        "VIF": vif,
    }


def main():
    samples_dir = Path("data/sample_pairs")
    results_dir = Path("results/sanity_check")
    results_dir.mkdir(parents=True, exist_ok=True)

    pair_meta = [
        ("pair1_mri_ct", "mri.png", "ct.png", "ct_mri"),
        ("pair2_mri_pet", "mri.png", "pet.png", "pet_mri"),
        ("pair3_ir_vis", "ir.jpg", "vis.jpg", "ivif"),
        ("pair4_multifocus", "near.jpg", "far.jpg", "mff"),
        ("pair5_mri_spect", "mri.png", "spect.png", "spect_mri"),
    ]

    records = []

    for p_id, src1_name, src2_name, swin_task in pair_meta:
        p_dir = samples_dir / p_id
        path1 = p_dir / src1_name
        path2 = p_dir / src2_name

        img1 = cv2.imread(str(path1), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(str(path2), cv2.IMREAD_UNCHANGED)

        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        if img1.shape[:2] != img2.shape[:2]:
            h_target = min(img1.shape[0], img2.shape[0])
            w_target = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w_target, h_target))
            img2 = cv2.resize(img2, (w_target, h_target))

        deep_methods = {
            "DenseFuse": lambda i1, i2: fuse_densefuse(i1, i2, color_space="YCbCr"),
            "U2Fusion": lambda i1, i2: fuse_u2fusion(i1, i2, color_space="YCbCr"),
            "SwinFusion": lambda i1, i2: fuse_swinfusion(i1, i2, task=swin_task, color_space="YCbCr"),
        }

        for m_name, m_fn in deep_methods.items():
            print(f"Running {m_name} on {p_id}...")
            fused = m_fn(img1, img2)

            # Save fused image
            out_img = (np.clip(fused, 0.0, 1.0) * 255.0).astype(np.uint8)
            if out_img.ndim == 3:
                out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            else:
                out_bgr = out_img
            cv2.imwrite(str(results_dir / f"{p_id}_{m_name}.png"), out_bgr)

            # Profile latency and memory
            prof = profile_execution(m_fn, img1, img2, warmup_runs=1, measured_runs=3)

            # Compute quality metrics
            metrics = compute_all_metrics(fused, img1, img2)

            records.append({
                "Pair": p_id,
                "Method": m_name,
                "Latency_ms": prof["mean_latency_ms"],
                "Peak_RAM_RSS_MB": prof["peak_ram_rss_mb"],
                **metrics,
            })

    df = pd.DataFrame(records)
    csv_out = results_dir / "deep_baselines_sanity_metrics.csv"
    json_out = results_dir / "deep_baselines_sanity_metrics.json"
    df.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(records, f, indent=2)

    print("Deep baselines execution complete!")
    print(df.to_string())


if __name__ == "__main__":
    main()

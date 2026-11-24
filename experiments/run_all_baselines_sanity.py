"""Run profiling and all metrics for Classical Baselines & Proposed CADM-MSWN on the 5 pairs,
then merge with Deep Baselines into a unified benchmark table.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.fuse import CADMSWNFusion
from baselines.averaging import fuse_averaging
from baselines.dwt import fuse_dwt
from baselines.gff import fuse_gff
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
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    samples_dir = Path("data/sample_pairs")
    results_dir = Path("results/sanity_check")
    results_dir.mkdir(parents=True, exist_ok=True)

    cadm_pipeline = CADMSWNFusion(config=config)

    classical_methods = {
        "Averaging": lambda i1, i2: fuse_averaging(i1, i2),
        "DWT": lambda i1, i2: fuse_dwt(i1, i2, wavelet="db2", level=2),
        "GFF": lambda i1, i2: fuse_gff(i1, i2),
        "Proposed_CADM_MSWN": lambda i1, i2: cadm_pipeline.fuse(i1, i2),
    }

    pair_meta = [
        ("pair1_mri_ct", "mri.png", "ct.png", "MRI-CT (Grayscale)"),
        ("pair2_mri_pet", "mri.png", "pet.png", "MRI-PET (Color)"),
        ("pair3_ir_vis", "ir.jpg", "vis.jpg", "IR-VIS (RoadScene)"),
        ("pair4_multifocus", "near.jpg", "far.jpg", "Multi-focus (Lytro)"),
        ("pair5_mri_spect", "mri.png", "spect.png", "MRI-SPECT (Color)"),
    ]

    records = []

    for p_id, src1_name, src2_name, modal_label in pair_meta:
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

        for m_name, m_fn in classical_methods.items():
            print(f"Profiling & evaluating {m_name} on {p_id}...")
            fused = m_fn(img1, img2)
            prof = profile_execution(m_fn, img1, img2, warmup_runs=1, measured_runs=3)
            metrics = compute_all_metrics(fused, img1, img2)

            records.append({
                "Pair": p_id,
                "Modality": modal_label,
                "Method": m_name,
                "Latency_ms": prof["mean_latency_ms"],
                "Peak_RAM_RSS_MB": prof["peak_ram_rss_mb"],
                **metrics,
            })

    # Load deep baselines metrics
    deep_csv = results_dir / "deep_baselines_sanity_metrics.csv"
    if deep_csv.exists():
        df_deep = pd.read_csv(deep_csv)
        # Map modality
        mod_map = {p: m for p, _, _, m in pair_meta}
        df_deep["Modality"] = df_deep["Pair"].map(mod_map)
        df_classical = pd.DataFrame(records)
        df_all = pd.concat([df_classical, df_deep], ignore_index=True)
    else:
        df_all = pd.DataFrame(records)

    out_csv = results_dir / "complete_sanity_metrics.csv"
    out_json = results_dir / "complete_sanity_metrics.json"
    df_all.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(df_all.to_dict(orient="records"), f, indent=2)

    print("\n=== COMPLETE BENCHMARK ON 5 PAIRS (CLASSICAL + DEEP + PROPOSED) ===")
    print(df_all.to_string())
    print(f"\nSaved complete benchmark metrics to {out_csv} and {out_json}")


if __name__ == "__main__":
    main()

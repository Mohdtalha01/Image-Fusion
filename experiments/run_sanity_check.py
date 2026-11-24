"""Sanity check benchmarking script.

Runs Averaging, DWT, GFF, and Proposed CADM-MSWN on the 5 benchmark pairs.
Computes EN, SD, SF, AG for all methods under strictly identical conditions.
Saves metrics to CSV/JSON and generates comprehensive comparison visuals.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.fuse import CADMSWNFusion
from baselines.averaging import fuse_averaging
from baselines.dwt import fuse_dwt
from baselines.gff import fuse_gff


def compute_metrics(image: np.ndarray) -> dict:
    """Compute measured objective quality metrics on an image array in [0, 1].

    Metrics computed:
        - EN: Shannon Entropy (bits)
        - SD: Standard Deviation
        - SF: Spatial Frequency (sqrt(RF^2 + CF^2))
        - AG: Average Gradient
    """
    if image.ndim == 3:
        lum = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        lum = image.copy()

    lum_8u = (np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8)

    # 1. Entropy (EN)
    hist, _ = np.histogram(lum_8u, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    en = float(-np.sum(hist * np.log2(hist)))

    # 2. Standard Deviation (SD)
    sd = float(np.std(lum))

    # 3. Spatial Frequency (SF)
    diff_r = lum[1:, :] - lum[:-1, :]
    diff_c = lum[:, 1:] - lum[:, :-1]
    rf = np.sqrt(np.mean(diff_r**2))
    cf = np.sqrt(np.mean(diff_c**2))
    sf = float(np.sqrt(rf**2 + cf**2))

    # 4. Average Gradient (AG)
    gx = cv2.Sobel(lum, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(lum, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    ag = float(np.mean(np.sqrt((gx**2 + gy**2) / 2.0)))

    return {
        "EN": round(en, 4),
        "SD": round(sd, 4),
        "SF": round(sf, 4),
        "AG": round(ag, 4),
    }


def run_benchmark():
    """Execute 4 methods across 5 pairs and record metrics."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    samples_dir = Path("data/sample_pairs")
    results_dir = Path("results/sanity_check")
    results_dir.mkdir(parents=True, exist_ok=True)

    cadm_pipeline = CADMSWNFusion(config=config)

    methods = {
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

        for m_name, m_fn in methods.items():
            fused = m_fn(img1, img2)

            # Save fused image
            out_bgr = (np.clip(fused, 0.0, 1.0) * 255.0).astype(np.uint8)
            if out_bgr.ndim == 3:
                out_bgr = cv2.cvtColor(out_bgr, cv2.COLOR_RGB2BGR)
            out_path = results_dir / f"{p_id}_{m_name}.png"
            cv2.imwrite(str(out_path), out_bgr)

            m_scores = compute_metrics(fused)
            records.append({
                "Pair": p_id,
                "Modality": modal_label,
                "Method": m_name,
                "EN": m_scores["EN"],
                "SD": m_scores["SD"],
                "SF": m_scores["SF"],
                "AG": m_scores["AG"],
            })

    df = pd.DataFrame(records)
    csv_path = results_dir / "sanity_metrics_comparison.csv"
    json_path = results_dir / "sanity_metrics_comparison.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print("\n--- MEASURED SANITY BENCHMARK METRICS ---")
    print(df.to_string(index=False))
    print(f"\nSaved raw data to {csv_path} and {json_path}")


if __name__ == "__main__":
    run_benchmark()

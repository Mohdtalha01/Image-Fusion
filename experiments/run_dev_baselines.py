"""Run GFF and Averaging reference baselines on the 120 Dev Pairs.

Saves granular results to results/dev_baselines_results.csv
and summary (mean & std) to results/dev_baselines_summary.csv.
"""

import os
import sys
import time
from pathlib import Path
import yaml
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.traditional.averaging import AveragingFusion
from baselines.traditional.gff import GFFFusion
from metrics.information import compute_entropy, compute_mutual_information
from metrics.features import compute_standard_deviation, compute_spatial_frequency, compute_average_gradient
from metrics.structural import compute_qabf_and_nabf, compute_ssim, compute_vif, compute_viff


def load_dev_pairs():
    with open("splits.yaml", "r") as f:
        splits = yaml.safe_load(f)

    dev_data = splits["dev"]
    pairs = []

    for item in dev_data["msrs"]:
        pairs.append({"category": "MSRS (IVIF)", "pair_id": item["id"], "path1": item["ir"], "path2": item["vi"]})

    for item in dev_data["harvard_ct_mri"]:
        pairs.append({"category": "Harvard (CT-MRI)", "pair_id": item["id"], "path1": item["src1"], "path2": item["src2"]})

    for item in dev_data["harvard_pet_mri"]:
        pairs.append({"category": "Harvard (PET-MRI)", "pair_id": item["id"], "path1": item["src1"], "path2": item["src2"]})

    for item in dev_data["harvard_spect_mri"]:
        pairs.append({"category": "Harvard (SPECT-MRI)", "pair_id": item["id"], "path1": item["src1"], "path2": item["src2"]})

    for item in dev_data["lytro"]:
        pairs.append({"category": "Lytro (Multi-focus)", "pair_id": item["id"], "path1": item["src1"], "path2": item["src2"]})

    return pairs


def main():
    print("Loading Dev Set pairs from splits.yaml...")
    dev_pairs = load_dev_pairs()
    print(f"Total dev pairs loaded: {len(dev_pairs)}")

    cached_images = []
    for item in dev_pairs:
        im1 = cv2.imread(item["path1"], cv2.IMREAD_UNCHANGED)
        im2 = cv2.imread(item["path2"], cv2.IMREAD_UNCHANGED)
        if im1 is None or im2 is None:
            raise FileNotFoundError(f"Failed to read {item}")
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

        cached_images.append({
            "category": item["category"],
            "pair_id": item["pair_id"],
            "img1": im1,
            "img2": im2,
        })

    baselines = [
        {"name": "Averaging", "model": AveragingFusion()},
        {"name": "GFF", "model": GFFFusion()},
    ]

    all_results = []
    summary_records = []

    for b in baselines:
        b_name = b["name"]
        pipeline = b["model"]
        print(f"\nRunning Baseline: {b_name} on 120 dev pairs...")
        t_start = time.perf_counter()

        b_records = []
        for pair_data in cached_images:
            cat = pair_data["category"]
            p_id = pair_data["pair_id"]
            i1 = pair_data["img1"]
            i2 = pair_data["img2"]

            t0 = time.perf_counter()
            fused = pipeline.fuse(i1, i2)
            lat_ms = (time.perf_counter() - t0) * 1000

            qabf, nabf = compute_qabf_and_nabf(fused, i1, i2)
            mi = compute_mutual_information(fused, i1, i2)
            ssim = compute_ssim(fused, i1, i2)
            vifp = compute_vif(fused, i1, i2)
            viff = compute_viff(fused, i1, i2)

            en = compute_entropy(fused)
            sd = compute_standard_deviation(fused)
            sf = compute_spatial_frequency(fused)
            ag = compute_average_gradient(fused)

            rec = {
                "Config": b_name,
                "Category": cat,
                "Pair": p_id,
                "Latency_ms": round(lat_ms, 2),
                "QABF": qabf,
                "NABF": nabf,
                "MI": mi,
                "SSIM": ssim,
                "VIFP": vifp,
                "VIFF": viff,
                "EN": en,
                "SD": sd,
                "SF": sf,
                "AG": ag,
            }
            b_records.append(rec)
            all_results.append(rec)

        total_time = time.perf_counter() - t_start
        df_b = pd.DataFrame(b_records)

        summary = {
            "Config": b_name,
            "Mean_QABF": round(float(df_b["QABF"].mean()), 4),
            "Std_QABF": round(float(df_b["QABF"].std()), 4),
            "Mean_NABF": round(float(df_b["NABF"].mean()), 4),
            "Std_NABF": round(float(df_b["NABF"].std()), 4),
            "Mean_MI": round(float(df_b["MI"].mean()), 4),
            "Std_MI": round(float(df_b["MI"].std()), 4),
            "Mean_SSIM": round(float(df_b["SSIM"].mean()), 4),
            "Std_SSIM": round(float(df_b["SSIM"].std()), 4),
            "Mean_VIFP": round(float(df_b["VIFP"].mean()), 4),
            "Std_VIFP": round(float(df_b["VIFP"].std()), 4),
            "Mean_VIFF": round(float(df_b["VIFF"].mean()), 4),
            "Std_VIFF": round(float(df_b["VIFF"].std()), 4),
            "Mean_EN": round(float(df_b["EN"].mean()), 4),
            "Std_EN": round(float(df_b["EN"].std()), 4),
            "Mean_SD": round(float(df_b["SD"].mean()), 4),
            "Std_SD": round(float(df_b["SD"].std()), 4),
            "Mean_SF": round(float(df_b["SF"].mean()), 4),
            "Std_SF": round(float(df_b["SF"].std()), 4),
            "Mean_AG": round(float(df_b["AG"].mean()), 4),
            "Std_AG": round(float(df_b["AG"].std()), 4),
            "Mean_Latency_ms": round(float(df_b["Latency_ms"].mean()), 2),
            "Std_Latency_ms": round(float(df_b["Latency_ms"].std()), 2),
            "Total_Wall_Time_s": round(total_time, 2),
        }
        summary_records.append(summary)
        print(f"Completed {b_name} in {total_time:.1f}s")

    out_dir = Path("results")
    pd.DataFrame(all_results).to_csv(out_dir / "dev_baselines_results.csv", index=False)
    pd.DataFrame(summary_records).to_csv(out_dir / "dev_baselines_summary.csv", index=False)
    print("Dev baselines complete!")


if __name__ == "__main__":
    main()

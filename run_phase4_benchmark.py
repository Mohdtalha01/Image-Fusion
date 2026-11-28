import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import yaml
import torch

from fusion.fuse import CADMSWNFusion
from baselines.averaging import fuse_averaging
from baselines.gff import fuse_gff
from baselines.dwt import fuse_dwt
from baselines.densefuse import fuse_densefuse
from baselines.u2fusion import fuse_u2fusion
from baselines.swinfusion import fuse_swinfusion
from metrics.information import compute_entropy, compute_mutual_information
from metrics.features import compute_standard_deviation, compute_spatial_frequency, compute_average_gradient
from metrics.structural import compute_qabf_and_nabf, compute_ssim, compute_vif

_FROZEN_CADM_CONFIG = {
    "color": {"color_space": "YCbCr"},
    "clustering": {"n_clusters": 1, "window_size": 7, "random_state": 42},
    "diffusion": {"iterations": 20, "kappa": 0.1176, "step_size": 0.20, "conduction": "exponential"},
    "weighting": {
        "window_size": 7,
        "normalization": "linear",
        "temperature": 0.1,
        "guided_filter": {"enabled": True, "radius": 5, "eps": 0.0001},
        "reconstruction": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0, "signed_fine_edge": False},
    },
}
_FROZEN_PIPELINE = CADMSWNFusion(config=_FROZEN_CADM_CONFIG)


def fuse_cadm_frozen(im1, im2):
    """Frozen CADM-MSWN v2 variant: T=20, K=1, gamma=0.0, linear norm, r=5."""
    return _FROZEN_PIPELINE.fuse(im1, im2)

def evaluate_metrics(fused, im1, im2):
    qabf, nabf = compute_qabf_and_nabf(fused, im1, im2)
    mi = compute_mutual_information(fused, im1, im2)
    ssim = compute_ssim(fused, im1, im2)
    vifp = compute_vif(fused, im1, im2)
    en = compute_entropy(fused)
    sd = compute_standard_deviation(fused)
    sf = compute_spatial_frequency(fused)
    ag = compute_average_gradient(fused)
    return {
        "QABF": qabf, "NABF": nabf, "MI": mi, "SSIM": ssim, "VIFP": vifp,
        "EN": en, "SD": sd, "SF": sf, "AG": ag,
    }

def main():
    with open("splits.yaml", "r") as f:
        splits = yaml.safe_load(f)

    test_splits = splits.get("test", {})
    print(f"Test datasets in splits.yaml: {list(test_splits.keys())}")

    # Also load TNO, RoadScene, MSRS test pairs
    all_datasets = {}
    for ds_name in ["harvard_ct_mri", "harvard_pet_mri", "harvard_spect_mri", "lytro"]:
        pairs = []
        for item in test_splits.get(ds_name, []):
            pairs.append({"id": item["id"], "path1": item["src1"], "path2": item["src2"]})
        all_datasets[ds_name] = pairs

    # TNO test (42 pairs)
    tno_dir = Path("data/raw/tno")
    if tno_dir.exists():
        tno_pairs = []
        # read pairs from inventory
        with open("results/all_datasets_verified_inventory.json", "r") as f:
            inv = json.load(f)
        for item in inv["datasets"].get("tno", []):
            tno_pairs.append({"id": item["pair_id"], "path1": item["source1"], "path2": item["source2"]})
        all_datasets["tno"] = tno_pairs

    # RoadScene test (sample of 15 pairs for quick benchmarking if not running all 220)
    with open("results/all_datasets_verified_inventory.json", "r") as f:
        inv = json.load(f)
    for ds_name in ["roadscene", "msrs"]:
        pairs = []
        for item in inv["datasets"].get(ds_name, []):
            pairs.append({"id": item["pair_id"], "path1": item["source1"], "path2": item["source2"]})
        all_datasets[ds_name] = pairs

    methods = {
        "Averaging": lambda i1, i2, task: fuse_averaging(i1, i2),
        "DWT": lambda i1, i2, task: fuse_dwt(i1, i2),
        "GFF": lambda i1, i2, task: fuse_gff(i1, i2),
        "DenseFuse": lambda i1, i2, task: fuse_densefuse(i1, i2, strategy="l1"),
        "U2Fusion": lambda i1, i2, task: fuse_u2fusion(i1, i2),
        "SwinFusion": lambda i1, i2, task: fuse_swinfusion(i1, i2, task=task),
        "CADM_MSWN_v2": lambda i1, i2, task: fuse_cadm_frozen(i1, i2),
    }

    task_map = {
        "harvard_ct_mri": "ct_mri",
        "harvard_pet_mri": "pet_mri",
        "harvard_spect_mri": "spect_mri",
        "lytro": "mff",
        "tno": "ivif",
        "roadscene": "ivif",
        "msrs": "ivif",
    }

    all_records = []
    print("Beginning test benchmarking across all datasets...")

    for ds_name, pairs in all_datasets.items():
        swin_task = task_map.get(ds_name, "ivif")
        print(f"\n=== Dataset: {ds_name} ({len(pairs)} pairs) ===")

        for p_info in pairs:
            p_id = p_info["id"]
            im1 = cv2.imread(p_info["path1"], cv2.IMREAD_UNCHANGED)
            im2 = cv2.imread(p_info["path2"], cv2.IMREAD_UNCHANGED)
            if im1 is None or im2 is None:
                continue

            if im1.ndim == 3: im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            if im2.ndim == 3: im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            im1 = im1.astype(np.float32) / 255.0
            im2 = im2.astype(np.float32) / 255.0
            if im1.shape[:2] != im2.shape[:2]:
                h = min(im1.shape[0], im2.shape[0])
                w = min(im1.shape[1], im2.shape[1])
                im1 = cv2.resize(im1, (w, h))
                im2 = cv2.resize(im2, (w, h))

            for m_name, m_fn in methods.items():
                t0 = time.perf_counter()
                fused = m_fn(im1, im2, swin_task)
                lat = (time.perf_counter() - t0) * 1000.0

                mets = evaluate_metrics(fused, im1, im2)
                all_records.append({
                    "Dataset": ds_name,
                    "Pair": p_id,
                    "Method": m_name,
                    "Latency_ms": round(lat, 2),
                    **mets,
                })

    df = pd.DataFrame(all_records)
    out_csv = "results/phase4_test_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nAll test metrics saved to {out_csv} ({len(df)} total evaluations)")

    # Print summary per dataset and method
    grouped = df.groupby(["Dataset", "Method"])[["QABF", "NABF", "MI", "SSIM", "VIFP", "EN", "SD", "SF", "AG", "Latency_ms"]].mean().round(4)
    summary_csv = "results/phase4_test_summary.csv"
    grouped.to_csv(summary_csv)
    print("\n=== TEST BENCHMARK SUMMARY ===")
    print(grouped.to_string())

if __name__ == "__main__":
    main()

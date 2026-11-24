import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import torch

from baselines.densefuse import fuse_densefuse
from baselines.u2fusion import fuse_u2fusion
from baselines.swinfusion import fuse_swinfusion
from metrics.information import compute_entropy, compute_mutual_information
from metrics.features import compute_standard_deviation, compute_spatial_frequency, compute_average_gradient
from metrics.structural import compute_ssim, compute_vif, compute_qabf_and_nabf

samples_dir = Path("data/sample_pairs")
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

pair_meta = [
    ("pair1_mri_ct", "mri.png", "ct.png", "ct_mri", "MRI-CT (Grayscale)"),
    ("pair2_mri_pet", "mri.png", "pet.png", "pet_mri", "MRI-PET (Color)"),
    ("pair3_ir_vis", "ir.jpg", "vis.jpg", "ivif", "IR-VIS (RoadScene)"),
    ("pair4_multifocus", "near.jpg", "far.jpg", "mff", "Multi-focus (Lytro)"),
    ("pair5_mri_spect", "mri.png", "spect.png", "spect_mri", "MRI-SPECT (Color)"),
]

records = []

for p_id, src1_name, src2_name, swin_task, modality in pair_meta:
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
        "DenseFuse": lambda i1, i2: fuse_densefuse(i1, i2, strategy="l1", color_space="YCbCr"),
        "U2Fusion": lambda i1, i2: fuse_u2fusion(i1, i2, color_space="YCbCr"),
        "SwinFusion": lambda i1, i2: fuse_swinfusion(i1, i2, task=swin_task, color_space="YCbCr"),
    }

    for m_name, m_fn in deep_methods.items():
        print(f"Evaluating fixed {m_name} on {p_id} ({modality})...")
        fused = m_fn(img1, img2)

        en = compute_entropy(fused)
        sd = compute_standard_deviation(fused)
        sf = compute_spatial_frequency(fused)
        ag = compute_average_gradient(fused)
        qabf, nabf = compute_qabf_and_nabf(fused, img1, img2)
        mi = compute_mutual_information(fused, img1, img2)
        ssim = compute_ssim(fused, img1, img2)
        vif = compute_vif(fused, img1, img2)

        records.append({
            "Pair": p_id,
            "Modality": modality,
            "Method": m_name,
            "EN": en,
            "SD": sd,
            "SF": sf,
            "AG": ag,
            "QABF": qabf,
            "NABF": nabf,
            "MI": mi,
            "SSIM": ssim,
            "VIF": vif,
        })

df = pd.DataFrame(records)
csv_out = results_dir / "baseline_wrapper_fix_evidence.csv"
json_out = results_dir / "baseline_wrapper_fix_evidence.json"
df.to_csv(csv_out, index=False)
with open(json_out, "w") as f:
    json.dump(records, f, indent=2)

print("\n=== BASELINE WRAPPER FIX VERIFICATION EVIDENCE ===")
print(df[["Pair", "Method", "SSIM", "QABF", "VIF", "MI"]].to_string())

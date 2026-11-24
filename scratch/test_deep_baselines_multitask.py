import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from baselines import fuse_densefuse, fuse_u2fusion, fuse_swinfusion, fuse_cddfuse, fuse_piafusion

tasks = [
    {
        "task_name": "Harvard Grayscale (MRI-CT)",
        "task_code": "ct_mri",
        "pair_id": "pair1_mri_ct",
        "path1": "data/sample_pairs/pair1_mri_ct/mri.png",
        "path2": "data/sample_pairs/pair1_mri_ct/ct.png",
        "swin_task": "ct_mri",
        "cdd_task": "mif",
    },
    {
        "task_name": "Harvard Color (MRI-PET)",
        "task_code": "pet_mri",
        "pair_id": "pair2_mri_pet",
        "path1": "data/sample_pairs/pair2_mri_pet/mri.png",
        "path2": "data/sample_pairs/pair2_mri_pet/pet.png",
        "swin_task": "pet_mri",
        "cdd_task": "mif",
    },
    {
        "task_name": "Multi-focus (Lytro)",
        "task_code": "multifocus",
        "pair_id": "pair4_multifocus",
        "path1": "data/sample_pairs/pair4_multifocus/near.jpg",
        "path2": "data/sample_pairs/pair4_multifocus/far.jpg",
        "swin_task": "mff",
        "cdd_task": "ivif",
    },
    {
        "task_name": "Infrared-Visible (RoadScene IR-VIS)",
        "task_code": "ivif",
        "pair_id": "pair3_ir_vis",
        "path1": "data/sample_pairs/pair3_ir_vis/ir.jpg",
        "path2": "data/sample_pairs/pair3_ir_vis/vis.jpg",
        "swin_task": "ivif",
        "cdd_task": "ivif",
    },
]

def load_pair(p1, p2):
    im1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
    im2 = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
    if im1.ndim == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    if im2.ndim == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    if im1.shape[:2] != im2.shape[:2]:
        h = min(im1.shape[0], im2.shape[0])
        w = min(im1.shape[1], im2.shape[1])
        im1 = cv2.resize(im1, (w, h))
        im2 = cv2.resize(im2, (w, h))
    return im1, im2

def main():
    print("=== MULTI-TASK DEEP BASELINE COMPREHENSIVE SMOKE TEST ===")
    results_matrix = []

    for t in tasks:
        t_name = t["task_name"]
        print(f"\n==========================================")
        print(f"TASK: {t_name}")
        print(f"==========================================")
        im1, im2 = load_pair(t["path1"], t["path2"])
        print(f"Input shapes: {im1.shape}, {im2.shape}")

        methods = [
            ("DenseFuse", lambda: fuse_densefuse(im1, im2)),
            ("U2Fusion", lambda: fuse_u2fusion(im1, im2)),
            ("SwinFusion", lambda: fuse_swinfusion(im1, im2, task=t["swin_task"])),
            ("CDDFuse", lambda: fuse_cddfuse(im1, im2, task=t["cdd_task"])),
            ("PIAFusion", lambda: fuse_piafusion(im1, im2, task=t["cdd_task"])),
        ]

        for m_name, fn in methods:
            row = {
                "Task": t_name,
                "Method": m_name,
                "Status": "UNKNOWN",
                "Latency_s": None,
                "Shape": None,
                "Details": "",
            }
            try:
                t0 = time.perf_counter()
                fused = fn()
                dt = time.perf_counter() - t0
                assert fused is not None
                assert 0.0 <= fused.min() and fused.max() <= 1.0
                row["Status"] = "PASS"
                row["Latency_s"] = round(dt, 3)
                row["Shape"] = str(fused.shape)
                row["Details"] = f"Range [{fused.min():.3f}, {fused.max():.3f}]"
                print(f"  [{m_name}] PASS ({dt:.3f}s, shape {fused.shape})")
            except Exception as e:
                row["Status"] = "FAIL"
                row["Details"] = str(e).split("\n")[0]
                print(f"  [{m_name}] FAIL: {row['Details']}")
            results_matrix.append(row)

    df = pd.DataFrame(results_matrix)
    df.to_csv("results/deep_baselines_multitask_matrix.csv", index=False)
    with open("results/deep_baselines_multitask_report.json", "w") as f:
        json.dump(results_matrix, f, indent=2)

    print("\n=== SUMMARY MATRIX ===")
    pivot = df.pivot(index="Method", columns="Task", values="Status")
    print(pivot.to_string())

if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "baselines", "deep", "cddfuse"))

from metrics.structural import compute_qabf_and_nabf
from utils.Evaluator import Evaluator
from metrics.features import _to_luminance

samples_dir = "data/sample_pairs"
pairs = [
    ("pair1_mri_ct", "mri.png", "ct.png"),
    ("pair2_mri_pet", "mri.png", "pet.png"),
    ("pair3_ir_vis", "ir.jpg", "vis.jpg"),
    ("pair4_multifocus", "near.jpg", "far.jpg"),
    ("pair5_mri_spect", "mri.png", "spect.png"),
]

from baselines.densefuse import fuse_densefuse
from baselines.swinfusion import fuse_swinfusion

print("=== QABF COMPARISON: OUR IMPLEMENTATION vs CDDFuse Evaluator ===")
print(f"{'Pair':18s} | {'Method':10s} | {'QABF (Ours)':12s} | {'QABF (CDDFuse)':14s} | {'Delta':10s}")
print("-" * 75)

for p_id, f1_name, f2_name in pairs:
    p1 = os.path.join(samples_dir, p_id, f1_name)
    p2 = os.path.join(samples_dir, p_id, f2_name)
    i1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
    i2 = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
    if i1.ndim == 3: i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
    if i2.ndim == 3: i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)
    i1 = i1.astype(np.float32) / 255.0
    i2 = i2.astype(np.float32) / 255.0
    if i1.shape[:2] != i2.shape[:2]:
        h, w = min(i1.shape[0], i2.shape[0]), min(i1.shape[1], i2.shape[1])
        i1 = cv2.resize(i1, (w, h))
        i2 = cv2.resize(i2, (w, h))

    # Test with fused image from DenseFuse
    fused = fuse_densefuse(i1, i2)

    # Ours:
    q_ours, n_ours = compute_qabf_and_nabf(fused, i1, i2)

    # CDDFuse Evaluator expects 2D grayscale [0, 255] float:
    f_gray = (_to_luminance(fused) * 255.0).astype(np.float32)
    a_gray = (_to_luminance(i1) * 255.0).astype(np.float32)
    b_gray = (_to_luminance(i2) * 255.0).astype(np.float32)

    q_cdd = Evaluator.Qabf(f_gray, a_gray, b_gray)
    delta = abs(q_ours - q_cdd)

    print(f"{p_id:18s} | {'DenseFuse':10s} | {q_ours:12.4f} | {q_cdd:14.4f} | {delta:10.4f}")

    # Test with SwinFusion
    fused_sw = fuse_swinfusion(i1, i2, task="ct_mri")
    q_ours_sw, _ = compute_qabf_and_nabf(fused_sw, i1, i2)
    f_sw_gray = (_to_luminance(fused_sw) * 255.0).astype(np.float32)
    q_cdd_sw = Evaluator.Qabf(f_sw_gray, a_gray, b_gray)
    delta_sw = abs(q_ours_sw - q_cdd_sw)
    print(f"{p_id:18s} | {'SwinFusion':10s} | {q_ours_sw:12.4f} | {q_cdd_sw:14.4f} | {delta_sw:10.4f}")

import os
import sys
import importlib.util
import numpy as np
import cv2
import pandas as pd

# Clean import of Evaluator from CDDFuse without polluting sys.path
eval_path = os.path.join(os.getcwd(), "baselines", "deep", "cddfuse", "utils", "Evaluator.py")
spec = importlib.util.spec_from_file_location("cddfuse_evaluator", eval_path)
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)
Evaluator = eval_mod.Evaluator

from metrics.structural import compute_qabf_and_nabf
from metrics.features import _to_luminance
from baselines.densefuse import fuse_densefuse
from baselines.swinfusion import fuse_swinfusion
from baselines.u2fusion import fuse_u2fusion

samples_dir = "data/sample_pairs"
pairs = [
    ("pair1_mri_ct", "mri.png", "ct.png", "ct_mri"),
    ("pair2_mri_pet", "mri.png", "pet.png", "ct_mri"),
    ("pair3_ir_vis", "ir.jpg", "vis.jpg", "ivif"),
    ("pair4_multifocus", "near.jpg", "far.jpg", "mff"),
    ("pair5_mri_spect", "mri.png", "spect.png", "ct_mri"),
]

records = []

for p_id, f1_name, f2_name, task in pairs:
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

    methods = {
        "DenseFuse": fuse_densefuse(i1, i2),
        "SwinFusion": fuse_swinfusion(i1, i2, task=task),
        "U2Fusion": fuse_u2fusion(i1, i2),
    }

    a_gray = (_to_luminance(i1) * 255.0).astype(np.float32)
    b_gray = (_to_luminance(i2) * 255.0).astype(np.float32)

    for m_name, fused in methods.items():
        q_ours, n_ours = compute_qabf_and_nabf(fused, i1, i2)
        f_gray = (_to_luminance(fused) * 255.0).astype(np.float32)
        q_cdd = float(Evaluator.Qabf(f_gray, a_gray, b_gray))
        delta = abs(q_ours - q_cdd)

        records.append({
            "Pair": p_id,
            "Method": m_name,
            "QABF_Canonical_PetrovicXydeas": q_ours,
            "QABF_CDDFuse_Evaluator": round(q_cdd, 4),
            "Delta": round(delta, 4),
            "Relative_Agreement_Pct": round((1.0 - delta / max(q_ours, 1e-4)) * 100.0, 2),
            "NABF_Canonical": n_ours,
        })

df = pd.DataFrame(records)
out_csv = "results/qabf_toolbox_cross_validation.csv"
df.to_csv(out_csv, index=False)
print("QABF cross-validation saved to", out_csv)
print(df.to_string())
print(f"\nMean Delta across all {len(df)} evaluations: {df['Delta'].mean():.4f}")
print(f"Mean Agreement: {df['Relative_Agreement_Pct'].mean():.2f}%")

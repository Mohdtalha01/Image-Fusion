import os
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
from metrics.structural import compute_qabf_and_nabf
from baselines.deep.cddfuse.utils.Evaluator import Evaluator

def main():
    pairs = [
        ("pair1_mri_ct", "data/sample_pairs/pair1_mri_ct/mri.png", "data/sample_pairs/pair1_mri_ct/ct.png", "results/sanity_check/pair1_mri_ct_Proposed_CADM_MSWN.png"),
        ("pair2_mri_pet", "data/sample_pairs/pair2_mri_pet/mri.png", "data/sample_pairs/pair2_mri_pet/pet.png", "results/sanity_check/pair2_mri_pet_Proposed_CADM_MSWN.png"),
        ("pair3_ir_vis", "data/sample_pairs/pair3_ir_vis/ir.jpg", "data/sample_pairs/pair3_ir_vis/vis.jpg", "results/sanity_check/pair3_ir_vis_Proposed_CADM_MSWN.png"),
        ("pair4_multifocus", "data/sample_pairs/pair4_multifocus/near.jpg", "data/sample_pairs/pair4_multifocus/far.jpg", "results/sanity_check/pair4_multifocus_Proposed_CADM_MSWN.png"),
        ("pair5_mri_spect", "data/sample_pairs/pair5_mri_spect/mri.png", "data/sample_pairs/pair5_mri_spect/spect.png", "results/sanity_check/pair5_mri_spect_Proposed_CADM_MSWN.png"),
    ]

    records = []
    print("=== COMPUTING METRIC INDEPENDENCE & TOOLBOX COMPARISON ===")
    for p_id, s1_p, s2_p, f_p in pairs:
        s1 = cv2.imread(s1_p, cv2.IMREAD_UNCHANGED)
        s2 = cv2.imread(s2_p, cv2.IMREAD_UNCHANGED)
        fused = cv2.imread(f_p, cv2.IMREAD_UNCHANGED)

        # Ensure spatial dimension match
        if s1.shape[:2] != s2.shape[:2]:
            h_tgt = min(s1.shape[0], s2.shape[0])
            w_tgt = min(s1.shape[1], s2.shape[1])
            s1 = cv2.resize(s1, (w_tgt, h_tgt))
            s2 = cv2.resize(s2, (w_tgt, h_tgt))
            
        if fused.shape[:2] != s1.shape[:2]:
            fused = cv2.resize(fused, (s1.shape[1], s1.shape[0]))

        # Luminance float64 for CDDFuse evaluator
        def to_lum(img):
            if img.ndim == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            return img.astype(np.float64)

        s1_lum = to_lum(s1)
        s2_lum = to_lum(s2)
        f_lum = to_lum(fused)

        # 1. Canonical Xydeas & Petrovic (2000)
        q_canonical, n_canonical = compute_qabf_and_nabf(fused, s1, s2)

        # 2. CDDFuse Toolbox Evaluator.Qabf (Richang Hong 2007 MATLAB port)
        q_cddfuse = round(float(Evaluator.Qabf(f_lum, s1_lum, s2_lum)), 4)

        delta_q = round(abs(q_canonical - q_cddfuse), 4)

        records.append({
            "Pair": p_id,
            "QABF_Canonical_Xydeas2000": q_canonical,
            "QABF_CDDFuse_Toolbox": q_cddfuse,
            "Delta_QABF": delta_q,
            "NABF_Petrovic2005": n_canonical,
            "NABF_Independent_Ref": "NONE (No public toolbox)",
            "Reason_for_Delta": "CDDFuse sets G_AF=0 on flat areas and omits unoriented angle symmetry modulo pi"
        })

    df = pd.DataFrame(records)
    out_csv = "results/metric_independence_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved comparison to {out_csv}")

if __name__ == "__main__":
    main()

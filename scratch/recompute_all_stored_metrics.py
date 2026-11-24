import os
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
from metrics.structural import compute_qabf_and_nabf

def recompute_sanity_metrics():
    print("=== Recomputing QABF and NABF in results/sanity_check/complete_sanity_metrics.csv ===")
    csv_path = "results/sanity_check/complete_sanity_metrics.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    pair_sources = {
        "pair1_mri_ct": ("data/sample_pairs/pair1_mri_ct/mri.png", "data/sample_pairs/pair1_mri_ct/ct.png"),
        "pair2_mri_pet": ("data/sample_pairs/pair2_mri_pet/mri.png", "data/sample_pairs/pair2_mri_pet/pet.png"),
        "pair3_ir_vis": ("data/sample_pairs/pair3_ir_vis/ir.jpg", "data/sample_pairs/pair3_ir_vis/vis.jpg"),
        "pair4_multifocus": ("data/sample_pairs/pair4_multifocus/near.jpg", "data/sample_pairs/pair4_multifocus/far.jpg"),
        "pair5_mri_spect": ("data/sample_pairs/pair5_mri_spect/mri.png", "data/sample_pairs/pair5_mri_spect/spect.png"),
    }
    
    for idx, row in df.iterrows():
        p_id = row["Pair"]
        method = row["Method"]
        if p_id not in pair_sources:
            continue
            
        s1_path, s2_path = pair_sources[p_id]
        s1 = cv2.imread(s1_path, cv2.IMREAD_UNCHANGED)
        s2 = cv2.imread(s2_path, cv2.IMREAD_UNCHANGED)
        
        if s1.shape[:2] != s2.shape[:2]:
            h_target = min(s1.shape[0], s2.shape[0])
            w_target = min(s1.shape[1], s2.shape[1])
            s1 = cv2.resize(s1, (w_target, h_target))
            s2 = cv2.resize(s2, (w_target, h_target))
        
        # Fused image path
        f_name = f"{p_id}_{method}.png"
        f_path = os.path.join("results/sanity_check", f_name)
        if not os.path.exists(f_path):
            print(f"Warning: {f_path} not found")
            continue
            
        fused = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        if fused.shape[:2] != s1.shape[:2]:
            fused = cv2.resize(fused, (s1.shape[1], s1.shape[0]))
            
        qabf, nabf = compute_qabf_and_nabf(fused, s1, s2)
        df.at[idx, "QABF"] = qabf
        df.at[idx, "NABF"] = nabf
        
    df.to_csv(csv_path, index=False)
    print(f"Saved recomputed metrics to {csv_path}")

def recompute_deep_sanity_metrics():
    print("=== Recomputing QABF and NABF in results/sanity_check/deep_baselines_sanity_metrics.csv ===")
    csv_path = "results/sanity_check/deep_baselines_sanity_metrics.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    pair_sources = {
        "pair1_mri_ct": ("data/sample_pairs/pair1_mri_ct/mri.png", "data/sample_pairs/pair1_mri_ct/ct.png"),
        "pair2_mri_pet": ("data/sample_pairs/pair2_mri_pet/mri.png", "data/sample_pairs/pair2_mri_pet/pet.png"),
        "pair3_ir_vis": ("data/sample_pairs/pair3_ir_vis/ir.jpg", "data/sample_pairs/pair3_ir_vis/vis.jpg"),
        "pair4_multifocus": ("data/sample_pairs/pair4_multifocus/near.jpg", "data/sample_pairs/pair4_multifocus/far.jpg"),
        "pair5_mri_spect": ("data/sample_pairs/pair5_mri_spect/mri.png", "data/sample_pairs/pair5_mri_spect/spect.png"),
    }
    
    for idx, row in df.iterrows():
        p_id = row["Pair"]
        method = row["Method"]
        if p_id not in pair_sources:
            continue
            
        s1_path, s2_path = pair_sources[p_id]
        s1 = cv2.imread(s1_path, cv2.IMREAD_UNCHANGED)
        s2 = cv2.imread(s2_path, cv2.IMREAD_UNCHANGED)
        
        if s1.shape[:2] != s2.shape[:2]:
            h_target = min(s1.shape[0], s2.shape[0])
            w_target = min(s1.shape[1], s2.shape[1])
            s1 = cv2.resize(s1, (w_target, h_target))
            s2 = cv2.resize(s2, (w_target, h_target))
        
        # Fused image path
        f_name = f"{p_id}_{method}.png"
        f_path = os.path.join("results/sanity_check", f_name)
        if not os.path.exists(f_path):
            print(f"Warning: {f_path} not found")
            continue
            
        fused = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        if fused.shape[:2] != s1.shape[:2]:
            fused = cv2.resize(fused, (s1.shape[1], s1.shape[0]))
            
        qabf, nabf = compute_qabf_and_nabf(fused, s1, s2)
        df.at[idx, "QABF"] = qabf
        df.at[idx, "NABF"] = nabf
        
    df.to_csv(csv_path, index=False)
    print(f"Saved recomputed metrics to {csv_path}")

if __name__ == "__main__":
    recompute_sanity_metrics()
    recompute_deep_sanity_metrics()

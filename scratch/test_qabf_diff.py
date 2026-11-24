import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
import os
from metrics.structural import compute_qabf_and_nabf
from baselines.deep.cddfuse.utils.Evaluator import Evaluator

pairs = [
    ("pair1_mri_ct", "data/sample_pairs/pair1_mri_ct/mri.png", "data/sample_pairs/pair1_mri_ct/ct.png", "results/sanity_check/pair1_mri_ct_Proposed_CADM_MSWN.png"),
    ("pair2_mri_pet", "data/sample_pairs/pair2_mri_pet/mri.png", "data/sample_pairs/pair2_mri_pet/pet.png", "results/sanity_check/pair2_mri_pet_Proposed_CADM_MSWN.png"),
    ("pair3_ir_vis", "data/sample_pairs/pair3_ir_vis/ir.jpg", "data/sample_pairs/pair3_ir_vis/vis.jpg", "results/sanity_check/pair3_ir_vis_Proposed_CADM_MSWN.png"),
    ("pair4_multifocus", "data/sample_pairs/pair4_multifocus/near.jpg", "data/sample_pairs/pair4_multifocus/far.jpg", "results/sanity_check/pair4_multifocus_Proposed_CADM_MSWN.png"),
    ("pair5_mri_spect", "data/sample_pairs/pair5_mri_spect/mri.png", "data/sample_pairs/pair5_mri_spect/spect.png", "results/sanity_check/pair5_mri_spect_Proposed_CADM_MSWN.png")
]

for name, s1_p, s2_p, f_p in pairs:
    s1 = cv2.imread(s1_p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    s2 = cv2.imread(s2_p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    f = cv2.imread(f_p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if s1.shape != f.shape:
        s1 = cv2.resize(s1, (f.shape[1], f.shape[0]))
    if s2.shape != f.shape:
        s2 = cv2.resize(s2, (f.shape[1], f.shape[0]))
    
    q_ours, n_ours = compute_qabf_and_nabf(f/255.0, s1/255.0, s2/255.0)
    q_cdd = Evaluator.Qabf(f, s1, s2)
    diff = abs(q_ours - q_cdd)
    print(f"{name}: Ours={q_ours:.4f}, CDDFuse={q_cdd:.4f}, Diff={diff:.4f}, NABF={n_ours:.4f}")

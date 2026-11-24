import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
import math
from scipy.signal import convolve2d
from baselines.deep.cddfuse.utils.Evaluator import Evaluator
from metrics.structural import compute_qabf_and_nabf

s1_p = "data/sample_pairs/pair1_mri_ct/mri.png"
s2_p = "data/sample_pairs/pair1_mri_ct/ct.png"
f_p = "results/sanity_check/pair1_mri_ct_Proposed_CADM_MSWN.png"

s1 = cv2.imread(s1_p, cv2.IMREAD_GRAYSCALE).astype(np.float64)
s2 = cv2.imread(s2_p, cv2.IMREAD_GRAYSCALE).astype(np.float64)
f = cv2.imread(f_p, cv2.IMREAD_GRAYSCALE).astype(np.float64)

# Original CDDFuse
q_cdd = Evaluator.Qabf(f, s1, s2)
# Our original
q_ours, n_ours = compute_qabf_and_nabf(f/255.0, s1/255.0, s2/255.0)

print(f"Original Ours: {q_ours:.4f}")
print(f"Original CDDFuse: {q_cdd:.4f}")
print(f"Difference: {abs(q_ours - q_cdd):.4f}")

# Now let's test variations step-by-step to see what causes the shift
def qabf_variant(img_F, img_A, img_B, 
                 use_cdd_sobel=True, 
                 gamma_a=0.9879, ka=-22.0, da=0.8,
                 gamma_g=0.9994, kg=-15.0, dg=0.5,
                 fold_angles=False,
                 fix_equal_g=False):
    if use_cdd_sobel:
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        SAx = convolve2d(img_A, h3, mode='same')
        SAy = convolve2d(img_A, h1, mode='same')
        gA = np.sqrt(SAx**2 + SAy**2)
        aA = np.zeros_like(img_A)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0] = np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        
        SBx = convolve2d(img_B, h3, mode='same')
        SBy = convolve2d(img_B, h1, mode='same')
        gB = np.sqrt(SBx**2 + SBy**2)
        aB = np.zeros_like(img_B)
        aB[SBx == 0] = math.pi / 2
        aB[SBx != 0] = np.arctan(SBy[SBx != 0] / SBx[SBx != 0])
        
        SFx = convolve2d(img_F, h3, mode='same')
        SFy = convolve2d(img_F, h1, mode='same')
        gF = np.sqrt(SFx**2 + SFy**2)
        aF = np.zeros_like(img_F)
        aF[SFx == 0] = math.pi / 2
        aF[SFx != 0] = np.arctan(SFy[SFx != 0] / SFx[SFx != 0])
    else:
        gx_a = cv2.Sobel(img_A, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        gy_a = cv2.Sobel(img_A, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        gA = np.sqrt(gx_a**2 + gy_a**2)
        aA = np.arctan2(gy_a, gx_a)
        
        gx_b = cv2.Sobel(img_B, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        gy_b = cv2.Sobel(img_B, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        gB = np.sqrt(gx_b**2 + gy_b**2)
        aB = np.arctan2(gy_b, gx_b)
        
        gx_f = cv2.Sobel(img_F, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        gy_f = cv2.Sobel(img_F, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        gF = np.sqrt(gx_f**2 + gy_f**2)
        aF = np.arctan2(gy_f, gx_f)

    def get_q(g_src, a_src):
        G = np.zeros_like(g_src)
        mask_gt = g_src > gF
        mask_lt = g_src < gF
        mask_eq = g_src == gF
        
        G[mask_gt] = gF[mask_gt] / g_src[mask_gt]
        G[mask_lt] = g_src[mask_lt] / gF[mask_lt]
        if fix_equal_g:
            G[mask_eq] = 1.0
        else:
            G[mask_eq] = gF[mask_eq] # CDDFuse bug
            
        if fold_angles:
            delta_a = np.abs(a_src - aF)
            delta_a = np.where(delta_a > np.pi / 2.0, np.pi - delta_a, delta_a)
            A = 1.0 - delta_a / (np.pi / 2.0)
        else:
            A = 1.0 - np.abs(a_src - aF) / (math.pi / 2.0)
            
        Qg = gamma_g / (1.0 + np.exp(kg * (G - dg)))
        Qa = gamma_a / (1.0 + np.exp(ka * (A - da)))
        return Qg * Qa
        
    QAF = get_q(gA, aA)
    QBF = get_q(gB, aB)
    
    deno = np.sum(gA + gB)
    nume = np.sum(QAF * gA + QBF * gB)
    return nume / (deno + 1e-12)

print("Testing variants:")
v_cdd_exact = qabf_variant(f, s1, s2, use_cdd_sobel=True, gamma_a=0.9879, ka=-22.0, da=0.8, fold_angles=False, fix_equal_g=False)
print(f"Variant 1 (CDDFuse exact logic in Python): {v_cdd_exact:.4f}")

v_cdd_fixed_g = qabf_variant(f, s1, s2, use_cdd_sobel=True, gamma_a=0.9879, ka=-22.0, da=0.8, fold_angles=False, fix_equal_g=True)
print(f"Variant 2 (CDDFuse + fix equal G): {v_cdd_fixed_g:.4f}")

v_cdd_folded_angle = qabf_variant(f, s1, s2, use_cdd_sobel=True, gamma_a=0.9879, ka=-22.0, da=0.8, fold_angles=True, fix_equal_g=True)
print(f"Variant 3 (CDDFuse + fold angles): {v_cdd_folded_angle:.4f}")

v_our_params = qabf_variant(f, s1, s2, use_cdd_sobel=True, gamma_a=0.9995, ka=-15.0, da=0.5, fold_angles=True, fix_equal_g=True)
print(f"Variant 4 (With our old parameters gamma_a=0.9995, ka=-15, da=0.5): {v_our_params:.4f}")

v_cv2_sobel = qabf_variant(f, s1, s2, use_cdd_sobel=False, gamma_a=0.9879, ka=-22.0, da=0.8, fold_angles=True, fix_equal_g=True)
print(f"Variant 5 (CV2 Sobel + Canonical Parameters Ta=0.9879, ka=-22, Da=0.8 + fold angles): {v_cv2_sobel:.4f}")

import numpy as np
import cv2
import math
from scipy.signal import convolve2d

def compute_canonical_qabf_nabf(fused, src1, src2, L=1.0):
    """Canonical Xydeas & Petrovic (2000) and Petrovic & Xydeas (2005) implementation.
    
    Model parameters (Electronics Letters 2000):
      Gamma_g = 0.9994, kappa_g = 15.0, sigma_g = 0.5 (or -15 with subtraction)
      Gamma_alpha = 0.9879, kappa_alpha = 22.0, sigma_alpha = 0.8
    """
    # Sobel kernels with 1/8 normalization
    def get_gradient(img):
        # Convert to float64 luminance
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lum = img.astype(np.float64)
        if lum.max() <= 1.0:
            lum = lum * 255.0
            
        gx = cv2.Sobel(lum, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(lum, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        g = np.sqrt(gx**2 + gy**2)
        a = np.arctan2(gy, gx) # range [-pi, pi]
        return g, a

    g_f, a_f = get_gradient(fused)
    g_a, a_a = get_gradient(src1)
    g_b, a_b = get_gradient(src2)

    Tg, kg, Dg = 0.9994, 15.0, 0.5
    Ta, ka, Da = 0.9879, 22.0, 0.8

    def calc_q(g_s, a_s):
        # 1. Edge strength preservation Qg
        g_max = np.maximum(g_s, g_f)
        g_min = np.minimum(g_s, g_f)
        
        G = np.zeros_like(g_s)
        mask = g_max > 1e-8
        G[mask] = g_min[mask] / g_max[mask]
        
        Qg = np.zeros_like(g_s)
        Qg[mask] = Tg / (1.0 + np.exp(-kg * (G[mask] - Dg)))
        
        # 2. Edge orientation preservation Qa with unoriented angle symmetry modulo pi
        delta_a = np.abs(a_s - a_f)
        delta_a = np.where(delta_a > np.pi / 2.0, np.pi - delta_a, delta_a)
        A = 1.0 - (delta_a / (np.pi / 2.0))
        
        Qa = Ta / (1.0 + np.exp(-ka * (A - Da)))
        
        return Qg * Qa

    q_af = calc_q(g_a, a_a)
    q_bf = calc_q(g_b, a_b)

    w_a = g_a**L
    w_b = g_b**L
    total_w = w_a + w_b
    sum_w = np.sum(total_w)

    if sum_w < 1e-8:
        return 1.0, 0.0

    qabf = float(np.sum(q_af * w_a + q_bf * w_b) / sum_w)

    # NABF: Petrovic & Xydeas (2005) artifact noise where fused edges exceed both sources
    excess = (g_f > g_a) & (g_f > g_b)
    # Additive weight preservation form:
    nabf_map = np.zeros_like(g_f)
    nabf_map[excess] = (1.0 - q_af[excess]) * w_a[excess] + (1.0 - q_bf[excess]) * w_b[excess]
    nabf = float(np.sum(nabf_map) / sum_w)

    return round(float(np.clip(qabf, 0.0, 1.0)), 4), round(float(np.clip(nabf, 0.0, 1.0)), 4)

pairs = [
    ("pair1_mri_ct", "data/sample_pairs/pair1_mri_ct/mri.png", "data/sample_pairs/pair1_mri_ct/ct.png", "results/sanity_check/pair1_mri_ct_Proposed_CADM_MSWN.png"),
    ("pair2_mri_pet", "data/sample_pairs/pair2_mri_pet/mri.png", "data/sample_pairs/pair2_mri_pet/pet.png", "results/sanity_check/pair2_mri_pet_Proposed_CADM_MSWN.png"),
    ("pair3_ir_vis", "data/sample_pairs/pair3_ir_vis/ir.jpg", "data/sample_pairs/pair3_ir_vis/vis.jpg", "results/sanity_check/pair3_ir_vis_Proposed_CADM_MSWN.png"),
    ("pair4_multifocus", "data/sample_pairs/pair4_multifocus/near.jpg", "data/sample_pairs/pair4_multifocus/far.jpg", "results/sanity_check/pair4_multifocus_Proposed_CADM_MSWN.png"),
    ("pair5_mri_spect", "data/sample_pairs/pair5_mri_spect/mri.png", "data/sample_pairs/pair5_mri_spect/spect.png", "results/sanity_check/pair5_mri_spect_Proposed_CADM_MSWN.png")
]

for name, s1_p, s2_p, f_p in pairs:
    s1 = cv2.imread(s1_p)
    s2 = cv2.imread(s2_p)
    f = cv2.imread(f_p)
    if s1.shape != f.shape:
        s1 = cv2.resize(s1, (f.shape[1], f.shape[0]))
    if s2.shape != f.shape:
        s2 = cv2.resize(s2, (f.shape[1], f.shape[0]))
        
    q, n = compute_canonical_qabf_nabf(f, s1, s2)
    print(f"{name}: Canonical QABF={q:.4f}, Canonical NABF={n:.4f}")

import os
import sys
import numpy as np
import cv2
import torch

from baselines.densefuse import fuse_densefuse, get_densefuse_model, fuse_densefuse_luminance
from baselines.swinfusion import fuse_swinfusion, get_swinfusion_model, fuse_swinfusion_luminance
from baselines.u2fusion import fuse_u2fusion
from metrics.structural import compute_ssim
from metrics.features import _to_luminance
from skimage.metrics import structural_similarity as ssim_fn

samples_dir = "data/sample_pairs"
pairs = [
    ("pair1_mri_ct", "mri.png", "ct.png", "ct_mri"),
    ("pair2_mri_pet", "mri.png", "pet.png", "pet_mri"),
    ("pair5_mri_spect", "mri.png", "spect.png", "spect_mri"),
]

for p_id, f1, f2, task in pairs:
    p1 = os.path.join(samples_dir, p_id, f1)
    p2 = os.path.join(samples_dir, p_id, f2)
    img1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
    if img1.ndim == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    if img2.ndim == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    print(f"\n=================== {p_id} ({task}) ===================")
    print(f"img1: shape={img1.shape}, min={img1.min():.4f}, max={img1.max():.4f}, mean={img1.mean():.4f}")
    print(f"img2: shape={img2.shape}, min={img2.min():.4f}, max={img2.max():.4f}, mean={img2.mean():.4f}")

    # Check DenseFuse
    f_df = fuse_densefuse(img1, img2)
    l_df = _to_luminance(f_df)
    l1 = _to_luminance(img1)
    l2 = _to_luminance(img2)
    s1_df = ssim_fn(l_df, l1, data_range=1.0)
    s2_df = ssim_fn(l_df, l2, data_range=1.0)
    print(f"DenseFuse: min={f_df.min():.4f}, max={f_df.max():.4f}, mean={f_df.mean():.4f}")
    print(f"  SSIM(F, img1)={s1_df:.4f}, SSIM(F, img2)={s2_df:.4f}, Mean={0.5*(s1_df+s2_df):.4f}")

    # Check SwinFusion
    f_sw = fuse_swinfusion(img1, img2, task=task)
    l_sw = _to_luminance(f_sw)
    s1_sw = ssim_fn(l_sw, l1, data_range=1.0)
    s2_sw = ssim_fn(l_sw, l2, data_range=1.0)
    print(f"SwinFusion: min={f_sw.min():.4f}, max={f_sw.max():.4f}, mean={f_sw.mean():.4f}")
    print(f"  SSIM(F, img1)={s1_sw:.4f}, SSIM(F, img2)={s2_sw:.4f}, Mean={0.5*(s1_sw+s2_sw):.4f}")

    # Let's also test SwinFusion with 'ct_mri' or 'ivif' or swapped inputs
    f_sw_ivif = fuse_swinfusion(img1, img2, task="ivif")
    s1_sw_ivif = ssim_fn(_to_luminance(f_sw_ivif), l1, data_range=1.0)
    s2_sw_ivif = ssim_fn(_to_luminance(f_sw_ivif), l2, data_range=1.0)
    print(f"SwinFusion (ivif task): SSIM(F, img1)={s1_sw_ivif:.4f}, SSIM(F, img2)={s2_sw_ivif:.4f}, Mean={0.5*(s1_sw_ivif+s2_sw_ivif):.4f}")

    # Test swapped inputs (img2, img1)
    f_sw_swap = fuse_swinfusion(img2, img1, task=task)
    s1_sw_swap = ssim_fn(_to_luminance(f_sw_swap), l1, data_range=1.0)
    s2_sw_swap = ssim_fn(_to_luminance(f_sw_swap), l2, data_range=1.0)
    print(f"SwinFusion (swapped inputs): SSIM(F, img1)={s1_sw_swap:.4f}, SSIM(F, img2)={s2_sw_swap:.4f}, Mean={0.5*(s1_sw_swap+s2_sw_swap):.4f}")

    f_df_swap = fuse_densefuse(img2, img1)
    s1_df_swap = ssim_fn(_to_luminance(f_df_swap), l1, data_range=1.0)
    s2_df_swap = ssim_fn(_to_luminance(f_df_swap), l2, data_range=1.0)
    print(f"DenseFuse (swapped inputs): SSIM(F, img1)={s1_df_swap:.4f}, SSIM(F, img2)={s2_df_swap:.4f}, Mean={0.5*(s1_df_swap+s2_df_swap):.4f}")

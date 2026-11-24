import os
import sys
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim_fn

from baselines.swinfusion import fuse_swinfusion_luminance
from metrics.features import _to_luminance

samples_dir = "data/sample_pairs"
p_dir = os.path.join(samples_dir, "pair5_mri_spect")
p1 = os.path.join(p_dir, "mri.png")
p2 = os.path.join(p_dir, "spect.png")

i1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
i2 = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
if i1.ndim == 3: i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
if i2.ndim == 3: i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)
i1 = i1.astype(np.float32) / 255.0
i2 = i2.astype(np.float32) / 255.0

l1 = _to_luminance(i1).astype(np.float32)
l2 = _to_luminance(i2).astype(np.float32)

print("Direct SSIM(MRI, SPECT):", ssim_fn(l1, l2, data_range=1.0))

for task in ["ct_mri", "pet_mri", "ivif", "mff"]:
    f_lum = fuse_swinfusion_luminance(l1, l2, task=task)
    s1 = ssim_fn(f_lum, l1, data_range=1.0)
    s2 = ssim_fn(f_lum, l2, data_range=1.0)
    print(f"Task [{task:8s}]: SSIM(F, MRI)={s1:.4f}, SSIM(F, SPECT)={s2:.4f}, Mean={0.5*(s1+s2):.4f}")

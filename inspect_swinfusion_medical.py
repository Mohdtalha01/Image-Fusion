import os
import sys
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim_fn

from baselines.swinfusion import get_swinfusion_model, fuse_swinfusion_luminance
from metrics.structural import compute_ssim
from metrics.features import _to_luminance

samples_dir = "data/sample_pairs"
p_dir = os.path.join(samples_dir, "pair2_mri_pet")
p1 = os.path.join(p_dir, "mri.png")
p2 = os.path.join(p_dir, "pet.png")

img1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
if img1.ndim == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
if img2.ndim == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img1 = img1.astype(np.float32) / 255.0
img2 = img2.astype(np.float32) / 255.0

lum1 = _to_luminance(img1)
lum2 = _to_luminance(img2)

print(f"lum1 (MRI): min={lum1.min():.4f}, max={lum1.max():.4f}, mean={lum1.mean():.4f}")
print(f"lum2 (PET): min={lum2.min():.4f}, max={lum2.max():.4f}, mean={lum2.mean():.4f}")

# Check SSIM between lum1 and lum2 directly
print(f"Direct SSIM(MRI, PET) = {ssim_fn(lum1, lum2, data_range=1.0):.4f}")

# Let's test different checkpoints of SwinFusion
for ckpt in ["ct_mri", "pet_mri", "ivif", "mff"]:
    f_lum = fuse_swinfusion_luminance(lum1, lum2, task=ckpt)
    s1 = ssim_fn(f_lum, lum1, data_range=1.0)
    s2 = ssim_fn(f_lum, lum2, data_range=1.0)
    print(f"\nSwinFusion [{ckpt}]: min={f_lum.min():.4f}, max={f_lum.max():.4f}, mean={f_lum.mean():.4f}")
    print(f"  SSIM(F, MRI) = {s1:.4f}")
    print(f"  SSIM(F, PET) = {s2:.4f}")
    print(f"  Mean SSIM    = {0.5*(s1+s2):.4f}")

# Also let's check what CDDFuse does on this pair
cdd_ckpt = "baselines/deep/cddfuse/models/CDDFuse_MIF.pth"
print("\nCDDFuse_MIF exists:", os.path.exists(cdd_ckpt))

# Let's check DenseFuse L1-norm vs addition vs average
from baselines.densefuse import get_densefuse_model
net_df = get_densefuse_model()
t1 = torch.from_numpy(lum1.astype(np.float32)).unsqueeze(0).unsqueeze(0)
t2 = torch.from_numpy(lum2.astype(np.float32)).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    feat1 = net_df.encoder(t1)
    feat2 = net_df.encoder(t2)
    # L1 norm fusion:
    # w1 = l1_norm(feat1) / (l1_norm(feat1) + l1_norm(feat2))
    # w2 = l1_norm(feat2) / (l1_norm(feat1) + l1_norm(feat2))
    # feat_f = w1 * feat1 + w2 * feat2
    l1_f1 = torch.sum(torch.abs(feat1), dim=1, keepdim=True)
    l1_f2 = torch.sum(torch.abs(feat2), dim=1, keepdim=True)
    denom = l1_f1 + l1_f2 + 1e-8
    w1 = l1_f1 / denom
    w2 = l1_f2 / denom
    feat_l1 = w1 * feat1 + w2 * feat2
    out_l1 = net_df.decoder(feat_l1).squeeze().numpy()

s1_l1 = ssim_fn(np.clip(out_l1, 0, 1), lum1, data_range=1.0)
s2_l1 = ssim_fn(np.clip(out_l1, 0, 1), lum2, data_range=1.0)
print(f"\nDenseFuse L1-norm rule on pair2: min={out_l1.min():.4f}, max={out_l1.max():.4f}, mean={out_l1.mean():.4f}")
print(f"  SSIM(F, MRI) = {s1_l1:.4f}")
print(f"  SSIM(F, PET) = {s2_l1:.4f}")
print(f"  Mean SSIM    = {0.5*(s1_l1+s2_l1):.4f}")

import os
import sys
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim_fn

from baselines.densefuse import get_densefuse_model
from baselines.swinfusion import get_swinfusion_model, fuse_swinfusion_luminance
from baselines.u2fusion import get_u2fusion_model, fuse_u2fusion_luminance
from metrics.features import _to_luminance

samples_dir = "data/sample_pairs"
pairs = [
    ("pair1_mri_ct", "mri.png", "ct.png"),
    ("pair2_mri_pet", "mri.png", "pet.png"),
    ("pair3_ir_vis", "ir.jpg", "vis.jpg"),
    ("pair4_multifocus", "near.jpg", "far.jpg"),
    ("pair5_mri_spect", "mri.png", "spect.png"),
]

net_df = get_densefuse_model()
net_u2 = get_u2fusion_model()

print("="*80)
print("TESTING DENSEFUSE STRATEGIES")
print("="*80)

def fuse_df_l1(t1, t2):
    f1 = net_df.encoder(t1)
    f2 = net_df.encoder(t2)
    # L1 norm activity with 3x3 box filter as per Li & Wu (2019)
    c1 = torch.sum(torch.abs(f1), dim=1, keepdim=True)
    c2 = torch.sum(torch.abs(f2), dim=1, keepdim=True)
    kernel = torch.ones(1, 1, 3, 3) / 9.0
    c1_bar = torch.nn.functional.conv2d(c1, kernel, padding=1)
    c2_bar = torch.nn.functional.conv2d(c2, kernel, padding=1)
    w1 = c1_bar / (c1_bar + c2_bar + 1e-8)
    w2 = c2_bar / (c1_bar + c2_bar + 1e-8)
    f_fused = w1 * f1 + w2 * f2
    return net_df.decoder(f_fused)

def fuse_df_avg(t1, t2):
    f1 = net_df.encoder(t1)
    f2 = net_df.encoder(t2)
    f_fused = 0.5 * (f1 + f2)
    return net_df.decoder(f_fused)

def fuse_df_add_minmax(t1, t2):
    f1 = net_df.encoder(t1)
    f2 = net_df.encoder(t2)
    f_fused = f1 + f2
    out = net_df.decoder(f_fused)
    # min-max normalize
    mi, ma = out.min(), out.max()
    return (out - mi) / (ma - mi + 1e-8)

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
    l1 = _to_luminance(i1).astype(np.float32)
    l2 = _to_luminance(i2).astype(np.float32)
    t1 = torch.from_numpy(l1).unsqueeze(0).unsqueeze(0)
    t2 = torch.from_numpy(l2).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out_raw_add = net_df(t1, t2).squeeze().numpy()
        out_l1 = fuse_df_l1(t1, t2).squeeze().numpy()
        out_avg = fuse_df_avg(t1, t2).squeeze().numpy()
        out_add_norm = fuse_df_add_minmax(t1, t2).squeeze().numpy()

    s_raw = 0.5 * (ssim_fn(np.clip(out_raw_add, 0, 1), l1, data_range=1.0) + ssim_fn(np.clip(out_raw_add, 0, 1), l2, data_range=1.0))
    s_l1 = 0.5 * (ssim_fn(np.clip(out_l1, 0, 1), l1, data_range=1.0) + ssim_fn(np.clip(out_l1, 0, 1), l2, data_range=1.0))
    s_avg = 0.5 * (ssim_fn(np.clip(out_avg, 0, 1), l1, data_range=1.0) + ssim_fn(np.clip(out_avg, 0, 1), l2, data_range=1.0))
    s_norm = 0.5 * (ssim_fn(np.clip(out_add_norm, 0, 1), l1, data_range=1.0) + ssim_fn(np.clip(out_add_norm, 0, 1), l2, data_range=1.0))

    print(f"{p_id:20s} | Raw Add SSIM: {s_raw:.4f} | L1-norm SSIM: {s_l1:.4f} | Avg SSIM: {s_avg:.4f} | Norm Add SSIM: {s_norm:.4f}")

print("\n"+"="*80)
print("TESTING U2FUSION OUTPUTS")
print("="*80)
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
    l1 = _to_luminance(i1).astype(np.float32)
    l2 = _to_luminance(i2).astype(np.float32)
    t1 = torch.from_numpy(l1).unsqueeze(0).unsqueeze(0)
    t2 = torch.from_numpy(l2).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out_u2 = net_u2(t1, t2).squeeze().numpy()

    # How does U2Fusion official repo process output?
    # Let's inspect out_u2 values
    s_u2_clip = 0.5 * (ssim_fn(np.clip(out_u2, 0, 1), l1, data_range=1.0) + ssim_fn(np.clip(out_u2, 0, 1), l2, data_range=1.0))
    mi, ma = out_u2.min(), out_u2.max()
    out_u2_norm = (out_u2 - mi) / (ma - mi + 1e-8)
    s_u2_norm = 0.5 * (ssim_fn(out_u2_norm, l1, data_range=1.0) + ssim_fn(out_u2_norm, l2, data_range=1.0))
    print(f"{p_id:20s} | U2 raw min={mi:.4f}, max={ma:.4f}, mean={out_u2.mean():.4f} | Clip SSIM: {s_u2_clip:.4f} | Norm SSIM: {s_u2_norm:.4f}")

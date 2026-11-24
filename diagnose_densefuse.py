import os
import sys
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim_fn

from baselines.densefuse import get_densefuse_model

samples_dir = "data/sample_pairs"
p1 = os.path.join(samples_dir, "pair1_mri_ct", "mri.png")
p2 = os.path.join(samples_dir, "pair1_mri_ct", "ct.png")

img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

net = get_densefuse_model()
print("Model loaded successfully")

t1_1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0)
t2_1 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    rec1 = net.decoder(net.encoder(t1_1)).squeeze().numpy()
    rec2 = net.decoder(net.encoder(t2_1)).squeeze().numpy()

print(f"img1: min={img1.min():.4f}, max={img1.max():.4f}, mean={img1.mean():.4f}")
print(f"rec1: min={rec1.min():.4f}, max={rec1.max():.4f}, mean={rec1.mean():.4f}")
print(f"SSIM(rec1, img1) = {ssim_fn(np.clip(rec1, 0, 1), img1, data_range=1.0):.4f}")

print(f"img2: min={img2.min():.4f}, max={img2.max():.4f}, mean={img2.mean():.4f}")
print(f"rec2: min={rec2.min():.4f}, max={rec2.max():.4f}, mean={rec2.mean():.4f}")
print(f"SSIM(rec2, img2) = {ssim_fn(np.clip(rec2, 0, 1), img2, data_range=1.0):.4f}")

with torch.no_grad():
    f1 = net.encoder(t1_1)
    f2 = net.encoder(t2_1)
    out_add = net.decoder(f1 + f2).squeeze().numpy()
    out_avg = net.decoder(0.5 * (f1 + f2)).squeeze().numpy()

s1_add = ssim_fn(np.clip(out_add, 0, 1), img1, data_range=1.0)
s2_add = ssim_fn(np.clip(out_add, 0, 1), img2, data_range=1.0)
print(f"Addition rule: min={out_add.min():.4f}, max={out_add.max():.4f}, mean={out_add.mean():.4f}")
print(f"  ssim1={s1_add:.4f}, ssim2={s2_add:.4f}, mean={0.5*(s1_add+s2_add):.4f}")

s1_avg = ssim_fn(np.clip(out_avg, 0, 1), img1, data_range=1.0)
s2_avg = ssim_fn(np.clip(out_avg, 0, 1), img2, data_range=1.0)
print(f"Average rule: min={out_avg.min():.4f}, max={out_avg.max():.4f}, mean={out_avg.mean():.4f}")
print(f"  ssim1={s1_avg:.4f}, ssim2={s2_avg:.4f}, mean={0.5*(s1_avg+s2_avg):.4f}")

# What if weights are from DenseFuse addition or l1-norm?
# Where did densefuse_weights.pkl come from? Let's inspect densefuse_weights.pkl
w_path = "baselines/deep/densefuse_weights.pkl"
w = torch.load(w_path, map_location="cpu")
print("Weights keys:", list(w.keys()) if isinstance(w, dict) else type(w))

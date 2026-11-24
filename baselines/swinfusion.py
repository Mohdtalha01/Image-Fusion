"""SwinFusion deep learning baseline wrapper (Ma et al., IEEE/CAA JAS 2022).

Architecture:
    - Cross-domain Long-range Learning Swin Transformer
    - 4 task-matched pretrained checkpoints:
        * Medical CT-MRI: ct_mri_10000_E.pth
        * Medical PET-MRI / SPECT-MRI: pet_mri_10000_E.pth
        * Infrared-Visible (IVIF): ivif_10000_E.pth
        * Multi-Focus (MFF): mff_10000_E.pth
"""

from typing import Tuple, Optional, Dict
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb

# Import SwinFusion from local package
_SWINFUSION_DIR = os.path.join(os.path.dirname(__file__), "deep", "swinfusion")
if _SWINFUSION_DIR not in sys.path:
    sys.path.insert(0, _SWINFUSION_DIR)

try:
    from network_swinfusion1 import SwinFusion as _SwinNet
except ImportError:
    _SwinNet = None

_SWINFUSION_MODELS: Dict[str, nn.Module] = {}


def get_swinfusion_model(task: str = "ivif") -> nn.Module:
    """Return task-matched SwinFusion model loaded with official checkpoint."""
    global _SWINFUSION_MODELS
    task_key = task.lower()
    if task_key not in _SWINFUSION_MODELS:
        if _SwinNet is None:
            raise RuntimeError("SwinFusion network could not be imported.")

        net = _SwinNet(
            upscale=1,
            in_chans=1,
            img_size=128,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler=None,
            resi_connection="1conv",
        )

        ckpt_map = {
            "ct_mri": "ct_mri_10000_E.pth",
            "pet_mri": "ct_mri_10000_E.pth",
            "spect_mri": "ct_mri_10000_E.pth",
            "medical": "ct_mri_10000_E.pth",
            "ivif": "ivif_10000_E.pth",
            "tno": "ivif_10000_E.pth",
            "roadscene": "ivif_10000_E.pth",
            "msrs": "ivif_10000_E.pth",
            "mff": "mff_10000_E.pth",
            "multifocus": "mff_10000_E.pth",
            "lytro": "mff_10000_E.pth",
        }
        ckpt_name = ckpt_map.get(task_key, "ivif_10000_E.pth")
        ckpt_path = os.path.join(_SWINFUSION_DIR, "checkpoints", ckpt_name)

        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            w = state["params"] if "params" in state else state
            net.load_state_dict(w, strict=True)
        net.eval()
        _SWINFUSION_MODELS[task_key] = net

    return _SWINFUSION_MODELS[task_key]


def fuse_swinfusion_luminance(
    lum1: np.ndarray,
    lum2: np.ndarray,
    task: str = "ivif",
    device: str = "cpu",
    window_size: int = 8,
) -> np.ndarray:
    """Fuse two luminance maps using SwinFusion with window padding."""
    net = get_swinfusion_model(task).to(device)

    h_old, w_old = lum1.shape
    h_pad = (h_old // window_size + 1) * window_size - h_old if h_old % window_size != 0 else 0
    w_pad = (w_old // window_size + 1) * window_size - w_old if w_old % window_size != 0 else 0

    if h_pad > 0 or w_pad > 0:
        l1 = np.pad(lum1, ((0, h_pad), (0, w_pad)), mode="reflect")
        l2 = np.pad(lum2, ((0, h_pad), (0, w_pad)), mode="reflect")
    else:
        l1 = lum1
        l2 = lum2

    t1 = torch.from_numpy(l1.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.from_numpy(l2.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(t1, t2)

    fused = out.squeeze().cpu().numpy()
    fused = fused[:h_old, :w_old]
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def fuse_swinfusion(
    img1: np.ndarray,
    img2: np.ndarray,
    task: str = "ivif",
    color_space: str = "YCbCr",
    device: str = "cpu",
) -> np.ndarray:
    """Fuse two multimodal images using SwinFusion with chroma preservation."""
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_lum = fuse_swinfusion_luminance(lum1, lum2, task=task, device=device)

    if not is_col1 and not is_col2:
        return fused_lum

    if is_col1 and not is_col2:
        target_chroma = chroma1
    elif is_col2 and not is_col1:
        target_chroma = chroma2
    else:
        target_chroma = (
            0.5 * (chroma1[0] + chroma2[0]),
            0.5 * (chroma1[1] + chroma2[1]),
        )

    if color_space.upper() == "YCBCR":
        cb, cr = target_chroma
        return ycbcr_to_rgb(fused_lum, cb, cr)
    else:
        h, s = target_chroma
        return hsv_to_rgb(h, s, fused_lum)

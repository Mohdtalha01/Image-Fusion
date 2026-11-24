"""U2Fusion deep learning baseline wrapper (Xu et al., IEEE TPAMI 2022).

Architecture:
    - Unified Unsupervised DenseNet Architecture
    - Growth rate 44, 5 dense layers, concatenation-based fusion
    - 659,105 parameters
"""

from typing import Tuple, Optional
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb

# Import DenseNet from local u2fusion package
_U2FUSION_DIR = os.path.join(os.path.dirname(__file__), "deep", "u2fusion")
if _U2FUSION_DIR not in sys.path:
    sys.path.insert(0, _U2FUSION_DIR)

try:
    from model import DenseNet as _U2DenseNet
except ImportError:
    _U2DenseNet = None

_U2FUSION_MODEL: Optional[nn.Module] = None


def get_u2fusion_model() -> nn.Module:
    """Return singleton U2Fusion model loaded with pretrained weights."""
    global _U2FUSION_MODEL
    if _U2FUSION_MODEL is None:
        if _U2DenseNet is None:
            raise RuntimeError("U2Fusion model definition could not be imported.")
        net = _U2DenseNet()
        ckpt_path = os.path.join(_U2FUSION_DIR, "u2fusion_model.pth")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            w = state["model"] if "model" in state else state
            net.load_state_dict(w, strict=True)
        net.eval()
        _U2FUSION_MODEL = net
    return _U2FUSION_MODEL


def fuse_u2fusion_luminance(
    lum1: np.ndarray,
    lum2: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Fuse two grayscale luminance maps using U2Fusion on [0.0, 1.0] scale."""
    net = get_u2fusion_model().to(device)

    # U2Fusion expects input in [0, 1]
    t1 = torch.from_numpy(lum1.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.from_numpy(lum2.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(t1, t2)

    # U2Fusion outputs via Tanh in [-1, 1], properly clipped to [0, 1]
    fused = out.squeeze().cpu().numpy()
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def fuse_u2fusion(
    img1: np.ndarray,
    img2: np.ndarray,
    color_space: str = "YCbCr",
    device: str = "cpu",
) -> np.ndarray:
    """Fuse two multimodal images using U2Fusion with chroma preservation."""
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_lum = fuse_u2fusion_luminance(lum1, lum2, device=device)

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

"""CDDFuse deep learning baseline wrapper (Zhao et al., CVPR 2023).

Correlation-Driven Dual-Branch Feature Decomposition:
    - Restormer Encoder & Decoder
    - Base Feature Extraction (Transformer Cross-Attention)
    - Detail Feature Extraction (Lite Transformer / Invertible blocks)
    - Checkpoints:
        * IVF: baselines/deep/cddfuse/models/CDDFuse_IVF.pth (Infrared-Visible & Multi-focus)
        * MIF: baselines/deep/cddfuse/models/CDDFuse_MIF.pth (Medical CT/MRI, PET/MRI, SPECT/MRI)
"""

from typing import Tuple, Optional, Dict
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb

_CDDFUSE_DIR = os.path.join(os.path.dirname(__file__), "deep", "cddfuse")
if _CDDFUSE_DIR not in sys.path:
    sys.path.insert(0, _CDDFUSE_DIR)

try:
    from net import (
        Restormer_Encoder as _Encoder,
        Restormer_Decoder as _Decoder,
        BaseFeatureExtraction as _BaseFuse,
        DetailFeatureExtraction as _DetailFuse,
    )
except ImportError:
    _Encoder = None
    _Decoder = None
    _BaseFuse = None
    _DetailFuse = None

_CDDFUSE_MODELS: Dict[str, Tuple[nn.Module, nn.Module, nn.Module, nn.Module]] = {}


def _strip_module(state_dict: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state_dict.items() if not k.startswith("_")}


def get_cddfuse_model(task: str = "ivif") -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """Return task-matched CDDFuse model tuple (Encoder, Decoder, BaseFuse, DetailFuse)."""
    global _CDDFUSE_MODELS
    task_key = task.lower()
    is_medical = any(m in task_key for m in ["medical", "ct", "pet", "spect", "mri"])
    model_key = "mif" if is_medical else "ivif"

    if model_key not in _CDDFUSE_MODELS:
        if _Encoder is None:
            raise RuntimeError("CDDFuse network could not be imported.")

        encoder = _Encoder()
        decoder = _Decoder()
        base_fuse = _BaseFuse(dim=64, num_heads=8)
        detail_fuse = _DetailFuse(num_layers=1)

        ckpt_file = "CDDFuse_MIF.pth" if is_medical else "CDDFuse_IVF.pth"
        ckpt_path = os.path.join(_CDDFUSE_DIR, "models", ckpt_file)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"CDDFuse checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        encoder.load_state_dict(_strip_module(checkpoint["DIDF_Encoder"]))
        decoder.load_state_dict(_strip_module(checkpoint["DIDF_Decoder"]))
        base_fuse.load_state_dict(_strip_module(checkpoint["BaseFuseLayer"]))
        detail_fuse.load_state_dict(_strip_module(checkpoint["DetailFuseLayer"]))

        encoder.eval()
        decoder.eval()
        base_fuse.eval()
        detail_fuse.eval()

        _CDDFUSE_MODELS[model_key] = (encoder, decoder, base_fuse, detail_fuse)

    return _CDDFUSE_MODELS[model_key]


def fuse_cddfuse_luminance(
    lum1: np.ndarray,
    lum2: np.ndarray,
    task: str = "ivif",
    device: Optional[str] = None,
) -> np.ndarray:
    """Fuse two single-channel luminance arrays [0, 1] using CDDFuse.

    Handles spatial padding to make dimensions divisible by 8 (required by Restormer).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder, decoder, base_fuse, detail_fuse = get_cddfuse_model(task)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    base_fuse = base_fuse.to(device)
    detail_fuse = detail_fuse.to(device)

    h, w = lum1.shape[:2]
    # Restormer downsamples; pad to multiples of 8
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8

    if pad_h > 0 or pad_w > 0:
        l1 = np.pad(lum1, ((0, pad_h), (0, pad_w)), mode="reflect")
        l2 = np.pad(lum2, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        l1, l2 = lum1, lum2

    t1 = torch.from_numpy(l1).unsqueeze(0).unsqueeze(0).float().to(device)
    t2 = torch.from_numpy(l2).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        fb1, fd1, _ = encoder(t1)
        fb2, fd2, _ = encoder(t2)
        fb = base_fuse(fb1 + fb2)
        fd = detail_fuse(fd1 + fd2)
        out, _ = decoder(t1, fb, fd)

        d_min, d_max = torch.min(out), torch.max(out)
        out = (out - d_min) / (d_max - d_min).clamp_min(1e-8)
        fused = out.squeeze().cpu().numpy()

    if pad_h > 0 or pad_w > 0:
        fused = fused[:h, :w]

    return np.clip(fused, 0.0, 1.0)


def fuse_cddfuse(
    img1: np.ndarray,
    img2: np.ndarray,
    task: str = "ivif",
    color_space: str = "YCbCr",
    device: Optional[str] = None,
) -> np.ndarray:
    """Fuse two multimodal images using CDDFuse with chroma preservation.

    Input: [0, 1] or [0, 255] float/uint8 image arrays.
    Returns: float64 array [0, 1].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_l = fuse_cddfuse_luminance(lum1, lum2, task=task, device=device)

    if not is_col1 and not is_col2:
        return fused_l

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
        return ycbcr_to_rgb(fused_l, cb, cr)
    else:
        h, s = target_chroma
        return hsv_to_rgb(h, s, fused_l)

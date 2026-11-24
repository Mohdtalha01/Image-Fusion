"""DenseFuse deep learning baseline wrapper (Li & Wu, IEEE TIP 2019).

Architecture:
    - Autoencoder with Dense Blocks (residual dense connections)
    - Addition-based feature fusion layer
    - 73,985 trainable parameters (frozen at inference)
"""

from typing import Tuple, Optional
import os
import numpy as np
import torch
import torch.nn as nn

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb


class _DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.DenseConv1 = nn.Conv2d(16, 16, 3, 1, 1, padding_mode="reflect")
        self.DenseConv2 = nn.Conv2d(32, 16, 3, 1, 1, padding_mode="reflect")
        self.DenseConv3 = nn.Conv2d(48, 16, 3, 1, 1, padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        d1 = self.relu(self.DenseConv1(x))
        c1 = torch.cat([x, d1], dim=1)
        d2 = self.relu(self.DenseConv2(c1))
        c2 = torch.cat([c1, d2], dim=1)
        d3 = self.relu(self.DenseConv3(c2))
        c3 = torch.cat([c2, d3], dim=1)
        return c3


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(1, 16, 3, 1, 1, padding_mode="reflect")
        self.layers = _DenseBlock()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.Conv1(x))
        return self.layers(h)


class _DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2 = nn.Conv2d(64, 64, 3, 1, 1, padding_mode="reflect")
        self.Conv3 = nn.Conv2d(64, 32, 3, 1, 1, padding_mode="reflect")
        self.Conv4 = nn.Conv2d(32, 16, 3, 1, 1, padding_mode="reflect")
        self.Conv5 = nn.Conv2d(16, 1, 3, 1, 1, padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.Conv2(x))
        h = self.relu(self.Conv3(h))
        h = self.relu(self.Conv4(h))
        return self.Conv5(h)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = _DecoderBlock()

    def forward(self, x):
        return self.layers(x)


class DenseFuseNet(nn.Module):
    """Complete DenseFuse network with L1-norm or addition fusion rule (Li & Wu, TIP 2019)."""
    def __init__(self, strategy: str = "l1"):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.strategy = strategy

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, strategy: Optional[str] = None) -> torch.Tensor:
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)
        strat = strategy or self.strategy
        if strat == "l1":
            # Li & Wu (2019) L1-norm activity measure with 3x3 spatial average filter
            c1 = torch.sum(torch.abs(feat1), dim=1, keepdim=True)
            c2 = torch.sum(torch.abs(feat2), dim=1, keepdim=True)
            kernel = torch.ones(1, 1, 3, 3, device=feat1.device, dtype=feat1.dtype) / 9.0
            c1_bar = torch.nn.functional.conv2d(c1, kernel, padding=1)
            c2_bar = torch.nn.functional.conv2d(c2, kernel, padding=1)
            w1 = c1_bar / (c1_bar + c2_bar + 1e-8)
            w2 = c2_bar / (c1_bar + c2_bar + 1e-8)
            feat_fused = w1 * feat1 + w2 * feat2
        elif strat == "addition":
            feat_fused = 0.5 * (feat1 + feat2)
        else:
            raise ValueError(f"Unknown fusion strategy: {strat}")
        return self.decoder(feat_fused)


_DENSEFUSE_MODEL: Optional[DenseFuseNet] = None


def get_densefuse_model(strategy: str = "l1") -> DenseFuseNet:
    """Return singleton DenseFuse model loaded with official pretrained weights."""
    global _DENSEFUSE_MODEL
    if _DENSEFUSE_MODEL is None or _DENSEFUSE_MODEL.strategy != strategy:
        weights_path = os.path.join(os.path.dirname(__file__), "deep", "densefuse_weights.pkl")
        net = DenseFuseNet(strategy=strategy)
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            w = state["weight"] if "weight" in state else state
            net.load_state_dict(w, strict=True)
        net.eval()
        _DENSEFUSE_MODEL = net
    return _DENSEFUSE_MODEL


def fuse_densefuse_luminance(
    lum1: np.ndarray,
    lum2: np.ndarray,
    strategy: str = "l1",
    device: str = "cpu",
) -> np.ndarray:
    """Fuse two grayscale luminance maps using DenseFuse on [0.0, 1.0] scale."""
    net = get_densefuse_model(strategy=strategy).to(device)

    t1 = torch.from_numpy(lum1.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.from_numpy(lum2.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(t1, t2, strategy=strategy)

    fused = out.squeeze().cpu().numpy()
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def fuse_densefuse(
    img1: np.ndarray,
    img2: np.ndarray,
    strategy: str = "l1",
    color_space: str = "YCbCr",
    device: str = "cpu",
) -> np.ndarray:
    """Fuse two multimodal images using DenseFuse with chroma preservation."""
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, color_space)
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, color_space)

    fused_lum = fuse_densefuse_luminance(lum1, lum2, strategy=strategy, device=device)

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

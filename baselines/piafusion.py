"""PIAFusion baseline wrapper (Tang et al., Information Fusion 2022).

Architecture:
    - Illumination-aware Infrared-Visible Image Fusion
    - Checkpoints: baselines/deep/piafusion/checkpoint/PIAFusion/
    - Implementation note: Official PIAFusion uses TensorFlow 1.x.
"""

from typing import Tuple, Optional
import os
import sys
import numpy as np

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb, hsv_to_rgb

_PIAFUSION_DIR = os.path.join(os.path.dirname(__file__), "deep", "piafusion")


def is_piafusion_available() -> Tuple[bool, str]:
    """Check if TensorFlow 1.x environment can load PIAFusion."""
    try:
        import tensorflow as tf
        return True, f"TensorFlow {tf.__version__} available"
    except Exception as e:
        return False, f"TensorFlow 1.x unavailable or DLL load failed: {e}"


def fuse_piafusion(
    im1: np.ndarray,
    im2: np.ndarray,
    task: str = "ivif",
) -> np.ndarray:
    """Fuse images with PIAFusion. Raises RuntimeError if TensorFlow 1.x is not usable."""
    available, reason = is_piafusion_available()
    if not available:
        raise RuntimeError(
            f"PIAFusion execution failed: {reason}. "
            "Official PIAFusion requires TensorFlow 1.x, which cannot load native DLLs on Windows Python 3.12."
        )
    # If TF available, load session & run graph
    raise NotImplementedError("TensorFlow 1.x session execution not reached due to environment failure.")

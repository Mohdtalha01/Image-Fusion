"""Main pipeline orchestration for CADM-MSWN image fusion.

Handles color space conversions (YCbCr / HSV), luminance extraction,
end-to-end execution of clustering, decomposition, multi-structured
weighting networks, and chromatic reconstruction.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import cv2

from fusion.decomposition import decompose_structures, perona_malik_diffusion
from fusion.clustering import cluster_source_pair
from fusion.weighting import fuse_cluster_components, reconstruct_fused_image


def rgb_to_ycbcr(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB float image in [0, 1] to Y, Cb, Cr channels in [0, 1].

    Formula (ITU-R BT.601 standard):
        Y  =  0.299000 * R + 0.587000 * G + 0.114000 * B
        Cb = -0.168736 * R - 0.331264 * G + 0.500000 * B + 0.5
        Cr =  0.500000 * R - 0.418688 * G - 0.081312 * B + 0.5
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    y  =  0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 0.5
    cr =  0.500000 * r - 0.418688 * g - 0.081312 * b + 0.5

    return np.clip(y, 0.0, 1.0), np.clip(cb, 0.0, 1.0), np.clip(cr, 0.0, 1.0)


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Convert Y, Cb, Cr float channels in [0, 1] back to RGB in [0, 1].

    Formula:
        R = Y + 1.402000 * (Cr - 0.5)
        G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
        B = Y + 1.772000 * (Cb - 0.5)
    """
    cb_shift = cb - 0.5
    cr_shift = cr - 0.5

    r = y + 1.402000 * cr_shift
    g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
    b = y + 1.772000 * cb_shift

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB float image in [0, 1] to H, S, V channels."""
    rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[..., 0] / 179.0
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    return h, s, v


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Convert H, S, V channels back to RGB float image in [0, 1]."""
    h_cv = (np.clip(h, 0.0, 1.0) * 179.0).astype(np.uint8)
    s_cv = (np.clip(s, 0.0, 1.0) * 255.0).astype(np.uint8)
    v_cv = (np.clip(v, 0.0, 1.0) * 255.0).astype(np.uint8)
    hsv = np.stack([h_cv, s_cv, v_cv], axis=-1)
    rgb_uint8 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_uint8.astype(np.float32) / 255.0


def extract_luminance_and_chroma(
    image: np.ndarray,
    color_space: str = "YCbCr",
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]], bool]:
    """Extract luminance/intensity and preserve chroma if color.

    Args:
        image: Array of shape (H, W) or (H, W, 3) in [0, 1] or [0, 255].
        color_space: 'YCbCr' or 'HSV'.

    Returns:
        Tuple of (luminance_2d, chroma_tuple_or_None, is_color_boolean).
    """
    if image.dtype == np.uint8:
        img = image.astype(np.float32) * np.float32(1.0 / 255.0)
    else:
        img = np.asarray(image, dtype=np.float32)
        if float(np.max(img)) > 1.0:
            img = img * np.float32(1.0 / 255.0)

    if img.ndim == 2:
        return np.clip(img, 0.0, 1.0), None, False

    if img.ndim == 3 and img.shape[2] == 1:
        return np.clip(img[..., 0], 0.0, 1.0), None, False

    if img.ndim == 3 and img.shape[2] == 3:
        if color_space.upper() == "YCBCR":
            y, cb, cr = rgb_to_ycbcr(img)
            return y, (cb, cr), True
        elif color_space.upper() == "HSV":
            h, s, v = rgb_to_hsv(img)
            return v, (h, s), True
        else:
            raise ValueError(f"Unsupported color space: {color_space}")

    raise ValueError(f"Unsupported image shape: {image.shape}")


class CADMSWNFusion:
    """Clustered Anisotropic Diffusion Model and Multi-structured Weighted Networks.

    A training-free, deterministic multimodal image fusion architecture.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with configuration parameters."""
        cfg = config or {}
        color_cfg = cfg.get("color", {})
        clust_cfg = cfg.get("clustering", {})
        diff_cfg  = cfg.get("diffusion", {})
        weight_cfg = cfg.get("weighting", {})
        gf_cfg    = weight_cfg.get("guided_filter", {})
        recon_cfg = weight_cfg.get("reconstruction", {})

        self.color_space = color_cfg.get("color_space", "YCbCr")

        # Clustering hyperparameters
        self.n_clusters = clust_cfg.get("n_clusters", 3)
        self.cluster_method = clust_cfg.get("method", "kmeans")
        self.cluster_mode = clust_cfg.get("mode", "per_source")
        self.cluster_window = clust_cfg.get("window_size", 7)
        self.random_state = clust_cfg.get("random_state", 42)

        # Diffusion hyperparameters
        self.diff_iterations = diff_cfg.get("iterations", 20)
        self.diff_kappa = diff_cfg.get("kappa", 0.1176)
        self.diff_step_size = diff_cfg.get("step_size", 0.20)
        self.diff_conduction = diff_cfg.get("conduction", "exponential")
        self.signed_fe = recon_cfg.get("signed_fine_edge", False)

        # Weighting hyperparameters
        self.stat_window = weight_cfg.get("window_size", 7)
        self.weight_norm = weight_cfg.get("normalization", "linear")
        self.temperature = weight_cfg.get("temperature", 0.1)

        # Guided filter hyperparameters
        self.gf_enabled = gf_cfg.get("enabled", True)
        self.gf_radius = gf_cfg.get("radius", 5) if self.gf_enabled else 0
        self.gf_eps = gf_cfg.get("eps", 1e-4)

        # Reconstruction hyperparameters
        self.alpha = recon_cfg.get("alpha", 1.0)
        self.beta = recon_cfg.get("beta", 1.0)
        self.gamma = recon_cfg.get("gamma", 0.5)

    def fuse_luminance(
        self,
        lum1: np.ndarray,
        lum2: np.ndarray,
    ) -> np.ndarray:
        """Execute core CADM-MSWN fusion on two registered 2D luminance maps.

        Args:
            lum1: First luminance array (H, W) in [0, 1].
            lum2: Second luminance array (H, W) in [0, 1].

        Returns:
            Fused luminance array (H, W) in [0, 1].
        """
        # Step 1: Pixel Feature Extraction & Clustering (skipped when K == 1)
        if self.n_clusters == 1:
            masks1, masks2 = None, None
        else:
            masks1, masks2 = cluster_source_pair(
                lum1,
                lum2,
                n_clusters=self.n_clusters,
                method=self.cluster_method,
                mode=self.cluster_mode,
                window_size=self.cluster_window,
                random_state=self.random_state,
            )

        # Steps 2 & 3: Structural Decomposition (skip edge maps when gamma == 0.0)
        if self.gamma == 0.0:
            gt1 = perona_malik_diffusion(
                lum1,
                iterations=self.diff_iterations,
                kappa=self.diff_kappa,
                step_size=self.diff_step_size,
                conduction=self.diff_conduction,
            )
            gt2 = perona_malik_diffusion(
                lum2,
                iterations=self.diff_iterations,
                kappa=self.diff_kappa,
                step_size=self.diff_step_size,
                conduction=self.diff_conduction,
            )
            empty = np.empty((0, 0), dtype=np.float32)
            comp1 = (gt1, empty, lum1 - gt1, empty)
            comp2 = (gt2, empty, lum2 - gt2, empty)
            active_comps = ("GT", "FT")
        else:
            comp1 = decompose_structures(
                lum1,
                iterations=self.diff_iterations,
                kappa=self.diff_kappa,
                step_size=self.diff_step_size,
                conduction=self.diff_conduction,
                signed_fe=self.signed_fe,
            )
            comp2 = decompose_structures(
                lum2,
                iterations=self.diff_iterations,
                kappa=self.diff_kappa,
                step_size=self.diff_step_size,
                conduction=self.diff_conduction,
                signed_fe=self.signed_fe,
            )
            active_comps = ("GT", "GE", "FT", "FE")

        # Steps 4 & 5: Multi-Structured Weighted Fusion
        fused_gt, fused_ge, fused_ft, fused_fe = fuse_cluster_components(
            components1=comp1,
            components2=comp2,
            masks1=masks1,
            masks2=masks2,
            guide1=lum1,
            guide2=lum2,
            window_size=self.stat_window,
            normalization=self.weight_norm,
            temperature=self.temperature,
            guided_radius=self.gf_radius,
            guided_eps=self.gf_eps,
            active_components=active_comps,
        )

        # Composite Reconstruction
        fused_lum = reconstruct_fused_image(
            fused_gt=fused_gt,
            fused_ge=fused_ge,
            fused_ft=fused_ft,
            fused_fe=fused_fe,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

        return fused_lum

    def fuse(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> np.ndarray:
        """Fuse two images (grayscale or color).

        Preserves original chrominance channels to eliminate color distortion.

        Args:
            img1: First input image, shape (H, W) or (H, W, 3).
            img2: Second input image, shape (H, W) or (H, W, 3).

        Returns:
            Fused image of matching spatial dimension and appropriate channels in [0, 1].
        """
        if img1.shape[:2] != img2.shape[:2]:
            raise ValueError(
                f"Source images must have identical spatial dimensions. "
                f"Got img1: {img1.shape[:2]} and img2: {img2.shape[:2]}"
            )

        lum1, chroma1, is_col1 = extract_luminance_and_chroma(img1, self.color_space)
        lum2, chroma2, is_col2 = extract_luminance_and_chroma(img2, self.color_space)

        fused_lum = self.fuse_luminance(lum1, lum2)

        # Case 1: Both images are grayscale
        if not is_col1 and not is_col2:
            return fused_lum

        # Case 2: One image is color (e.g. PET or SPECT), one is grayscale (MRI or CT)
        if is_col1 and not is_col2:
            target_chroma = chroma1
        elif is_col2 and not is_col1:
            target_chroma = chroma2
        else:
            # Case 3: Both images are color (e.g. color multi-focus or color IR-VIS)
            # Average chrominance components
            target_chroma = (
                0.5 * (chroma1[0] + chroma2[0]),
                0.5 * (chroma1[1] + chroma2[1]),
            )

        if self.color_space.upper() == "YCBCR":
            cb, cr = target_chroma
            fused_rgb = ycbcr_to_rgb(fused_lum, cb, cr)
        else:  # HSV
            h, s = target_chroma
            fused_rgb = hsv_to_rgb(h, s, fused_lum)

        return fused_rgb


def fuse_images(
    img1: np.ndarray,
    img2: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Convenience functional wrapper for CADM-MSWN fusion.

    Args:
        img1: First source image.
        img2: Second source image.
        config: Optional configuration dictionary.

    Returns:
        Fused image in [0, 1].
    """
    pipeline = CADMSWNFusion(config=config)
    return pipeline.fuse(img1, img2)

"""CADM-MSWN: Clustered Anisotropic Diffusion Model and Multi-structured Weighted Networks.

A training-free multimodal image fusion package.
"""

from fusion.decomposition import (
    perona_malik_diffusion,
    compute_gradient_magnitude,
    decompose_structures,
)
from fusion.clustering import (
    extract_pixel_features,
    generate_cluster_masks,
    cluster_source_pair,
)
from fusion.weighting import (
    compute_component_saliency,
    compute_inter_source_weights,
    fast_guided_filter,
    fuse_cluster_components,
    reconstruct_fused_image,
)
from fusion.fuse import (
    CADMSWNFusion,
    fuse_images,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    rgb_to_hsv,
    hsv_to_rgb,
)

__all__ = [
    "perona_malik_diffusion",
    "compute_gradient_magnitude",
    "decompose_structures",
    "extract_pixel_features",
    "generate_cluster_masks",
    "cluster_source_pair",
    "compute_component_saliency",
    "compute_inter_source_weights",
    "fast_guided_filter",
    "fuse_cluster_components",
    "reconstruct_fused_image",
    "CADMSWNFusion",
    "fuse_images",
    "rgb_to_ycbcr",
    "ycbcr_to_rgb",
    "rgb_to_hsv",
    "hsv_to_rgb",
]

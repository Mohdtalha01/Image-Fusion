"""Baseline registry for classical and deep learning multimodal image fusion methods."""

from baselines.averaging import fuse_averaging
from baselines.dwt import fuse_dwt
from baselines.gff import fuse_gff
from baselines.adf import fuse_adf
from baselines.cbf import fuse_cbf
from baselines.nsct_substitute import (
    fuse_nsct_substitute,
    fuse_nsct_substitute as fuse_swt_nsct_substitute,
)
from baselines.densefuse import fuse_densefuse
from baselines.u2fusion import fuse_u2fusion
from baselines.swinfusion import fuse_swinfusion
from baselines.cddfuse import fuse_cddfuse
from baselines.piafusion import fuse_piafusion

__all__ = [
    "fuse_averaging",
    "fuse_dwt",
    "fuse_gff",
    "fuse_adf",
    "fuse_cbf",
    "fuse_nsct_substitute",
    "fuse_swt_nsct_substitute",
    "fuse_densefuse",
    "fuse_u2fusion",
    "fuse_swinfusion",
    "fuse_cddfuse",
    "fuse_piafusion",
]

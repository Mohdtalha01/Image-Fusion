"""Decomposition module for CADM-MSWN image fusion.

Implements Perona-Malik Anisotropic Diffusion (PMAD) with exponential (g1)
and rational (g2) conduction functions, gradient computation, and multiscale
structural decomposition into Gross Texture (GT), Gross Edge (GE),
Fine Texture (FT), and Fine Edge (FE).
"""

import cv2
from typing import Tuple, Literal
import numpy as np


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel operators.

    Vectorized float32 implementation with reflect boundary conditions.

    Args:
        image: 2D numpy array of shape (H, W) normalized to [0, 1].

    Returns:
        Gradient magnitude array of shape (H, W), normalized to [0, 1].
    """
    img_32 = np.asarray(image, dtype=np.float32)
    gx = cv2.Sobel(img_32, cv2.CV_32F, 1, 0, ksize=3, scale=0.125, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(img_32, cv2.CV_32F, 0, 1, ksize=3, scale=0.125, borderType=cv2.BORDER_REFLECT)
    magnitude = cv2.magnitude(gx, gy)
    max_val = float(np.max(magnitude))
    if max_val > 1e-8:
        magnitude *= np.float32(1.0 / max_val)
    return magnitude if image.dtype == np.float32 else magnitude.astype(image.dtype)


def perona_malik_diffusion(
    image: np.ndarray,
    iterations: int = 20,
    kappa: float = 0.1176,
    step_size: float = 0.20,
    conduction: Literal["exponential", "rational"] = "exponential",
) -> np.ndarray:
    """Perform Perona-Malik anisotropic diffusion on a 2D image.

    Solves the PDE:
        dI/dt = div( g(|grad I|) grad I )
    using a 4-neighborhood discretization with Neumann boundary conditions.
    Exploits directional flux antisymmetry (flux_S = -flux_N, flux_W = -flux_E)
    with preallocated float32 buffers and SIMD-accelerated cv2.exp.

    Conduction functions:
        - exponential (g1): g(s) = exp(-(s / kappa)^2)
        - rational (g2):    g(s) = 1.0 / (1.0 + (s / kappa)^2)

    Args:
        image: 2D float array of shape (H, W) in range [0, 1].
        iterations: Number of diffusion iterations T.
        kappa: Conduction threshold parameter.
        step_size: Time integration step lambda (must be <= 0.25 for stability).
        conduction: Choice of conduction function ('exponential' or 'rational').

    Returns:
        Diffused 2D array of shape (H, W) in range [0, 1].
    """
    if step_size > 0.25:
        raise ValueError(f"Step size {step_size} exceeds numerical stability bound (<= 0.25).")
    if iterations < 0:
        raise ValueError("Iterations must be non-negative.")
    if kappa <= 0:
        raise ValueError("Kappa must be strictly positive.")
    if conduction not in ("exponential", "rational"):
        raise ValueError(f"Unknown conduction function: {conduction}")

    u = np.array(image, dtype=np.float32, copy=True)
    if iterations == 0:
        np.clip(u, 0.0, 1.0, out=u)
        return u if image.dtype == np.float32 else u.astype(image.dtype)

    h, w = u.shape
    inv_kappa_sq = np.float32(1.0 / (kappa * kappa))
    neg_inv_kappa_sq = np.float32(-inv_kappa_sq)
    step = np.float32(step_size)

    dv = np.empty((max(h - 1, 0), w), dtype=np.float32)
    dh = np.empty((h, max(w - 1, 0)), dtype=np.float32)
    tmp_v = np.empty_like(dv)
    tmp_h = np.empty_like(dh)

    for _ in range(iterations):
        if h > 1:
            np.subtract(u[1:, :], u[:-1, :], out=dv)
            np.multiply(dv, dv, out=tmp_v)
            if conduction == "exponential":
                tmp_v *= neg_inv_kappa_sq
                cv2.exp(tmp_v, tmp_v)
            else:
                tmp_v *= inv_kappa_sq
                tmp_v += np.float32(1.0)
                np.divide(np.float32(1.0), tmp_v, out=tmp_v)
            tmp_v *= dv
            tmp_v *= step

        if w > 1:
            np.subtract(u[:, 1:], u[:, :-1], out=dh)
            np.multiply(dh, dh, out=tmp_h)
            if conduction == "exponential":
                tmp_h *= neg_inv_kappa_sq
                cv2.exp(tmp_h, tmp_h)
            else:
                tmp_h *= inv_kappa_sq
                tmp_h += np.float32(1.0)
                np.divide(np.float32(1.0), tmp_h, out=tmp_h)
            tmp_h *= dh
            tmp_h *= step

        if h > 1:
            u[:-1, :] += tmp_v
            u[1:, :] -= tmp_v
        if w > 1:
            u[:, :-1] += tmp_h
            u[:, 1:] -= tmp_h

    np.clip(u, 0.0, 1.0, out=u)
    return u if image.dtype == np.float32 else u.astype(image.dtype)


def decompose_structures(
    image: np.ndarray,
    iterations: int = 20,
    kappa: float = 0.1176,
    step_size: float = 0.20,
    conduction: Literal["exponential", "rational"] = "exponential",
    signed_fe: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose an image into Gross and Fine structural components.

    Formulations:
        Gross Texture: GT = PMAD(I)
        Gross Edge:    GE = |grad(GT)|
        Fine Texture:  FT = I - GT
        Fine Edge:     FE = |grad(I)| - GE (rectified to >=0 if not signed_fe)

    Args:
        image: 2D numpy array of shape (H, W) in [0, 1].
        iterations: Diffusion iterations.
        kappa: Conduction threshold.
        step_size: Time step lambda.
        conduction: Conduction function.
        signed_fe: If True, keep signed edge differences; if False, clip at 0.

    Returns:
        Tuple of (GT, GE, FT, FE).
    """
    gt = perona_malik_diffusion(
        image,
        iterations=iterations,
        kappa=kappa,
        step_size=step_size,
        conduction=conduction,
    )
    ge = compute_gradient_magnitude(gt)
    ft = image - gt

    grad_i = compute_gradient_magnitude(image)
    fe_diff = grad_i - ge
    if not signed_fe:
        fe = np.maximum(0.0, fe_diff)
    else:
        fe = fe_diff

    return gt, ge, ft, fe

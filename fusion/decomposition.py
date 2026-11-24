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
    img_32 = image.astype(np.float32)
    gx = cv2.Sobel(img_32, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT) / 8.0
    gy = cv2.Sobel(img_32, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT) / 8.0
    magnitude = np.sqrt(gx**2 + gy**2)
    max_val = np.max(magnitude)
    if max_val > 1e-8:
        magnitude = magnitude / max_val
    return magnitude.astype(image.dtype)


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
    Optimized with preallocated float32 buffers and vectorized difference stencils.

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

    u = image.astype(np.float32, copy=True)
    kappa_sq = np.float32(kappa**2)
    inv_kappa = np.float32(1.0 / kappa)
    step = np.float32(step_size)

    delta_n = np.zeros_like(u)
    delta_s = np.zeros_like(u)
    delta_e = np.zeros_like(u)
    delta_w = np.zeros_like(u)

    for _ in range(iterations):
        delta_n[:-1, :] = u[1:, :] - u[:-1, :]
        delta_n[-1, :] = 0.0
        delta_s[1:, :] = u[:-1, :] - u[1:, :]
        delta_s[0, :] = 0.0
        delta_e[:, :-1] = u[:, 1:] - u[:, :-1]
        delta_e[:, -1] = 0.0
        delta_w[:, 1:] = u[:, :-1] - u[:, 1:]
        delta_w[:, 0] = 0.0

        if conduction == "exponential":
            c_n = np.exp(-((delta_n * inv_kappa) ** 2))
            c_s = np.exp(-((delta_s * inv_kappa) ** 2))
            c_e = np.exp(-((delta_e * inv_kappa) ** 2))
            c_w = np.exp(-((delta_w * inv_kappa) ** 2))
        elif conduction == "rational":
            c_n = 1.0 / (1.0 + (delta_n**2) / kappa_sq)
            c_s = 1.0 / (1.0 + (delta_s**2) / kappa_sq)
            c_e = 1.0 / (1.0 + (delta_e**2) / kappa_sq)
            c_w = 1.0 / (1.0 + (delta_w**2) / kappa_sq)
        else:
            raise ValueError(f"Unknown conduction function: {conduction}")

        flux = c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w
        u += step * flux

    return np.clip(u, 0.0, 1.0).astype(image.dtype)


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

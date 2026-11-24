"""Feature and statistical quality metrics for image fusion.

Implements:
    - SD: Standard Deviation (global contrast)
    - SF: Spatial Frequency (texture rate of change)
    - AG: Average Gradient (edge clarity)
"""

import numpy as np
import cv2


def _to_luminance(image: np.ndarray) -> np.ndarray:
    """Extract 2D float luminance in [0, 1]."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def compute_standard_deviation(image: np.ndarray) -> float:
    """Compute Standard Deviation (SD) of pixel intensities.

    Measures global scene contrast and dispersion.

    Args:
        image: Image array.

    Returns:
        Standard deviation value in [0, 1].
    """
    lum = _to_luminance(image)
    return round(float(np.std(lum)), 4)


def compute_spatial_frequency(image: np.ndarray) -> float:
    """Compute Spatial Frequency (SF = sqrt(RF^2 + CF^2)).

    Measures overall spatial texture activity and sharpness.

    Formula:
        RF = sqrt( 1/HW * sum (F(x, y) - F(x-1, y))^2 )
        CF = sqrt( 1/HW * sum (F(x, y) - F(x, y-1))^2 )
        SF = sqrt( RF^2 + CF^2 )

    Args:
        image: Image array.

    Returns:
        Spatial frequency value.
    """
    lum = _to_luminance(image)
    diff_r = lum[1:, :] - lum[:-1, :]
    diff_c = lum[:, 1:] - lum[:, :-1]

    rf_sq = np.mean(diff_r**2)
    cf_sq = np.mean(diff_c**2)

    sf = np.sqrt(rf_sq + cf_sq)
    return round(float(sf), 4)


def compute_average_gradient(image: np.ndarray) -> float:
    """Compute Average Gradient (AG).

    Measures the clarity of minute details and high-frequency edge transitions.

    Formula:
        AG = 1/((H-1)(W-1)) * sum sqrt( ((dF/dx)^2 + (dF/dy)^2) / 2 )

    Args:
        image: Image array.

    Returns:
        Average gradient value.
    """
    lum = _to_luminance(image)
    gx = cv2.Sobel(lum, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(lum, cv2.CV_64F, 0, 1, ksize=3) / 8.0

    grad_mag = np.sqrt((gx**2 + gy**2) / 2.0)
    ag = np.mean(grad_mag)
    return round(float(ag), 4)

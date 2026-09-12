"""Texture Feature Extraction Module.

Calculates Gray-Level Co-occurrence Matrix (GLCM) texture descriptors and Local Binary Pattern
(LBP) histogram metrics restricted to the predicted polyp Region of Interest.
"""

from typing import Dict, Tuple
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_texture_features(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    distances: Tuple[int, ...] = (1, 3, 5),
    angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    lbp_points: int = 24,
    lbp_radius: int = 3,
) -> Dict[str, float]:
    """Extracts GLCM texture properties and LBP histogram features from predicted polyp ROI.

    Args:
        roi_rgb: (h_roi, w_roi, 3) uint8 cropped RGB frame.
        roi_mask: (h_roi, w_roi) uint8 cropped binary mask.
        distances: Sequence of pixel distance offsets for GLCM computation.
        angles: Sequence of angles in radians for GLCM computation.
        lbp_points: Number of circular neighbor set points for LBP.
        lbp_radius: Radius of circle for LBP.

    Returns:
        Dict mapping texture feature names to float values.
    """
    default_features = {
        "texture_glcm_contrast": 0.0,
        "texture_glcm_dissimilarity": 0.0,
        "texture_glcm_homogeneity": 0.0,
        "texture_glcm_energy": 0.0,
        "texture_glcm_correlation": 0.0,
        "texture_glcm_asm": 0.0,
        "texture_lbp_mean": 0.0,
        "texture_lbp_std": 0.0,
        "texture_lbp_energy": 0.0,
        "texture_lbp_entropy": 0.0,
    }

    if (
        roi_rgb is None
        or roi_mask is None
        or roi_mask.size == 0
        or np.count_nonzero(roi_mask) == 0
        or roi_rgb.shape[0] < 4
        or roi_rgb.shape[1] < 4
    ):
        return default_features

    # 1. Convert ROI to Grayscale
    if roi_rgb.ndim == 3:
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    else:
        roi_gray = roi_rgb.copy()

    cur_h, cur_w = roi_gray.shape
    max_tex_dim = 256
    if max(cur_h, cur_w) > max_tex_dim:
        t_scale = max_tex_dim / float(max(cur_h, cur_w))
        t_w, t_h = max(1, int(cur_w * t_scale)), max(1, int(cur_h * t_scale))
        roi_gray = cv2.resize(roi_gray, (t_w, t_h), interpolation=cv2.INTER_AREA)
        proc_mask = cv2.resize(roi_mask, (t_w, t_h), interpolation=cv2.INTER_NEAREST)
    else:
        proc_mask = roi_mask

    # Mask out non-polyp pixels in ROI (set background to zero)
    masked_gray = cv2.bitwise_and(roi_gray, roi_gray, mask=proc_mask)

    # 2. GLCM Feature Extraction
    # Reduce gray levels to 32 bins to speed up GLCM and prevent empty matrices
    gray_levels = 32
    quantized_gray = (masked_gray // (256 // gray_levels)).astype(np.uint8)

    try:
        glcm = graycomatrix(
            quantized_gray,
            distances=list(distances),
            angles=list(angles),
            levels=gray_levels,
            symmetric=True,
            normed=True,
        )

        glcm_contrast = float(np.mean(graycoprops(glcm, "contrast")))
        glcm_dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
        glcm_homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        glcm_energy = float(np.mean(graycoprops(glcm, "energy")))
        glcm_correlation = float(np.mean(graycoprops(glcm, "correlation")))
        glcm_asm = float(np.mean(graycoprops(glcm, "ASM")))

    except Exception:
        glcm_contrast = 0.0
        glcm_dissimilarity = 0.0
        glcm_homogeneity = 0.0
        glcm_energy = 0.0
        glcm_correlation = 0.0
        glcm_asm = 0.0

    # 3. LBP Feature Extraction
    try:
        lbp = local_binary_pattern(
            roi_gray, P=lbp_points, R=lbp_radius, method="uniform"
        )
        polyp_lbp = lbp[proc_mask > 0]

        if len(polyp_lbp) > 0:
            n_bins = int(lbp_points + 2)
            hist, _ = np.histogram(
                polyp_lbp, bins=n_bins, range=(0, n_bins), density=True
            )
            lbp_mean = float(np.mean(polyp_lbp))
            lbp_std = float(np.std(polyp_lbp))
            lbp_energy = float(np.sum(hist**2))
            # Calculate LBP entropy safely
            non_zero_hist = hist[hist > 0]
            lbp_entropy = float(-np.sum(non_zero_hist * np.log2(non_zero_hist)))
        else:
            lbp_mean = 0.0
            lbp_std = 0.0
            lbp_energy = 0.0
            lbp_entropy = 0.0

    except Exception:
        lbp_mean = 0.0
        lbp_std = 0.0
        lbp_energy = 0.0
        lbp_entropy = 0.0

    features = {
        "texture_glcm_contrast": glcm_contrast,
        "texture_glcm_dissimilarity": glcm_dissimilarity,
        "texture_glcm_homogeneity": glcm_homogeneity,
        "texture_glcm_energy": glcm_energy,
        "texture_glcm_correlation": glcm_correlation,
        "texture_glcm_asm": glcm_asm,
        "texture_lbp_mean": lbp_mean,
        "texture_lbp_std": lbp_std,
        "texture_lbp_energy": lbp_energy,
        "texture_lbp_entropy": lbp_entropy,
    }

    # Ensure all feature values are clean finite numbers
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features

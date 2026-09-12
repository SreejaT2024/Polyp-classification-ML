"""Color Feature Extraction Module.

Extracts statistical color descriptors (mean, standard deviation) across RGB, HSV,
and LAB color spaces exclusively from pixels within the predicted polyp region.
"""

from typing import Dict
import cv2
import numpy as np


def extract_color_features(rgb_image: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """Extracts first- and second-order color statistics from RGB, HSV, and LAB spaces.

    Args:
        rgb_image: (H, W, 3) uint8 original RGB endoscopy frame.
        pred_mask: (H, W) uint8 binary predicted segmentation mask.

    Returns:
        Dict mapping color feature names to float values.
    """
    default_features = {
        "color_rgb_r_mean": 0.0,
        "color_rgb_r_std": 0.0,
        "color_rgb_g_mean": 0.0,
        "color_rgb_g_std": 0.0,
        "color_rgb_b_mean": 0.0,
        "color_rgb_b_std": 0.0,
        "color_hsv_h_mean": 0.0,
        "color_hsv_h_std": 0.0,
        "color_hsv_s_mean": 0.0,
        "color_hsv_s_std": 0.0,
        "color_hsv_v_mean": 0.0,
        "color_hsv_v_std": 0.0,
        "color_lab_l_mean": 0.0,
        "color_lab_l_std": 0.0,
        "color_lab_a_mean": 0.0,
        "color_lab_a_std": 0.0,
        "color_lab_b_mean": 0.0,
        "color_lab_b_std": 0.0,
    }

    if (
        rgb_image is None
        or pred_mask is None
        or pred_mask.size == 0
        or np.count_nonzero(pred_mask) == 0
    ):
        return default_features

    valid_pixel_indices = pred_mask > 0

    # 1. Extract RGB pixels inside polyp mask
    rgb_polyp_pixels = rgb_image[valid_pixel_indices]  # Shape: (N, 3)

    if len(rgb_polyp_pixels) == 0:
        return default_features

    r_vals = rgb_polyp_pixels[:, 0].astype(np.float64)
    g_vals = rgb_polyp_pixels[:, 1].astype(np.float64)
    b_vals = rgb_polyp_pixels[:, 2].astype(np.float64)

    # 2. Extract HSV pixels
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_polyp_pixels = hsv_image[valid_pixel_indices]
    h_vals = hsv_polyp_pixels[:, 0].astype(np.float64)
    s_vals = hsv_polyp_pixels[:, 1].astype(np.float64)
    v_vals = hsv_polyp_pixels[:, 2].astype(np.float64)

    # 3. Extract LAB pixels
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    lab_polyp_pixels = lab_image[valid_pixel_indices]
    l_vals = lab_polyp_pixels[:, 0].astype(np.float64)
    a_vals = lab_polyp_pixels[:, 1].astype(np.float64)
    lab_b_vals = lab_polyp_pixels[:, 2].astype(np.float64)

    features = {
        "color_rgb_r_mean": float(np.mean(r_vals)),
        "color_rgb_r_std": float(np.std(r_vals)),
        "color_rgb_g_mean": float(np.mean(g_vals)),
        "color_rgb_g_std": float(np.std(g_vals)),
        "color_rgb_b_mean": float(np.mean(b_vals)),
        "color_rgb_b_std": float(np.std(b_vals)),
        "color_hsv_h_mean": float(np.mean(h_vals)),
        "color_hsv_h_std": float(np.std(h_vals)),
        "color_hsv_s_mean": float(np.mean(s_vals)),
        "color_hsv_s_std": float(np.std(s_vals)),
        "color_hsv_v_mean": float(np.mean(v_vals)),
        "color_hsv_v_std": float(np.std(v_vals)),
        "color_lab_l_mean": float(np.mean(l_vals)),
        "color_lab_l_std": float(np.std(l_vals)),
        "color_lab_a_mean": float(np.mean(a_vals)),
        "color_lab_a_std": float(np.std(a_vals)),
        "color_lab_b_mean": float(np.mean(lab_b_vals)),
        "color_lab_b_std": float(np.std(lab_b_vals)),
    }

    # Ensure finite values
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features

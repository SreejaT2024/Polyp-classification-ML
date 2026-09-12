"""Shape Feature Extraction Module.

Extracts geometric and topological shape descriptors from predicted polyp binary masks.
"""

from typing import Dict
import cv2
import numpy as np
from skimage.measure import regionprops, label


def extract_shape_features(pred_mask: np.ndarray) -> Dict[str, float]:
    """Calculates geometric shape feature descriptors from a predicted binary polyp mask.

    Args:
        pred_mask: (H, W) uint8 binary NumPy array (values 0 or 255).

    Returns:
        Dict mapping feature names to float values.
    """
    default_features = {
        "shape_area": 0.0,
        "shape_perimeter": 0.0,
        "shape_circularity": 0.0,
        "shape_aspect_ratio": 0.0,
        "shape_extent": 0.0,
        "shape_solidity": 0.0,
        "shape_eccentricity": 0.0,
        "shape_major_axis_length": 0.0,
        "shape_minor_axis_length": 0.0,
    }

    if pred_mask is None or pred_mask.size == 0 or np.count_nonzero(pred_mask) == 0:
        return default_features

    bin_mask = (pred_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return default_features

    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))

    if area == 0.0:
        return default_features

    perimeter = float(cv2.arcLength(largest_contour, True))
    circularity = (4.0 * np.pi * area) / ((perimeter**2) + 1e-6)

    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w / h) if h > 0 else 0.0
    bounding_box_area = float(w * h)
    extent = area / bounding_box_area if bounding_box_area > 0 else 0.0

    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Extract skimage regionprops for eccentricity and axis lengths
    labeled_mask = label(bin_mask)
    props = regionprops(labeled_mask)

    eccentricity = 0.0
    major_axis_length = 0.0
    minor_axis_length = 0.0

    if props:
        # Select largest region
        largest_prop = max(props, key=lambda r: r.area)
        eccentricity = float(getattr(largest_prop, "eccentricity", 0.0))
        if hasattr(largest_prop, "axis_major_length"):
            major_axis_length = float(largest_prop.axis_major_length)
        else:
            major_axis_length = float(getattr(largest_prop, "major_axis_length", 0.0))
        if hasattr(largest_prop, "axis_minor_length"):
            minor_axis_length = float(largest_prop.axis_minor_length)
        else:
            minor_axis_length = float(getattr(largest_prop, "minor_axis_length", 0.0))

    features = {
        "shape_area": area,
        "shape_perimeter": perimeter,
        "shape_circularity": float(circularity),
        "shape_aspect_ratio": aspect_ratio,
        "shape_extent": float(extent),
        "shape_solidity": float(solidity),
        "shape_eccentricity": eccentricity,
        "shape_major_axis_length": major_axis_length,
        "shape_minor_axis_length": minor_axis_length,
    }

    # Ensure all values are finite float numbers
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features

"""Polyp Region of Interest (ROI) Extractor.

Extracts bounding box ROI and masked RGB polyp regions exclusively from the
predicted active-contour segmentation mask without referencing ground-truth masks.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class ROIExtractor:
    """Extracts bounding box crops and masked polyp regions from predicted masks."""

    def __init__(self, min_area_pixels: int = 10) -> None:
        """Initializes ROIExtractor.

        Args:
            min_area_pixels: Minimum pixel area threshold for a valid connected component.
        """
        self.min_area_pixels = min_area_pixels

    def extract_roi(
        self, rgb_image: np.ndarray, pred_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int], bool]:
        """Extracts ROI cropped image, binary ROI mask, masked RGB ROI, and bounding box tuple.

        Args:
            rgb_image: (H, W, 3) uint8 original RGB endoscopy frame.
            pred_mask: (H, W) uint8 binary predicted segmentation mask (values 0 or 255).

        Returns:
            Tuple of:
                - roi_rgb: (h_roi, w_roi, 3) cropped RGB frame.
                - roi_mask: (h_roi, w_roi) cropped binary mask.
                - roi_masked_rgb: (h_roi, w_roi, 3) RGB frame with non-polyp pixels zeroed out.
                - bbox: Tuple (x, y, width, height).
                - valid_roi: bool indicating whether valid non-empty ROI was extracted.
        """
        if rgb_image.shape[:2] != pred_mask.shape:
            raise ValueError(
                f"Dimension mismatch between RGB image {rgb_image.shape} and mask {pred_mask.shape}."
            )

        h, w = pred_mask.shape

        # Find connected components in predicted mask
        contours, _ = cv2.findContours(
            pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # Fallback for empty mask: full image frame
            bbox = (0, 0, w, h)
            return (
                rgb_image.copy(),
                pred_mask.copy(),
                np.zeros_like(rgb_image),
                bbox,
                False,
            )

        # Select largest connected component
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.min_area_pixels:
            bbox = (0, 0, w, h)
            return (
                rgb_image.copy(),
                pred_mask.copy(),
                np.zeros_like(rgb_image),
                bbox,
                False,
            )

        # Calculate bounding box (x, y, width, height)
        x, y, bw, bh = cv2.boundingRect(largest_contour)

        # Ensure valid non-zero crop bounds
        x = max(0, x)
        y = max(0, y)
        bw = min(w - x, bw)
        bh = min(h - y, bh)

        # Crop RGB frame and mask
        roi_rgb = rgb_image[y : y + bh, x : x + bw].copy()
        roi_mask = pred_mask[y : y + bh, x : x + bw].copy()

        # Masked RGB ROI (exclude pixels outside predicted polyp mask)
        roi_masked_rgb = cv2.bitwise_and(roi_rgb, roi_rgb, mask=roi_mask)

        bbox = (x, y, bw, bh)
        return roi_rgb, roi_mask, roi_masked_rgb, bbox, True

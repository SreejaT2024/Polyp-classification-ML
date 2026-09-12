"""Image Preprocessing Module for Active Contour Polyp Segmentation.

Applies contrast enhancement (CLAHE) and noise reduction (Gaussian smoothing)
to prepare endoscopic frames for boundary delineation while preserving the raw RGB image.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class PolypImagePreprocessor:
    """Preprocessor for endoscopic RGB images prior to active contour segmentation."""

    def __init__(
        self,
        gaussian_kernel_size: Tuple[int, int] = (5, 5),
        gaussian_sigma: float = 1.0,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        """Initializes preprocessing hyperparameters.

        Args:
            gaussian_kernel_size: Tuple (width, height) for Gaussian kernel blur.
            gaussian_sigma: Gaussian blur standard deviation.
            clahe_clip_limit: Threshold clip limit for CLAHE contrast enhancement.
            clahe_tile_grid_size: Size of grid for CLAHE histogram equalization.
        """
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

    def preprocess_for_segmentation(self, rgb_image: np.ndarray) -> np.ndarray:
        """Processes an RGB endoscopy frame to produce a clean 2D representation for active contours.

        Args:
            rgb_image: (H, W, 3) uint8 NumPy array in RGB color space.

        Returns:
            (H, W) uint8 NumPy array optimized for active contour segmentation.
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image array, got shape {rgb_image.shape}.")

        # 1. Convert to Grayscale (or Green channel/LAB L-channel where polyp contrast is strong)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # 2. Apply Mild Gaussian Smoothing for noise reduction
        blurred = cv2.GaussianBlur(
            gray, self.gaussian_kernel_size, sigmaX=self.gaussian_sigma
        )

        # 3. Apply CLAHE for local contrast and boundary enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size
        )
        enhanced = clahe.apply(blurred)

        return enhanced

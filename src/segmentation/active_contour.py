"""Active Contour Segmentation Module for Polyp Boundary Delineation.

Implements Morphological Chan-Vese Active Contour Models without Edges (ACWE)
for automated delineation of endoscopic polyp boundaries without using ground-truth masks.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from skimage.segmentation import morphological_chan_vese, disk_level_set
import cv2

from config.config import GlobalConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("active_contour")


class ActiveContourSegmenter:
    """Active contour segmenter utilizing Morphological Chan-Vese formulation."""

    def __init__(
        self,
        config: Optional[GlobalConfig] = None,
        max_iterations: int = 80,
        smoothing: int = 1,
        lambda1: float = 1.0,
        lambda2: float = 1.5,
        init_radius_ratio: float = 0.15,
        init_type: str = "otsu_seed",
    ) -> None:
        """Initializes ActiveContourSegmenter with configurable parameters.

        Args:
            config: Optional GlobalConfig instance.
            max_iterations: Number of active contour evolution iterations.
            smoothing: Morphological smoothing iterations per step.
            lambda1: Weight parameter for inside region energy term.
            lambda2: Weight parameter for outside region energy term (penalizes over-expansion).
            init_radius_ratio: Ratio of image dimensions for fallback central seed circle.
            init_type: Initialization strategy ('otsu_seed' or 'center_disk').
        """
        self.config = config or get_config()
        seg_cfg = getattr(self.config, "segmentation", None)

        self.max_iterations = seg_cfg.max_iterations if seg_cfg else max_iterations
        self.smoothing = seg_cfg.smoothing if seg_cfg else smoothing
        self.lambda1 = seg_cfg.lambda1 if seg_cfg else lambda1
        self.lambda2 = seg_cfg.lambda2 if seg_cfg else lambda2
        self.init_radius_ratio = seg_cfg.init_radius_ratio if seg_cfg else init_radius_ratio
        self.init_type = seg_cfg.init_type if seg_cfg else init_type

    @staticmethod
    def get_black_border_mask(rgb_image: np.ndarray) -> np.ndarray:
        """Generates a binary mask excluding black endoscopy frame corners/vignette.

        Args:
            rgb_image: (H, W, 3) uint8 image.

        Returns:
            (H, W) uint8 binary mask (255 inside active view, 0 in dark corners).
        """
        if rgb_image.ndim == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_image.copy()

        _, border_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        return border_mask

    def _generate_initial_level_set(
        self, shape: Tuple[int, int], rgb_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generates an automated initial contour level set without ground-truth masks.

        Args:
            shape: Tuple of (height, width).
            rgb_image: Optional (H, W, 3) original RGB image for image-based seed initialization.

        Returns:
            (H, W) uint8 binary initial level set mask (1 inside, 0 outside).
        """
        height, width = shape

        if self.init_type == "otsu_seed" and rgb_image is not None and rgb_image.ndim == 3:
            # 1. Convert to LAB color space (A-channel highlights pink/red tissue contrast)
            lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            a_chan = lab[:, :, 1]
            blur_a = cv2.GaussianBlur(a_chan, (9, 9), 2.0)

            # 2. Otsu thresholding for automated seed selection
            _, thresh = cv2.threshold(blur_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Exclude dark corners
            border_mask = self.get_black_border_mask(rgb_image)
            thresh = cv2.bitwise_and(thresh, border_mask)

            # Morphological opening to clean small noise specks
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            init_ls = (thresh > 0).astype(np.uint8)

            if np.count_nonzero(init_ls) > 0:
                return init_ls

        # Fallback: Compact central disk level set
        center = (height // 2, width // 2)
        radius = int(min(height, width) * self.init_radius_ratio)
        init_ls = disk_level_set(shape, center=center, radius=max(radius, 15))
        return init_ls.astype(np.uint8)

    def segment(
        self, preprocessed_image: np.ndarray, rgb_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Evolves active contour on preprocessed image frame to extract polyp region.

        Args:
            preprocessed_image: (H, W) uint8 single-channel preprocessed image array.
            rgb_image: Optional (H, W, 3) original RGB image for image-based seed generation.

        Returns:
            Tuple of:
                - pred_mask: (H, W) uint8 binary mask with values 0 or 255.
                - success: bool indicating whether valid contour converged.
                - info: dict containing segmentation metadata.
        """
        if preprocessed_image.ndim != 2:
            raise ValueError(f"Expected 2D single-channel image, got shape {preprocessed_image.shape}.")

        h, w = preprocessed_image.shape
        max_dim = 512

        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            proc_img = cv2.resize(preprocessed_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_sub = (
                cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if rgb_image is not None
                else None
            )
            init_ls = self._generate_initial_level_set((new_h, new_w), rgb_image=rgb_sub)
        else:
            proc_img = preprocessed_image
            init_ls = self._generate_initial_level_set((h, w), rgb_image=rgb_image)

        try:
            # Evolve Morphological Chan-Vese Active Contour
            cv_res = morphological_chan_vese(
                image=proc_img,
                num_iter=self.max_iterations,
                init_level_set=init_ls,
                smoothing=self.smoothing,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
            )

            # Convert boolean level set to uint8 (0 or 255)
            pred_mask = (cv_res.astype(np.uint8)) * 255

            if max(h, w) > max_dim:
                pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Mask out dark endoscopy frame corners if RGB image provided
            if rgb_image is not None:
                border_mask = self.get_black_border_mask(rgb_image)
                pred_mask = cv2.bitwise_and(pred_mask, border_mask)

            # Post-processing: Fill internal holes in binary contour
            contours, _ = cv2.findContours(
                pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            filled_mask = np.zeros_like(pred_mask)
            if contours:
                cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
                pred_mask = filled_mask

            # Check validity (must not be completely empty or completely full)
            fg_pixels = np.count_nonzero(pred_mask == 255)
            total_pixels = h * w
            fg_ratio = fg_pixels / total_pixels

            # Valid segmentation if foreground ratio is strictly between 0.05% and 95%
            success = bool(0.0005 <= fg_ratio <= 0.95)

            info = {
                "max_iterations": self.max_iterations,
                "fg_pixels": fg_pixels,
                "fg_ratio_pct": float(fg_ratio * 100.0),
                "success": success,
            }

            return pred_mask, success, info

        except Exception as e:
            logger.error(f"Active contour segmentation failed: {str(e)}")
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            info = {"error": str(e), "success": False, "fg_pixels": 0, "fg_ratio_pct": 0.0}
            return empty_mask, False, info

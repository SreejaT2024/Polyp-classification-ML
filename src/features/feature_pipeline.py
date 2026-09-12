"""Stage A Feature Extraction Pipeline.

Orchestrates the complete end-to-end feature extraction workflow:
Image → Preprocessing → Active Contour Segmentation → Predicted Mask
→ ROI Extraction → Shape/Color/Texture Features → Feature Dictionary.

Ground-truth masks are used ONLY for evaluation metrics (Dice, IoU, Precision, Recall)
and are never used as input to segmentation, ROI extraction, or feature computation.
"""

from typing import Dict, Any, Optional
import numpy as np

from config.config import GlobalConfig, get_config
from src.preprocessing.image_preprocess import PolypImagePreprocessor
from src.segmentation.active_contour import ActiveContourSegmenter
from src.segmentation.metrics import evaluate_segmentation
from src.features.roi_extractor import ROIExtractor
from src.features.shape_features import extract_shape_features
from src.features.color_features import extract_color_features
from src.features.texture_features import extract_texture_features
from src.utils.logger import get_logger

logger = get_logger("feature_pipeline")


class StageAFeaturePipeline:
    """End-to-end pipeline: preprocessing → segmentation → ROI → features."""

    def __init__(self, config: Optional[GlobalConfig] = None) -> None:
        """Initializes pipeline components.

        Args:
            config: Optional GlobalConfig instance.
        """
        self.config = config or get_config()
        self.preprocessor = PolypImagePreprocessor()
        self.segmenter = ActiveContourSegmenter(config=self.config)
        self.roi_extractor = ROIExtractor()

    def process_sample(
        self,
        rgb_image: np.ndarray,
        filename: str,
        gt_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Processes a single endoscopic image through the complete Stage A pipeline.

        Args:
            rgb_image: (H, W, 3) uint8 RGB endoscopy frame.
            filename: Sample identifier string.
            gt_mask: Optional (H, W) uint8 binary ground-truth mask for evaluation only.

        Returns:
            Dict containing:
                - All extracted shape, color, texture features (classifier inputs).
                - Evaluation metadata (Dice, IoU, Precision, Recall) if gt_mask provided.
                - Segmentation status metadata.
        """
        result: Dict[str, Any] = {"filename": filename}

        # ── Step 1: Preprocessing ──
        try:
            preprocessed = self.preprocessor.preprocess_for_segmentation(rgb_image)
        except Exception as e:
            logger.error(f"Preprocessing failed for {filename}: {e}")
            result["seg_success"] = False
            result["error"] = f"preprocessing: {e}"
            return self._fill_defaults(result)

        # ── Step 2: Active Contour Segmentation (NO ground-truth used) ──
        try:
            pred_mask, seg_success, seg_info = self.segmenter.segment(
                preprocessed, rgb_image=rgb_image
            )
        except Exception as e:
            logger.error(f"Segmentation failed for {filename}: {e}")
            result["seg_success"] = False
            result["error"] = f"segmentation: {e}"
            return self._fill_defaults(result)

        result["seg_success"] = seg_success
        result["seg_fg_ratio_pct"] = seg_info.get("fg_ratio_pct", 0.0)

        # ── Step 3: Evaluation against ground-truth (evaluation only) ──
        if gt_mask is not None:
            metrics = evaluate_segmentation(pred_mask, gt_mask)
            result["dice"] = metrics["dice"]
            result["iou"] = metrics["iou"]
            result["precision"] = metrics["precision"]
            result["recall"] = metrics["recall"]
        else:
            result["dice"] = None
            result["iou"] = None
            result["precision"] = None
            result["recall"] = None

        # ── Step 4: ROI Extraction from predicted mask (NO ground-truth) ──
        try:
            roi_rgb, roi_mask, roi_masked_rgb, bbox, valid_roi = (
                self.roi_extractor.extract_roi(rgb_image, pred_mask)
            )
        except Exception as e:
            logger.error(f"ROI extraction failed for {filename}: {e}")
            result["error"] = f"roi_extraction: {e}"
            return self._fill_defaults(result)

        result["roi_valid"] = valid_roi
        result["roi_bbox_x"] = bbox[0]
        result["roi_bbox_y"] = bbox[1]
        result["roi_bbox_w"] = bbox[2]
        result["roi_bbox_h"] = bbox[3]

        # ── Step 5: Shape Features from predicted mask ──
        shape_feats = extract_shape_features(pred_mask)
        result.update(shape_feats)

        # ── Step 6: Color Features from original RGB + predicted mask ──
        color_feats = extract_color_features(rgb_image, pred_mask)
        result.update(color_feats)

        # ── Step 7: Texture Features from ROI ──
        texture_feats = extract_texture_features(roi_rgb, roi_mask)
        result.update(texture_feats)

        # Store intermediate arrays for visualization (not serialized to CSV)
        result["_pred_mask"] = pred_mask
        result["_roi_rgb"] = roi_rgb
        result["_roi_mask"] = roi_mask

        return result

    @staticmethod
    def _fill_defaults(result: Dict[str, Any]) -> Dict[str, Any]:
        """Fills a result dict with default zero values for all feature columns."""
        from src.features.shape_features import extract_shape_features
        from src.features.color_features import extract_color_features
        from src.features.texture_features import extract_texture_features

        dummy_mask = np.zeros((10, 10), dtype=np.uint8)
        dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)

        for k, v in extract_shape_features(dummy_mask).items():
            result.setdefault(k, v)
        for k, v in extract_color_features(dummy_rgb, dummy_mask).items():
            result.setdefault(k, v)
        for k, v in extract_texture_features(dummy_rgb, dummy_mask).items():
            result.setdefault(k, v)

        result.setdefault("seg_success", False)
        result.setdefault("seg_fg_ratio_pct", 0.0)
        result.setdefault("roi_valid", False)
        result.setdefault("roi_bbox_x", 0)
        result.setdefault("roi_bbox_y", 0)
        result.setdefault("roi_bbox_w", 0)
        result.setdefault("roi_bbox_h", 0)
        result.setdefault("dice", None)
        result.setdefault("iou", None)
        result.setdefault("precision", None)
        result.setdefault("recall", None)

        return result

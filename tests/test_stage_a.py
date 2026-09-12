"""PyTest Suite for Stage A Preprocessing, Active Contour, ROI, and Feature Extraction Modules."""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config
from src.data.kvasir_dataset import KvasirSEGDataset
from src.preprocessing.image_preprocess import PolypImagePreprocessor
from src.segmentation.active_contour import ActiveContourSegmenter
from src.segmentation.metrics import evaluate_segmentation, calculate_dice, calculate_iou
from src.features.roi_extractor import ROIExtractor
from src.features.shape_features import extract_shape_features
from src.features.color_features import extract_color_features
from src.features.texture_features import extract_texture_features
from src.features.feature_pipeline import StageAFeaturePipeline


@pytest.fixture
def sample_data():
    """Fixture returning sample dataset image and mask."""
    config = get_config()
    dataset = KvasirSEGDataset(config=config)
    return dataset[0]


def test_preprocessing(sample_data):
    """Verifies that preprocessor returns 2D uint8 image with correct dimensions."""
    rgb_img = sample_data["image"]
    preprocessor = PolypImagePreprocessor()
    prep_img = preprocessor.preprocess_for_segmentation(rgb_img)

    assert isinstance(prep_img, np.ndarray)
    assert prep_img.ndim == 2
    assert prep_img.shape == rgb_img.shape[:2]
    assert prep_img.dtype == np.uint8


def test_active_contour_segmentation(sample_data):
    """Verifies active contour produces binary mask with shape matching input."""
    rgb_img = sample_data["image"]
    preprocessor = PolypImagePreprocessor()
    prep_img = preprocessor.preprocess_for_segmentation(rgb_img)

    segmenter = ActiveContourSegmenter(max_iterations=50)
    pred_mask, success, info = segmenter.segment(prep_img, rgb_image=rgb_img)

    assert isinstance(pred_mask, np.ndarray)
    assert pred_mask.shape == prep_img.shape
    assert set(np.unique(pred_mask)).issubset({0, 255})
    assert isinstance(success, (bool, np.bool_))
    assert "fg_ratio_pct" in info


def test_segmentation_metrics():
    """Verifies Dice and IoU calculation edge cases."""
    m1 = np.ones((50, 50), dtype=np.uint8) * 255
    m2 = np.ones((50, 50), dtype=np.uint8) * 255

    assert calculate_dice(m1, m2) == 1.0
    assert calculate_iou(m1, m2) == 1.0

    m_empty = np.zeros((50, 50), dtype=np.uint8)
    assert calculate_dice(m1, m_empty) == 0.0
    assert calculate_iou(m1, m_empty) == 0.0

    metrics = evaluate_segmentation(m1, m2)
    assert set(metrics.keys()) == {"dice", "iou", "precision", "recall"}


def test_roi_extractor(sample_data):
    """Verifies ROI extraction bounds and masked output shape."""
    rgb_img = sample_data["image"]
    mask = sample_data["mask"]

    extractor = ROIExtractor()
    roi_rgb, roi_mask, roi_masked, bbox, valid = extractor.extract_roi(rgb_img, mask)

    assert valid is True
    assert roi_rgb.ndim == 3
    assert roi_mask.ndim == 2
    assert roi_rgb.shape[:2] == roi_mask.shape
    assert len(bbox) == 4
    assert bbox[2] > 0 and bbox[3] > 0  # Width and height > 0


def test_feature_extractors(sample_data):
    """Verifies shape, color, and texture features return non-null finite floats."""
    rgb_img = sample_data["image"]
    mask = sample_data["mask"]

    shape_f = extract_shape_features(mask)
    assert "shape_area" in shape_f
    assert shape_f["shape_area"] > 0

    color_f = extract_color_features(rgb_img, mask)
    assert "color_rgb_r_mean" in color_f
    assert "color_lab_l_mean" in color_f

    roi_extractor = ROIExtractor()
    roi_rgb, roi_mask, _, _, _ = roi_extractor.extract_roi(rgb_img, mask)
    texture_f = extract_texture_features(roi_rgb, roi_mask)
    assert "texture_glcm_contrast" in texture_f
    assert "texture_lbp_mean" in texture_f

    # Ensure no NaN or Inf in any extracted features
    all_features = {**shape_f, **color_f, **texture_f}
    for k, v in all_features.items():
        assert np.isfinite(v), f"Feature {k} is non-finite: {v}"


def test_feature_pipeline(sample_data):
    """Verifies end-to-end Stage A feature pipeline execution."""
    pipeline = StageAFeaturePipeline()
    res = pipeline.process_sample(
        rgb_image=sample_data["image"],
        filename=sample_data["filename"],
        gt_mask=sample_data["mask"],
    )

    assert res["filename"] == sample_data["filename"]
    assert "dice" in res
    assert res["dice"] is not None
    assert "shape_area" in res
    assert "color_rgb_r_mean" in res
    assert "texture_glcm_contrast" in res

"""Segmentation Evaluation Metrics Module.

Calculates Dice Similarity Coefficient (DSC), IoU (Jaccard Index), Precision,
and Recall comparing predicted active-contour masks against ground-truth masks.
These metrics are strictly reserved for post-hoc evaluation and analysis.
"""

from typing import Dict
import numpy as np


def calculate_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates Dice Similarity Coefficient (DSC) between binary masks.

    Args:
        pred_mask: 2D binary uint8 NumPy array (values 0 or 255).
        gt_mask: 2D binary uint8 NumPy array (values 0 or 255).

    Returns:
        Float score in range [0.0, 1.0].
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()

    if total == 0:
        return 1.0  # Both empty masks match perfectly
    return float((2.0 * intersection) / total)


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates Intersection over Union (IoU / Jaccard Index) between binary masks.

    Args:
        pred_mask: 2D binary uint8 NumPy array.
        gt_mask: 2D binary uint8 NumPy array.

    Returns:
        Float score in range [0.0, 1.0].
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 1.0  # Both empty masks match perfectly
    return float(intersection / union)


def calculate_precision(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates Precision (Positive Predictive Value) between binary masks.

    Args:
        pred_mask: 2D binary uint8 NumPy array.
        gt_mask: 2D binary uint8 NumPy array.

    Returns:
        Float score in range [0.0, 1.0].
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    tp = np.logical_and(pred_bin, gt_bin).sum()
    pred_positive = pred_bin.sum()

    if pred_positive == 0:
        return 0.0
    return float(tp / pred_positive)


def calculate_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculates Recall (Sensitivity / True Positive Rate) between binary masks.

    Args:
        pred_mask: 2D binary uint8 NumPy array.
        gt_mask: 2D binary uint8 NumPy array.

    Returns:
        Float score in range [0.0, 1.0].
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    tp = np.logical_and(pred_bin, gt_bin).sum()
    gt_positive = gt_bin.sum()

    if gt_positive == 0:
        return 0.0
    return float(tp / gt_positive)


def evaluate_segmentation(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Calculates comprehensive segmentation metrics dictionary.

    Args:
        pred_mask: 2D binary uint8 predicted mask.
        gt_mask: 2D binary uint8 ground-truth mask.

    Returns:
        Dict containing 'dice', 'iou', 'precision', and 'recall'.
    """
    return {
        "dice": calculate_dice(pred_mask, gt_mask),
        "iou": calculate_iou(pred_mask, gt_mask),
        "precision": calculate_precision(pred_mask, gt_mask),
        "recall": calculate_recall(pred_mask, gt_mask),
    }

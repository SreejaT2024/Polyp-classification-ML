"""PyTest Suite for Data Ingestion & Dataset Integrity."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config
from src.data.kvasir_dataset import KvasirSEGDataset


def test_config_loading():
    """Verifies config loading and directory path resolutions."""
    config = get_config()
    assert config.paths.kvasir_root.exists()
    assert config.paths.kvasir_images_dir.exists()
    assert config.paths.kvasir_masks_dir.exists()


def test_dataset_initialization():
    """Verifies that KvasirSEGDataset indexes images and masks properly."""
    dataset = KvasirSEGDataset()
    assert len(dataset) > 0
    assert len(dataset) == 1000  # Kvasir-SEG has 1000 pairs


def test_dataset_getitem():
    """Verifies indexing, numpy array shapes, and binary mask value range."""
    dataset = KvasirSEGDataset()
    sample = dataset[0]

    assert "image" in sample
    assert "mask" in sample
    assert "filename" in sample

    image = sample["image"]
    mask = sample["mask"]

    # Image checks (H, W, 3)
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 3

    # Mask checks (H, W) single channel
    assert isinstance(mask, np.ndarray)
    assert mask.ndim == 2
    assert image.shape[0] == mask.shape[0]
    assert image.shape[1] == mask.shape[1]

    # Binary mask check (values should be exclusively 0 or 255)
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})


def test_dataset_integrity():
    """Verifies that 100% of samples match in spatial dimensions."""
    dataset = KvasirSEGDataset()
    is_valid, valid_count, summary = dataset.verify_integrity()
    assert is_valid is True
    assert valid_count == 1000
    assert summary["corrupted_count"] == 0

"""Centralized Configuration Module.

Provides dataclasses for managing project paths, dataset parameters,
image specifications, and logging configurations across all processing phases.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass
class PathConfig:
    """Project directory and dataset file paths configuration."""

    # Project Root
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    # Kvasir-SEG Dataset Paths
    kvasir_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "kvasir-seg" / "Kvasir-SEG"
    )
    kvasir_images_dir: Path = field(init=False)
    kvasir_masks_dir: Path = field(init=False)
    kvasir_bboxes_path: Path = field(init=False)

    # Output & Artifact Directories
    output_dir: Path = field(init=False)
    features_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Derive sub-paths and ensure output directories exist."""
        self.kvasir_images_dir = self.kvasir_root / "images"
        self.kvasir_masks_dir = self.kvasir_root / "masks"
        self.kvasir_bboxes_path = self.kvasir_root / "kavsir_bboxes.json"

        self.output_dir = self.project_root / "outputs"
        self.features_dir = self.output_dir / "features"
        self.logs_dir = self.project_root / "logs"
        self.reports_dir = self.project_root / "reports"

        # Auto-create output directories if they do not exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)


@dataclass
class DatasetConfig:
    """Dataset loading and preprocessing specifications."""

    supported_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    mask_suffix: str = ""  # Masks match images by filename in Kvasir-SEG
    mask_threshold: int = 127
    target_image_size: Tuple[int, int] = (256, 256)
    color_mode: str = "RGB"
    seed: int = 42


@dataclass
class SegmentationConfig:
    """Active contour segmentation parameters."""

    init_type: str = "otsu_seed"
    init_radius_ratio: float = 0.15
    max_iterations: int = 80
    smoothing: int = 1
    lambda1: float = 1.0
    lambda2: float = 1.5


@dataclass
class FeatureConfig:
    """Feature extraction parameters."""

    output_filename: str = "kvasir_stageA_features.csv"


@dataclass
class LogConfig:
    """Logging settings."""

    level: str = "INFO"
    log_file_name: str = "pipeline.log"
    log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class GlobalConfig:
    """Master configuration encapsulating paths, dataset specs, and logging."""

    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    logging: LogConfig = field(default_factory=LogConfig)


def get_config(custom_kvasir_path: Optional[str] = None) -> GlobalConfig:
    """Factory function to load global configuration.

    Args:
        custom_kvasir_path: Optional custom path to Kvasir-SEG root directory.

    Returns:
        GlobalConfig instance initialized with validated paths.
    """
    config = GlobalConfig()
    if custom_kvasir_path:
        custom_root = Path(custom_kvasir_path).resolve()
        config.paths.kvasir_root = custom_root
        config.paths.kvasir_images_dir = custom_root / "images"
        config.paths.kvasir_masks_dir = custom_root / "masks"
        config.paths.kvasir_bboxes_path = custom_root / "kavsir_bboxes.json"

    return config

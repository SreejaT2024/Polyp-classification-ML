"""Kvasir-SEG Dataset Loader & Verification Pipeline.

Implements KvasirSEGDataset inheriting from BasePolypDataset. Performs automatic
pairing of RGB endoscopy frames with binary segmentation masks, validates integrity,
and computes dataset distribution statistics.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm

from config.config import GlobalConfig, get_config
from src.data.base_dataset import BasePolypDataset
from src.utils.io import (
    load_image,
    load_mask,
    get_image_dimensions,
    validate_image_mask_pair,
)
from src.utils.logger import get_logger

logger = get_logger("kvasir_dataset")


class KvasirSEGDataset(BasePolypDataset):
    """Dataset class for loading and managing Kvasir-SEG endoscopy samples."""

    def __init__(
        self,
        config: Optional[GlobalConfig] = None,
        kvasir_root: Optional[Union[str, Path]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        """Initializes KvasirSEGDataset.

        Args:
            config: Optional GlobalConfig dataclass instance.
            kvasir_root: Optional override path for Kvasir-SEG root directory.
            transform: Optional spatial or color transformations (for future phases).
        """
        self.config = config or get_config(
            custom_kvasir_path=str(kvasir_root) if kvasir_root else None
        )
        self.images_dir = Path(self.config.paths.kvasir_images_dir)
        self.masks_dir = Path(self.config.paths.kvasir_masks_dir)
        self.transform = transform

        self.samples: List[Dict[str, Path]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Discovers images and pairs each image with its corresponding mask."""
        if not self.images_dir.is_dir():
            logger.error(f"Images directory not found at: {self.images_dir}")
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")
        if not self.masks_dir.is_dir():
            logger.error(f"Masks directory not found at: {self.masks_dir}")
            raise FileNotFoundError(f"Masks directory does not exist: {self.masks_dir}")

        image_files = sorted(
            [
                f
                for f in self.images_dir.iterdir()
                if f.suffix.lower() in self.config.dataset.supported_extensions
            ]
        )

        paired_samples = []
        unmatched_count = 0

        for img_path in image_files:
            mask_path = self.masks_dir / img_path.name
            if mask_path.is_file():
                paired_samples.append(
                    {
                        "filename": img_path.name,
                        "stem": img_path.stem,
                        "image_path": img_path,
                        "mask_path": mask_path,
                    }
                )
            else:
                unmatched_count += 1
                logger.warning(f"No matching mask found for image: {img_path.name}")

        self.samples = paired_samples
        logger.info(
            f"Successfully indexed Kvasir-SEG dataset: {len(self.samples)} valid image-mask pairs found."
        )
        if unmatched_count > 0:
            logger.warning(f"Encountered {unmatched_count} images without matching masks.")

    def __len__(self) -> int:
        """Returns total number of paired samples in dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Loads and returns the sample dictionary at given index.

        Args:
            index: Integer sample index.

        Returns:
            Dict containing raw numpy image array, mask array, and metadata.
        """
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of bounds for dataset length {len(self.samples)}.")

        sample_info = self.samples[index]
        image = load_image(sample_info["image_path"], color_mode=self.config.dataset.color_mode)
        mask = load_mask(
            sample_info["mask_path"], binary_threshold=self.config.dataset.mask_threshold
        )

        sample_dict = {
            "image": image,
            "mask": mask,
            "filename": sample_info["filename"],
            "image_path": str(sample_info["image_path"]),
            "mask_path": str(sample_info["mask_path"]),
            "shape": image.shape,
        }

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)

        return sample_dict

    def verify_integrity(self) -> Tuple[bool, int, Dict[str, Any]]:
        """Scans all samples to verify dimension alignment, readability, and binary values.

        Returns:
            Tuple of (is_all_valid: bool, valid_count: int, audit_summary: Dict).
        """
        logger.info("Executing deep integrity verification across all dataset pairs...")
        valid_count = 0
        corrupted_pairs = []

        for sample in tqdm(self.samples, desc="Verifying Image-Mask Pairs"):
            is_valid, msg = validate_image_mask_pair(
                sample["image_path"], sample["mask_path"]
            )
            if is_valid:
                valid_count += 1
            else:
                corrupted_pairs.append({"filename": sample["filename"], "reason": msg})

        is_all_valid = len(corrupted_pairs) == 0
        audit_summary = {
            "total_samples": len(self.samples),
            "valid_samples": valid_count,
            "corrupted_count": len(corrupted_pairs),
            "corrupted_details": corrupted_pairs,
        }

        if is_all_valid:
            logger.info("Integrity Verification Passed: 100% of image-mask pairs match perfectly.")
        else:
            logger.error(
                f"Integrity Verification Failed: Found {len(corrupted_pairs)} invalid pairs."
            )

        return is_all_valid, valid_count, audit_summary

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Computes statistical metrics across the entire dataset.

        Returns:
            Dict containing total images, unique resolution dimensions,
            aspect ratios, and mask polyp coverage percentages.
        """
        logger.info("Calculating dataset summary statistics...")
        resolutions = []
        polyp_coverages = []

        for sample in self.samples:
            w, h = get_image_dimensions(sample["image_path"])
            resolutions.append((w, h))

            # Load mask to calculate polyp area ratio
            mask = load_mask(
                sample["mask_path"], binary_threshold=self.config.dataset.mask_threshold
            )
            polyp_pixels = np.count_nonzero(mask == 255)
            total_pixels = mask.size
            coverage_pct = (polyp_pixels / total_pixels) * 100.0
            polyp_coverages.append(coverage_pct)

        widths, heights = zip(*resolutions)
        unique_resolutions = list(set(resolutions))

        stats = {
            "total_images": len(self.samples),
            "min_dimension": (min(widths), min(heights)),
            "max_dimension": (max(widths), max(heights)),
            "avg_width": float(np.mean(widths)),
            "avg_height": float(np.mean(heights)),
            "unique_resolutions_count": len(unique_resolutions),
            "unique_resolutions_sample": unique_resolutions[:10],
            "min_polyp_coverage_pct": float(np.min(polyp_coverages)),
            "max_polyp_coverage_pct": float(np.max(polyp_coverages)),
            "mean_polyp_coverage_pct": float(np.mean(polyp_coverages)),
            "median_polyp_coverage_pct": float(np.median(polyp_coverages)),
        }
        return stats

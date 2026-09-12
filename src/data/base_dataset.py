"""Abstract Base Dataset Interface for Endoscopic Polyp Datasets.

Enforces a consistent contract across different endoscopy datasets (e.g., Kvasir-SEG,
HyperKvasir, CVC-ClinicDB) following SOLID design principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BasePolypDataset(ABC):
    """Abstract Base Class for Endoscopy Polyp Datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetches the sample at the given index.

        Returns:
            Dict containing:
                - 'image': np.ndarray RGB image frame
                - 'mask': np.ndarray uint8 binary segmentation mask
                - 'filename': str sample identifier/filename
                - 'image_path': str absolute path to image
                - 'mask_path': str absolute path to mask
        """
        pass

    @abstractmethod
    def verify_integrity(self) -> Tuple[bool, int, Dict[str, Any]]:
        """Verifies dataset completeness, file existence, and image-mask pair alignment.

        Returns:
            Tuple of (is_valid: bool, total_valid_samples: int, summary_dict: Dict).
        """
        pass

    @abstractmethod
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Calculates and reports comprehensive statistical metrics of the dataset."""
        pass

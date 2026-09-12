"""Image and Mask File I/O Utilities.

Provides robust functions for loading RGB endoscopy frames, loading binary masks,
inspecting image dimensions, and validating image-mask pair integrity.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import cv2

from src.utils.logger import get_logger

logger = get_logger("io_utils")


def load_image(path: Union[str, Path], color_mode: str = "RGB") -> np.ndarray:
    """Loads an image from disk into a NumPy array.

    Args:
        path: Absolute or relative path to the image file.
        color_mode: Target color mode ('RGB', 'BGR', or 'GRAY').

    Returns:
        NumPy ndarray representing the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        logger.error(f"Image file not found: {path_obj}")
        raise FileNotFoundError(f"Image file does not exist: {path_obj}")

    try:
        pil_img = Image.open(path_obj)
        if color_mode == "RGB":
            pil_img = pil_img.convert("RGB")
            return np.array(pil_img)
        elif color_mode == "GRAY":
            pil_img = pil_img.convert("L")
            return np.array(pil_img)
        elif color_mode == "BGR":
            pil_img = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported color_mode: {color_mode}")
    except Exception as e:
        logger.error(f"Failed to load image from {path_obj}: {str(e)}")
        raise ValueError(f"Could not decode image at {path_obj}: {str(e)}") from e


def load_mask(path: Union[str, Path], binary_threshold: int = 127) -> np.ndarray:
    """Loads a binary segmentation mask from disk as a 2D single-channel NumPy array.

    Args:
        path: Path to the mask file.
        binary_threshold: Threshold to convert mask to binary 0/255 array.

    Returns:
        Single channel 2D uint8 NumPy ndarray (values 0 or 255).

    Raises:
        FileNotFoundError: If the mask file does not exist.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        logger.error(f"Mask file not found: {path_obj}")
        raise FileNotFoundError(f"Mask file does not exist: {path_obj}")

    try:
        pil_mask = Image.open(path_obj).convert("L")
        mask_arr = np.array(pil_mask, dtype=np.uint8)
        # Binarize mask to ensure clean {0, 255} binary ground truth
        _, binary_mask = cv2.threshold(mask_arr, binary_threshold, 255, cv2.THRESH_BINARY)
        return binary_mask
    except Exception as e:
        logger.error(f"Failed to load mask from {path_obj}: {str(e)}")
        raise ValueError(f"Could not decode mask at {path_obj}: {str(e)}") from e


def get_image_dimensions(path: Union[str, Path]) -> Tuple[int, int]:
    """Retrieves image dimensions (width, height) without loading pixel array into memory.

    Args:
        path: Path to image file.

    Returns:
        Tuple of (width, height) in pixels.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"File not found: {path_obj}")

    with Image.open(path_obj) as img:
        return img.size  # Returns (width, height)


def validate_image_mask_pair(
    image_path: Union[str, Path], mask_path: Union[str, Path]
) -> Tuple[bool, str]:
    """Validates that an image file and mask file exist and match spatial dimensions.

    Args:
        image_path: Path to raw endoscopy frame.
        mask_path: Path to corresponding segmentation mask.

    Returns:
        Tuple of (is_valid: bool, status_message: str).
    """
    img_p = Path(image_path)
    msk_p = Path(mask_path)

    if not img_p.is_file():
        return False, f"Missing image file: {img_p.name}"
    if not msk_p.is_file():
        return False, f"Missing mask file: {msk_p.name}"

    try:
        img_w, img_h = get_image_dimensions(img_p)
        msk_w, msk_h = get_image_dimensions(msk_p)

        if (img_w, img_h) != (msk_w, msk_h):
            return (
                False,
                f"Dimension mismatch! Image: ({img_w}x{img_h}), Mask: ({msk_w}x{msk_h})",
            )
        return True, "Valid matching image-mask pair"
    except Exception as e:
        return False, f"Corrupted file pair error: {str(e)}"

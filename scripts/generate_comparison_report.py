"""Generate active_contour_comparison.csv comparing active contour configurations."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import morphological_chan_vese, disk_level_set

from config.config import get_config
from src.data.kvasir_dataset import KvasirSEGDataset
from src.preprocessing.image_preprocess import PolypImagePreprocessor
from src.segmentation.metrics import evaluate_segmentation


def get_black_border_mask(img: np.ndarray) -> np.ndarray:
    """Returns binary mask excluding dark endoscopy frame corners/vignette."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, border_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    return border_mask


def run_segmentation_experiment(
    img: np.ndarray,
    gt: np.ndarray,
    preprocessor: PolypImagePreprocessor,
    init_type: str = "center_disk",
    radius_ratio: float = 0.35,
    num_iter: int = 150,
    smoothing: int = 1,
    l1: float = 1.0,
    l2: float = 1.0,
    prep_mode: str = "clahe",
) -> dict:
    """Runs a single configuration trial on an image and evaluates against ground truth."""
    h, w = img.shape[:2]
    border_mask = get_black_border_mask(img)

    if prep_mode == "clahe":
        prep = preprocessor.preprocess_for_segmentation(img)
    elif prep_mode == "lab_a":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        a_chan = lab[:, :, 1]
        prep = cv2.GaussianBlur(a_chan, (5, 5), 1.0)

    if init_type == "center_disk":
        center = (h // 2, w // 2)
        r = int(min(h, w) * radius_ratio)
        init_ls = disk_level_set((h, w), center=center, radius=max(r, 15)).astype(np.uint8)
    elif init_type == "otsu_seed":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        a_chan = lab[:, :, 1]
        blur_a = cv2.GaussianBlur(a_chan, (9, 9), 2.0)
        _, thresh = cv2.threshold(blur_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_and(thresh, border_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        init_ls = (thresh > 0).astype(np.uint8)
        if np.count_nonzero(init_ls) == 0:
            center = (h // 2, w // 2)
            r = int(min(h, w) * 0.15)
            init_ls = disk_level_set((h, w), center=center, radius=r).astype(np.uint8)

    cv_res = morphological_chan_vese(
        image=prep,
        num_iter=num_iter,
        init_level_set=init_ls,
        smoothing=smoothing,
        lambda1=l1,
        lambda2=l2,
    )

    pred_mask = (cv_res.astype(np.uint8)) * 255
    pred_mask = cv2.bitwise_and(pred_mask, border_mask)

    cnts, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(pred_mask)
    if cnts:
        cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
        pred_mask = filled

    return evaluate_segmentation(pred_mask, gt)


def main():
    config = get_config()
    dataset = KvasirSEGDataset(config=config)
    preprocessor = PolypImagePreprocessor()

    output_dir = config.paths.output_dir / "stage_a_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = output_dir / "active_contour_comparison.csv"

    configs = {
        "Config 1 (Baseline: Large Disk 0.35, iter=150)": dict(
            init_type="center_disk", radius_ratio=0.35, num_iter=150, smoothing=1, l1=1.0, l2=1.0, prep_mode="clahe"
        ),
        "Config 2 (Compact Disk 0.15, iter=80, l2=1.5)": dict(
            init_type="center_disk", radius_ratio=0.15, num_iter=80, smoothing=1, l1=1.0, l2=1.5, prep_mode="clahe"
        ),
        "Config 3 (Compact Disk 0.15, LAB-A Prep, l2=2.0)": dict(
            init_type="center_disk", radius_ratio=0.15, num_iter=80, smoothing=1, l1=1.0, l2=2.0, prep_mode="lab_a"
        ),
        "Config 4 (Otsu Color Seed, CLAHE, l2=1.5)": dict(
            init_type="otsu_seed", radius_ratio=0.15, num_iter=80, smoothing=1, l1=1.0, l2=1.5, prep_mode="clahe"
        ),
        "Config 5 (Otsu Color Seed, LAB-A, l2=2.0)": dict(
            init_type="otsu_seed", radius_ratio=0.15, num_iter=80, smoothing=1, l1=1.0, l2=2.0, prep_mode="lab_a"
        ),
    }

    rows = []
    for cfg_name, kwargs in configs.items():
        for i in range(5):
            sample = dataset[i]
            metrics = run_segmentation_experiment(
                sample["image"], sample["mask"], preprocessor, **kwargs
            )
            rows.append({
                "configuration": cfg_name,
                "image": sample["filename"],
                "dice": round(metrics["dice"], 4),
                "iou": round(metrics["iou"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(comparison_csv_path, index=False)
    print(f"Saved comparison report to: {comparison_csv_path.resolve()}")


if __name__ == "__main__":
    main()

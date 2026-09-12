"""Phase 2I: Stage A Pipeline Validation & Physical Disk Persistence (5 Kvasir-SEG Images).

Runs the complete Stage A feature extraction pipeline on 5 sample images, evaluates metrics,
extracts shape/color/texture features, and physically saves outputs:
- outputs/stage_a_validation/results.csv
- outputs/stage_a_validation/feature_preview.csv
- outputs/stage_a_validation/visualizations/*.png (5 side-by-side visualization panels)
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.config import get_config
from src.data.kvasir_dataset import KvasirSEGDataset
from src.features.feature_pipeline import StageAFeaturePipeline


def create_visualization(
    rgb_img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    roi_rgb: np.ndarray,
    filename: str,
    output_path: Path,
) -> None:
    """Generates and saves a 5-panel visualization figure for a single sample.

    Panel 1: Original RGB Image
    Panel 2: Ground-Truth Mask
    Panel 3: Active-Contour Predicted Mask
    Panel 4: Predicted Contour Overlay
    Panel 5: Extracted Polyp ROI
    """
    # Create overlay image
    overlay = rgb_img.copy()
    contours, _ = cv2.findContours(
        pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=3)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(f"Stage A Validation: {filename}", fontsize=14, fontweight="bold")

    # 1. Original Image
    axes[0].imshow(rgb_img)
    axes[0].set_title("1. Original RGB")
    axes[0].axis("off")

    # 2. Ground-Truth Mask (Evaluation Only)
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("2. Ground-Truth Mask")
    axes[1].axis("off")

    # 3. Active Contour Predicted Mask
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("3. Predicted Mask")
    axes[2].axis("off")

    # 4. Predicted Contour Overlay
    axes[3].imshow(overlay)
    axes[3].set_title("4. Predicted Contour")
    axes[3].axis("off")

    # 5. Extracted Polyp ROI
    axes[4].imshow(roi_rgb)
    axes[4].set_title(f"5. Extracted ROI\n({roi_rgb.shape[1]}x{roi_rgb.shape[0]})")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    NUM_SAMPLES = 5

    print("=" * 80)
    print("  STAGE A PIPELINE VALIDATION & DISK PERSISTENCE (5-IMAGE TEST)")
    print("=" * 80)

    config = get_config()
    dataset = KvasirSEGDataset(config=config)
    pipeline = StageAFeaturePipeline(config=config)

    # Prepare output directories
    val_dir = config.paths.output_dir / "stage_a_validation"
    vis_dir = val_dir / "visualizations"
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"\nDataset loaded: {len(dataset)} samples")
    print(f"Output directory : {val_dir}")
    print(f"Processing first {NUM_SAMPLES} images...\n")

    results = []
    results_csv_rows = []
    feature_rows = []

    for idx in range(NUM_SAMPLES):
        sample = dataset[idx]
        rgb_image = sample["image"]
        gt_mask = sample["mask"]
        filename = sample["filename"]

        print(f"  [{idx + 1}/{NUM_SAMPLES}] Processing: {filename} "
              f"({rgb_image.shape[1]}x{rgb_image.shape[0]}) ...")

        # Run pipeline (Ground truth mask passed ONLY for evaluation metrics)
        result = pipeline.process_sample(
            rgb_image=rgb_image,
            filename=filename,
            gt_mask=gt_mask,
        )
        results.append(result)

        # 1. Row for results.csv
        results_csv_rows.append({
            "filename": filename,
            "dice": result.get("dice"),
            "iou": result.get("iou"),
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "segmentation_success": result.get("seg_success", False),
        })

        # 2. Row for feature_preview.csv (exclude internal arrays & non-feature keys)
        feat_dict = {
            "filename": filename,
            "seg_success": result.get("seg_success", False),
        }
        feature_keys = [
            k for k in result.keys()
            if k.startswith(("shape_", "color_", "texture_"))
        ]
        for fk in feature_keys:
            feat_dict[fk] = result[fk]
        feature_rows.append(feat_dict)

        # 3. Create & save visualization image
        pred_mask = result["_pred_mask"]
        roi_rgb = result["_roi_rgb"]
        stem = Path(filename).stem
        vis_path = vis_dir / f"{stem}_val.png"
        create_visualization(rgb_image, gt_mask, pred_mask, roi_rgb, filename, vis_path)

        status = "OK" if result.get("seg_success") else "WARN"
        dice_str = f"{result['dice']:.4f}" if result.get("dice") is not None else "N/A"
        iou_str = f"{result['iou']:.4f}" if result.get("iou") is not None else "N/A"
        print(f"         -> [{status}] Dice={dice_str}  IoU={iou_str}  "
              f"Saved Vis: {vis_path.name}")

    # ── Write CSV Files ──
    results_df = pd.DataFrame(results_csv_rows)
    results_csv_path = val_dir / "results.csv"
    results_df.to_csv(results_csv_path, index=False)

    feature_df = pd.DataFrame(feature_rows)
    feature_csv_path = val_dir / "feature_preview.csv"
    feature_df.to_csv(feature_csv_path, index=False)

    print("\n" + "=" * 80)
    print("  PHYSICAL DISK FILE VERIFICATION")
    print("=" * 80)

    generated_files = [results_csv_path, feature_csv_path]
    for idx in range(NUM_SAMPLES):
        stem = Path(dataset[idx]["filename"]).stem
        generated_files.append(vis_dir / f"{stem}_val.png")

    all_exist = True
    for fp in generated_files:
        exists = fp.exists()
        size_bytes = fp.stat().st_size if exists else 0
        print(f"  [{'EXISTS' if exists else 'MISSING'}] {fp.resolve()} ({size_bytes} bytes)")
        if not exists:
            all_exist = False

    # ── Validation & Integrity Checks ──
    print("\n" + "=" * 80)
    print("  FEATURE PREVIEW INTEGRITY CHECKS")
    print("=" * 80)

    numeric_cols = [c for c in feature_df.columns if c not in ("filename", "seg_success")]
    has_nan = feature_df[numeric_cols].isna().any().any()
    has_inf = np.isinf(feature_df[numeric_cols].to_numpy()).any()

    print(f"  One row per image       : {'PASS' if len(feature_df) == NUM_SAMPLES else 'FAIL'} ({len(feature_df)} rows)")
    print(f"  No NaN values           : {'PASS' if not has_nan else 'FAIL'}")
    print(f"  No Inf values           : {'PASS' if not has_inf else 'FAIL'}")
    print(f"  Total feature columns   : {len(numeric_cols)}")
    print(f"  Feature column names    : {list(numeric_cols)}")

    # ── Summary Table ──
    print("\n" + "=" * 80)
    print("  5-IMAGE RESULTS SUMMARY TABLE")
    print("=" * 80)

    summary_display = pd.DataFrame({
        "Filename": results_df["filename"],
        "Dice": results_df["dice"].map(lambda x: f"{x:.4f}"),
        "IoU": results_df["iou"].map(lambda x: f"{x:.4f}"),
        "Precision": results_df["precision"].map(lambda x: f"{x:.4f}"),
        "Recall": results_df["recall"].map(lambda x: f"{x:.4f}"),
        "Seg Success": results_df["segmentation_success"],
    })
    print(summary_display.to_string(index=False))

    avg_dice = results_df["dice"].mean()
    avg_iou = results_df["iou"].mean()
    avg_prec = results_df["precision"].mean()
    avg_rec = results_df["recall"].mean()

    print(f"\n  Average Dice      : {avg_dice:.4f}")
    print(f"  Average IoU       : {avg_iou:.4f}")
    print(f"  Average Precision : {avg_prec:.4f}")
    print(f"  Average Recall    : {avg_rec:.4f}")

    print("\n" + "=" * 80)
    print("  PHASE 2 VALIDATION COMPLETE - DO NOT PROCEED TO 1000-IMAGE BATCH")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Batch Stage A Feature Extraction Pipeline for all 1,000 Kvasir-SEG images.

Iterates over all 1,000 Kvasir-SEG endoscopy samples, runs the Stage A feature extraction
pipeline (preprocessing → active contour → ROI → shape/color/texture features), evaluates
segmentation metrics against ground-truth, and physically writes:
- outputs/features/kvasir_stageA_features.csv
- outputs/features/kvasir_stageA_segmentation_metrics.csv
- outputs/features/stageA_summary.txt
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import get_config
from src.data.kvasir_dataset import KvasirSEGDataset
from src.features.feature_pipeline import StageAFeaturePipeline
from src.utils.logger import get_logger

logger = get_logger("batch_stage_a")


def main() -> None:
    print("=" * 80)
    print("  STAGE A BATCH FEATURE EXTRACTION PIPELINE (1,000 KVASIR-SEG IMAGES)")
    print("=" * 80)

    config = get_config()
    dataset = KvasirSEGDataset(config=config)
    pipeline = StageAFeaturePipeline(config=config)

    output_features_dir = config.paths.output_dir / "features"
    os.makedirs(output_features_dir, exist_ok=True)

    feature_csv_path = output_features_dir / "kvasir_stageA_features.csv"
    metrics_csv_path = output_features_dir / "kvasir_stageA_segmentation_metrics.csv"
    summary_txt_path = output_features_dir / "stageA_summary.txt"

    total_samples = len(dataset)
    print(f"\nDataset loaded  : {total_samples} samples")
    print(f"Output directory: {output_features_dir}\n")

    feature_rows = []
    metrics_rows = []

    success_count = 0
    fail_count = 0

    for idx in tqdm(range(total_samples), desc="Extracting Stage A Features"):
        sample = dataset[idx]
        rgb_image = sample["image"]
        gt_mask = sample["mask"]
        filename = sample["filename"]

        try:
            res = pipeline.process_sample(
                rgb_image=rgb_image,
                filename=filename,
                gt_mask=gt_mask,
            )

            is_success = res.get("seg_success", False)
            if is_success:
                success_count += 1
            else:
                fail_count += 1

            # 1. Feature CSV row
            feature_keys = [
                k for k in res.keys()
                if k.startswith(("shape_", "color_", "texture_"))
            ]
            feat_row = {
                "filename": filename,
                "segmentation_success": is_success,
            }
            for fk in feature_keys:
                feat_row[fk] = res[fk]
            feature_rows.append(feat_row)

            # 2. Evaluation metrics CSV row
            metrics_rows.append({
                "filename": filename,
                "dice": res.get("dice"),
                "iou": res.get("iou"),
                "precision": res.get("precision"),
                "recall": res.get("recall"),
                "segmentation_success": is_success,
            })

        except Exception as e:
            logger.error(f"Error processing sample #{idx} ({filename}): {e}")
            fail_count += 1

            # Safe fallback feature row
            dummy_feat_row = {
                "filename": filename,
                "segmentation_success": False,
            }
            # Fill 37 feature keys with 0.0
            from src.features.shape_features import extract_shape_features
            from src.features.color_features import extract_color_features
            from src.features.texture_features import extract_texture_features

            d_mask = np.zeros((10, 10), dtype=np.uint8)
            d_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            for k in extract_shape_features(d_mask).keys():
                dummy_feat_row[k] = 0.0
            for k in extract_color_features(d_rgb, d_mask).keys():
                dummy_feat_row[k] = 0.0
            for k in extract_texture_features(d_rgb, d_mask).keys():
                dummy_feat_row[k] = 0.0
            feature_rows.append(dummy_feat_row)

            metrics_rows.append({
                "filename": filename,
                "dice": 0.0,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "segmentation_success": False,
            })

    # Convert to DataFrames
    df_features = pd.DataFrame(feature_rows)
    df_metrics = pd.DataFrame(metrics_rows)

    # Save CSVs
    df_features.to_csv(feature_csv_path, index=False)
    df_metrics.to_csv(metrics_csv_path, index=False)

    # Calculate statistics
    feature_cols = [c for c in df_features.columns if c not in ("filename", "segmentation_success")]
    nan_count = df_features[feature_cols].isna().sum().sum()
    inf_count = np.isinf(df_features[feature_cols].to_numpy()).sum()

    valid_dice = df_metrics["dice"].dropna()
    valid_iou = df_metrics["iou"].dropna()
    valid_prec = df_metrics["precision"].dropna()
    valid_rec = df_metrics["recall"].dropna()

    mean_dice = float(valid_dice.mean()) if len(valid_dice) > 0 else 0.0
    median_dice = float(valid_dice.median()) if len(valid_dice) > 0 else 0.0
    mean_iou = float(valid_iou.mean()) if len(valid_iou) > 0 else 0.0
    median_iou = float(valid_iou.median()) if len(valid_iou) > 0 else 0.0
    mean_prec = float(valid_prec.mean()) if len(valid_prec) > 0 else 0.0
    mean_rec = float(valid_rec.mean()) if len(valid_rec) > 0 else 0.0

    # Save summary report text file
    summary_text = f"""================================================================================
STAGE A FEATURE EXTRACTION PIPELINE - FINAL BATCH SUMMARY REPORT
================================================================================
Dataset Name                     : Kvasir-SEG
Total Images Processed           : {total_samples}
Successfully Segmented Images   : {success_count} ({success_count / total_samples * 100.0:.2f}%)
Failed Segmentation Images       : {fail_count} ({fail_count / total_samples * 100.0:.2f}%)

Feature Matrix Specifications:
  Feature CSV Path               : {feature_csv_path.resolve()}
  Total Rows                     : {len(df_features)}
  Total Columns                  : {len(df_features.columns)}
  Extracted Feature Count        : {len(feature_cols)} (9 Shape, 18 Color, 10 Texture)
  NaN Count in Feature Columns   : {nan_count}
  Inf Count in Feature Columns   : {inf_count}
  Duplicate Filenames            : {df_features['filename'].duplicated().sum()}

Segmentation Evaluation Metrics (Active Contour vs Ground Truth):
  Evaluation CSV Path            : {metrics_csv_path.resolve()}
  Mean Dice Similarity (DSC)     : {mean_dice:.4f}
  Median Dice Similarity (DSC)   : {median_dice:.4f}
  Mean Intersection over Union   : {mean_iou:.4f}
  Median Intersection over Union : {median_iou:.4f}
  Mean Precision                 : {mean_prec:.4f}
  Mean Recall                    : {mean_rec:.4f}
================================================================================
"""
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # ── Console Report ──
    print("\n" + "=" * 80)
    print("  PHYSICAL DISK FILE VERIFICATION")
    print("=" * 80)
    for p in [feature_csv_path, metrics_csv_path, summary_txt_path]:
        exists = p.exists()
        size_bytes = p.stat().st_size if exists else 0
        print(f"  [{'EXISTS' if exists else 'MISSING'}] {p.resolve()} ({size_bytes:,} bytes)")

    print("\n" + "=" * 80)
    print("  FEATURE MATRIX INTEGRITY VERIFICATION")
    print("=" * 80)
    print(f"  Feature CSV Shape       : {df_features.shape[0]} rows x {df_features.shape[1]} columns")
    print(f"  Metrics CSV Shape       : {df_metrics.shape[0]} rows x {df_metrics.shape[1]} columns")
    print(f"  Unique Filenames        : {df_features['filename'].nunique()} / {total_samples}")
    print(f"  NaN Count in Features   : {nan_count}")
    print(f"  Inf Count in Features   : {inf_count}")
    print(f"  Total Feature Count     : {len(feature_cols)}")

    print("\n" + "=" * 80)
    print("  SEGMENTATION EVALUATION METRICS (1,000 SAMPLES)")
    print("=" * 80)
    print(f"  Successful Segmentations: {success_count} / {total_samples} ({success_count / total_samples * 100.0:.2f}%)")
    print(f"  Failed Segmentations    : {fail_count} / {total_samples} ({fail_count / total_samples * 100.0:.2f}%)")
    print(f"  Mean Dice (DSC)         : {mean_dice:.4f}")
    print(f"  Median Dice (DSC)       : {median_dice:.4f}")
    print(f"  Mean IoU                : {mean_iou:.4f}")
    print(f"  Median IoU              : {median_iou:.4f}")
    print(f"  Mean Precision          : {mean_prec:.4f}")
    print(f"  Mean Recall             : {mean_rec:.4f}")

    print("\n" + "=" * 80)
    print("  STAGE A BATCH FEATURE EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

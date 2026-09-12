"""Stage A Batch Feature Extraction Pipeline for All Kvasir-SEG Images.

Processes all 1,000 Kvasir-SEG images through the complete Stage A pipeline:
- Preprocessing
- Active Contour Segmentation (without ground-truth)
- ROI Extraction
- Shape, Color, Texture Feature Extraction (37 features)
- Segmentation Evaluation (Dice, IoU, Precision, Recall)

Outputs:
- outputs/features/kvasir_stageA_features.csv (1,000 rows, 37+ feature columns)
- outputs/features/kvasir_stageA_segmentation_metrics.csv (1,000 rows, metrics)
- outputs/features/stageA_summary.txt (summary statistics)
- outputs/features/stageA_failures.csv (failure log if any failures occur)
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

logger = get_logger("batch_pipeline")


def process_batch(
    dataset: KvasirSEGDataset,
    pipeline: StageAFeaturePipeline,
    num_samples: int = None,
    desc: str = "Processing",
    output_dir: Path = None,
    checkpoint_interval: int = 100,
) -> tuple:
    """Processes a batch of images through the Stage A pipeline.

    Args:
        dataset: KvasirSEGDataset instance.
        pipeline: StageAFeaturePipeline instance.
        num_samples: Number of samples to process (None = all samples).
        desc: Description for progress bar.
        output_dir: Output directory for checkpoints.
        checkpoint_interval: Save progress every N samples.

    Returns:
        Tuple of (feature_rows, metrics_rows, failure_rows)
    """
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    feature_rows = []
    metrics_rows = []
    failure_rows = []

    for idx in tqdm(range(num_samples), desc=desc):
        try:
            # Load sample
            sample = dataset[idx]
            rgb_image = sample["image"]
            gt_mask = sample["mask"]
            filename = sample["filename"]

            # Process through pipeline
            result = pipeline.process_sample(
                rgb_image=rgb_image,
                filename=filename,
                gt_mask=gt_mask,
            )

            # Extract segmentation success status
            seg_success = result.get("seg_success", False)

            # Build feature row (exclude internal arrays and evaluation metrics)
            feat_dict = {"filename": filename, "segmentation_success": seg_success}

            # Extract only the 37 feature columns (shape, color, texture prefixes)
            feature_keys = [
                k for k in result.keys()
                if k.startswith(("shape_", "color_", "texture_"))
            ]
            for fk in sorted(feature_keys):
                feat_dict[fk] = result[fk]

            feature_rows.append(feat_dict)

            # Build metrics row
            metrics_dict = {
                "filename": filename,
                "dice": result.get("dice"),
                "iou": result.get("iou"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "segmentation_success": seg_success,
            }
            metrics_rows.append(metrics_dict)

        except Exception as e:
            # Log failure and continue
            failure_dict = {
                "filename": filename if 'filename' in locals() else f"index_{idx}",
                "stage": "pipeline_processing",
                "error": str(e),
            }
            failure_rows.append(failure_dict)
            logger.error(f"Failed to process sample {idx}: {e}")

            # Add placeholder row with NaN values for features
            feat_dict = {"filename": filename if 'filename' in locals() else f"index_{idx}",
                        "segmentation_success": False}
            # Get feature keys from pipeline defaults
            dummy_mask = np.zeros((10, 10), dtype=np.uint8)
            dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            from src.features.shape_features import extract_shape_features
            from src.features.color_features import extract_color_features
            from src.features.texture_features import extract_texture_features

            for k in extract_shape_features(dummy_mask).keys():
                feat_dict[k] = np.nan
            for k in extract_color_features(dummy_rgb, dummy_mask).keys():
                feat_dict[k] = np.nan
            for k in extract_texture_features(dummy_rgb, dummy_mask).keys():
                feat_dict[k] = np.nan

            feature_rows.append(feat_dict)

            # Add placeholder metrics row
            metrics_dict = {
                "filename": filename if 'filename' in locals() else f"index_{idx}",
                "dice": np.nan,
                "iou": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "segmentation_success": False,
            }
            metrics_rows.append(metrics_dict)

        # Save checkpoint every checkpoint_interval samples
        if output_dir and checkpoint_interval > 0 and (idx + 1) % checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_{idx+1}.csv"
            checkpoint_df = pd.DataFrame(feature_rows)
            checkpoint_df.to_csv(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved at {idx+1} samples: {checkpoint_path}")

    return feature_rows, metrics_rows, failure_rows


def save_outputs(
    feature_rows: list,
    metrics_rows: list,
    failure_rows: list,
    output_dir: Path,
) -> dict:
    """Saves all output CSV files and generates summary statistics.

    Args:
        feature_rows: List of feature dictionaries.
        metrics_rows: List of metrics dictionaries.
        failure_rows: List of failure dictionaries.
        output_dir: Output directory path.

    Returns:
        Dict containing summary statistics.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save feature CSV
    feature_df = pd.DataFrame(feature_rows)
    feature_csv_path = output_dir / "kvasir_stageA_features.csv"
    feature_df.to_csv(feature_csv_path, index=False)
    logger.info(f"Saved feature CSV: {feature_csv_path}")

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = output_dir / "kvasir_stageA_segmentation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved metrics CSV: {metrics_csv_path}")

    # Save failure log if any failures occurred
    if failure_rows:
        failure_df = pd.DataFrame(failure_rows)
        failure_csv_path = output_dir / "stageA_failures.csv"
        failure_df.to_csv(failure_csv_path, index=False)
        logger.warning(f"Saved failure log: {failure_csv_path} ({len(failure_rows)} failures)")

    # Calculate summary statistics
    total_images = len(feature_rows)
    successful_count = feature_df["segmentation_success"].sum()
    failed_count = total_images - successful_count

    # Identify feature columns (exclude filename and segmentation_success)
    feature_cols = [c for c in feature_df.columns 
                   if c not in ("filename", "segmentation_success")]
    num_features = len(feature_cols)

    # Count NaN and Inf values in feature columns
    nan_count = int(feature_df[feature_cols].isna().sum().sum())
    inf_count = int(np.isinf(feature_df[feature_cols].select_dtypes(include=[np.number]).values).sum())

    # Calculate metrics statistics (excluding NaN values)
    metrics_valid = metrics_df[metrics_df["segmentation_success"] == True]

    if len(metrics_valid) > 0:
        mean_dice = float(metrics_valid["dice"].mean())
        median_dice = float(metrics_valid["dice"].median())
        mean_iou = float(metrics_valid["iou"].mean())
        median_iou = float(metrics_valid["iou"].median())
        mean_precision = float(metrics_valid["precision"].mean())
        mean_recall = float(metrics_valid["recall"].mean())
    else:
        mean_dice = median_dice = mean_iou = median_iou = 0.0
        mean_precision = mean_recall = 0.0

    summary = {
        "total_images": total_images,
        "successful_segmentations": int(successful_count),
        "failed_segmentations": int(failed_count),
        "num_features": num_features,
        "feature_columns": feature_cols,
        "mean_dice": mean_dice,
        "median_dice": median_dice,
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "num_failures": len(failure_rows),
    }

    # Write summary text file
    summary_path = output_dir / "stageA_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE A BATCH PROCESSING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total images processed        : {summary['total_images']}\n")
        f.write(f"Successful segmentations      : {summary['successful_segmentations']}\n")
        f.write(f"Failed segmentations          : {summary['failed_segmentations']}\n")
        f.write(f"Number of extracted features  : {summary['num_features']}\n\n")
        
        f.write("SEGMENTATION METRICS (successful only):\n")
        f.write(f"  Mean Dice                   : {summary['mean_dice']:.4f}\n")
        f.write(f"  Median Dice                 : {summary['median_dice']:.4f}\n")
        f.write(f"  Mean IoU                    : {summary['mean_iou']:.4f}\n")
        f.write(f"  Median IoU                  : {summary['median_iou']:.4f}\n")
        f.write(f"  Mean Precision              : {summary['mean_precision']:.4f}\n")
        f.write(f"  Mean Recall                 : {summary['mean_recall']:.4f}\n\n")
        
        f.write("DATA QUALITY:\n")
        f.write(f"  NaN values in features      : {summary['nan_count']}\n")
        f.write(f"  Inf values in features      : {summary['inf_count']}\n")
        f.write(f"  Processing failures         : {summary['num_failures']}\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write(f"  Feature CSV                 : {feature_csv_path}\n")
        f.write(f"  Metrics CSV                 : {metrics_csv_path}\n")
        if failure_rows:
            f.write(f"  Failure log                 : {output_dir / 'stageA_failures.csv'}\n")
        f.write(f"  Summary file                : {summary_path}\n\n")
        
        f.write("FEATURE COLUMNS:\n")
        for i, col in enumerate(summary['feature_columns'], 1):
            f.write(f"  {i:2d}. {col}\n")
        
        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Saved summary file: {summary_path}")

    return summary


def validate_outputs(
    feature_csv_path: Path,
    metrics_csv_path: Path,
    expected_samples: int
) -> dict:
    """Validates the output CSV files.

    Args:
        feature_csv_path: Path to feature CSV file.
        metrics_csv_path: Path to metrics CSV file.
        expected_samples: Expected number of rows.

    Returns:
        Dict containing validation results.
    """
    validation = {}

    # Check feature CSV
    if feature_csv_path.exists():
        feature_df = pd.read_csv(feature_csv_path)
        validation["feature_csv_exists"] = True
        validation["feature_csv_rows"] = len(feature_df)
        validation["feature_csv_cols"] = len(feature_df.columns)
        validation["has_duplicates"] = feature_df["filename"].duplicated().any()

        # Check for expected 37 features
        feature_cols = [c for c in feature_df.columns 
                       if c not in ("filename", "segmentation_success")]
        validation["num_features"] = len(feature_cols)
        validation["expected_features"] = 37

        # Check for ground-truth leakage (should not have "gt_" prefix columns)
        gt_cols = [c for c in feature_df.columns if c.startswith("gt_")]
        validation["has_gt_leakage"] = len(gt_cols) > 0
        validation["gt_columns"] = gt_cols

    else:
        validation["feature_csv_exists"] = False

    # Check metrics CSV
    if metrics_csv_path.exists():
        metrics_df = pd.read_csv(metrics_csv_path)
        validation["metrics_csv_exists"] = True
        validation["metrics_csv_rows"] = len(metrics_df)
    else:
        validation["metrics_csv_exists"] = False

    return validation


def main(num_samples: int = None, dry_run: bool = False) -> None:
    """Main entry point for batch processing pipeline.

    Args:
        num_samples: Number of samples to process (None = all 1,000).
        dry_run: If True, process only 10 samples for testing.
    """
    print("=" * 80)
    print("  STAGE A BATCH FEATURE EXTRACTION PIPELINE")
    print("=" * 80)

    config = get_config()
    dataset = KvasirSEGDataset(config=config)
    pipeline = StageAFeaturePipeline(config=config)

    total_available = len(dataset)
    print(f"\nDataset loaded: {total_available} samples")

    # Determine processing mode
    if dry_run:
        num_samples = 10
        print(f"\nDRY RUN MODE: Processing first {num_samples} images for validation...")
    elif num_samples is None:
        num_samples = total_available
        print(f"\nFULL BATCH MODE: Processing all {num_samples} images...")
    else:
        num_samples = min(num_samples, total_available)
        print(f"\nCUSTOM BATCH MODE: Processing {num_samples} images...")

    output_dir = config.paths.features_dir
    print(f"Output directory: {output_dir}\n")

    # Process batch
    feature_rows, metrics_rows, failure_rows = process_batch(
        dataset=dataset,
        pipeline=pipeline,
        num_samples=num_samples,
        desc="Stage A Processing",
        output_dir=output_dir,
        checkpoint_interval=100,
    )

    print(f"\n{'=' * 80}")
    print("  SAVING OUTPUTS")
    print("=" * 80)

    # Save outputs and generate summary
    summary = save_outputs(
        feature_rows=feature_rows,
        metrics_rows=metrics_rows,
        failure_rows=failure_rows,
        output_dir=output_dir,
    )

    # Validate outputs
    print(f"\n{'=' * 80}")
    print("  OUTPUT VALIDATION")
    print("=" * 80)

    feature_csv_path = output_dir / "kvasir_stageA_features.csv"
    metrics_csv_path = output_dir / "kvasir_stageA_segmentation_metrics.csv"

    validation = validate_outputs(
        feature_csv_path=feature_csv_path,
        metrics_csv_path=metrics_csv_path,
        expected_samples=num_samples,
    )

    print(f"\nFeature CSV exists            : {validation.get('feature_csv_exists', False)}")
    if validation.get('feature_csv_exists'):
        print(f"Feature CSV rows              : {validation['feature_csv_rows']}")
        print(f"Feature CSV columns           : {validation['feature_csv_cols']}")
        print(f"Number of features            : {validation['num_features']}")
        print(f"Expected features             : {validation['expected_features']}")
        print(f"Has duplicate filenames       : {validation['has_duplicates']}")
        print(f"Has ground-truth leakage      : {validation['has_gt_leakage']}")
        if validation['has_gt_leakage']:
            print(f"  Ground-truth columns found  : {validation['gt_columns']}")

    print(f"\nMetrics CSV exists            : {validation.get('metrics_csv_exists', False)}")
    if validation.get('metrics_csv_exists'):
        print(f"Metrics CSV rows              : {validation['metrics_csv_rows']}")

    # Print summary statistics
    print(f"\n{'=' * 80}")
    print("  SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal images processed        : {summary['total_images']}")
    print(f"Successful segmentations      : {summary['successful_segmentations']}")
    print(f"Failed segmentations          : {summary['failed_segmentations']}")
    print(f"Number of features            : {summary['num_features']}")
    print(f"\nSegmentation Metrics (successful only):")
    print(f"  Mean Dice                   : {summary['mean_dice']:.4f}")
    print(f"  Median Dice                 : {summary['median_dice']:.4f}")
    print(f"  Mean IoU                    : {summary['mean_iou']:.4f}")
    print(f"  Median IoU                  : {summary['median_iou']:.4f}")
    print(f"  Mean Precision              : {summary['mean_precision']:.4f}")
    print(f"  Mean Recall                 : {summary['mean_recall']:.4f}")
    print(f"\nData Quality:")
    print(f"  NaN count                   : {summary['nan_count']}")
    print(f"  Inf count                   : {summary['inf_count']}")
    print(f"  Processing failures         : {summary['num_failures']}")

    print(f"\n{'=' * 80}")
    print("  OUTPUT FILE PATHS")
    print("=" * 80)
    print(f"\nFeature CSV  : {feature_csv_path.resolve()}")
    print(f"Metrics CSV  : {metrics_csv_path.resolve()}")
    print(f"Summary file : {(output_dir / 'stageA_summary.txt').resolve()}")
    if failure_rows:
        print(f"Failure log  : {(output_dir / 'stageA_failures.csv').resolve()}")

    print(f"\n{'=' * 80}")
    if dry_run:
        print("  DRY RUN COMPLETE - Review outputs before running full batch")
    else:
        print("  STAGE A BATCH PROCESSING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage A Batch Feature Extraction Pipeline"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 10 samples for validation",
    )

    args = parser.parse_args()
    main(num_samples=args.num_samples, dry_run=args.dry_run)

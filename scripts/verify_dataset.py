"""Dataset Verification CLI Script.

Executes 100% integrity verification across the Kvasir-SEG dataset, verifying
image-mask pair matching, spatial dimensions, and printing comprehensive summary statistics.
"""

import sys
from pathlib import Path

# Ensure project root is in Python path for execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config
from src.data.kvasir_dataset import KvasirSEGDataset
from src.utils.logger import get_logger

logger = get_logger("verify_script")


def main() -> None:
    """Main verification routine."""
    logger.info("Starting Dataset Ingestion & Pair Matching Verification...")

    config = get_config()
    print(f"\n=======================================================")
    print(f"  ACTIVE CONTOUR POLYP PIPELINE - DATASET VERIFICATION ")
    print(f"=======================================================")
    print(f"Dataset Root Dir : {config.paths.kvasir_root}")
    print(f"Images Directory : {config.paths.kvasir_images_dir}")
    print(f"Masks Directory  : {config.paths.kvasir_masks_dir}")
    print(f"BBoxes File Path : {config.paths.kvasir_bboxes_path}")
    print(f"-------------------------------------------------------\n")

    # Initialize Dataset
    dataset = KvasirSEGDataset(config=config)

    # 1. Total Number of Images
    total_images = len(dataset)
    print(f"1. TOTAL NUMBER OF IMAGES & MASKS: {total_images}")

    if total_images == 0:
        logger.error("No images found! Please check the dataset directory path.")
        sys.exit(1)

    # 2. Pair Integrity Verification
    print("\n2. VERIFYING IMAGE-MASK MATCHING & CORRUPTION CHECKS:")
    is_valid, valid_count, summary = dataset.verify_integrity()
    print(f"   - Verified Pairs  : {valid_count} / {total_images}")
    print(f"   - Corrupted Pairs : {summary['corrupted_count']}")
    print(f"   - Matching Status : {'SUCCESS (100% Valid)' if is_valid else 'FAILED'}")

    # 3. Dimensions & Spatial Breakdown
    print("\n3. IMAGE AND MASK DIMENSIONS BREAKDOWN:")
    stats = dataset.get_dataset_stats()
    print(f"   - Min Resolution (Width x Height) : {stats['min_dimension'][0]} x {stats['min_dimension'][1]}")
    print(f"   - Max Resolution (Width x Height) : {stats['max_dimension'][0]} x {stats['max_dimension'][1]}")
    print(f"   - Average Dimensions             : {stats['avg_width']:.1f} x {stats['avg_height']:.1f}")
    print(f"   - Unique Spatial Resolutions     : {stats['unique_resolutions_count']} distinct shapes")

    # 4. Polyp Mask Coverage Statistics
    print("\n4. GROUND TRUTH POLYP COVERAGE (% OF TOTAL IMAGE AREA):")
    print(f"   - Min Polyp Coverage : {stats['min_polyp_coverage_pct']:.2f}%")
    print(f"   - Max Polyp Coverage : {stats['max_polyp_coverage_pct']:.2f}%")
    print(f"   - Mean Polyp Coverage: {stats['mean_polyp_coverage_pct']:.2f}%")

    # 5. Display Sample Image-Mask Pairs
    print("\n5. SAMPLE IMAGE-MASK PAIRS (FIRST 5 SAMPLES):")
    print("-" * 90)
    print(f"{'Sample #':<10} | {'Filename':<35} | {'Image Shape':<15} | {'Mask Shape':<15}")
    print("-" * 90)

    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        img_shape_str = f"{sample['image'].shape[1]}x{sample['image'].shape[0]}x{sample['image'].shape[2]}"
        msk_shape_str = f"{sample['mask'].shape[1]}x{sample['mask'].shape[0]}"
        print(f"Sample #{i+1:<3} | {sample['filename']:<35} | {img_shape_str:<15} | {msk_shape_str:<15}")

    print("-" * 90)

    if is_valid:
        print("\n[SUCCESS] Phase 1 Foundation Verified! All image-mask pairs are aligned and ready for active contours.")
    else:
        print("\n[WARNING] Dataset verification encountered errors. Please check the log files.")


if __name__ == "__main__":
    main()

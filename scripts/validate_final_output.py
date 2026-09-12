"""Final validation of Stage A outputs."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("FINAL STAGE A OUTPUT VALIDATION")
    print("=" * 80)
    
    # Load CSVs
    feat_path = PROJECT_ROOT / "outputs" / "features" / "kvasir_stageA_features.csv"
    metrics_path = PROJECT_ROOT / "outputs" / "features" / "kvasir_stageA_segmentation_metrics.csv"
    summary_path = PROJECT_ROOT / "outputs" / "features" / "stageA_summary.txt"
    
    df_feat = pd.read_csv(feat_path)
    df_metrics = pd.read_csv(metrics_path)
    
    # Feature CSV validation
    print(f"\n1. FEATURE CSV: {feat_path.name}")
    print(f"   Path: {feat_path}")
    print(f"   Dimensions: {df_feat.shape}")
    print(f"   Rows: {len(df_feat)}")
    print(f"   Columns: {len(df_feat.columns)}")
    feature_cols = [c for c in df_feat.columns if c.startswith(("shape_", "color_", "texture_"))]
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Duplicate filenames: {df_feat['filename'].duplicated().sum()}")
    numeric_cols = df_feat.select_dtypes(include=[np.number])
    print(f"   NaN count: {numeric_cols.isna().sum().sum()}")
    print(f"   Inf count: {np.isinf(numeric_cols.values).sum()}")
    
    # Metrics CSV validation
    print(f"\n2. METRICS CSV: {metrics_path.name}")
    print(f"   Path: {metrics_path}")
    print(f"   Dimensions: {df_metrics.shape}")
    print(f"   Rows: {len(df_metrics)}")
    print(f"   Successful segmentations: {df_metrics['segmentation_success'].sum()}")
    print(f"   Failed segmentations: {(~df_metrics['segmentation_success']).sum()}")
    
    # Segmentation metrics
    print(f"\n3. SEGMENTATION METRICS (successful only):")
    valid = df_metrics[df_metrics['segmentation_success'] == True]
    print(f"   Mean Dice: {valid['dice'].mean():.4f}")
    print(f"   Median Dice: {valid['dice'].median():.4f}")
    print(f"   Mean IoU: {valid['iou'].mean():.4f}")
    print(f"   Median IoU: {valid['iou'].median():.4f}")
    print(f"   Mean Precision: {valid['precision'].mean():.4f}")
    print(f"   Mean Recall: {valid['recall'].mean():.4f}")
    
    # Sample data
    print(f"\n4. SAMPLE FEATURE VALUES (first 3 images):")
    sample_cols = ['filename', 'segmentation_success', 'shape_area', 'color_rgb_r_mean', 'texture_glcm_contrast']
    print(df_feat.head(3)[sample_cols].to_string(index=False))
    
    # Data quality checks
    print(f"\n5. DATA QUALITY CHECKS:")
    all_unique = df_feat['filename'].nunique() == len(df_feat)
    all_finite = np.isfinite(numeric_cols.values).all()
    no_gt_leakage = not any(c.startswith("gt_") for c in df_feat.columns)
    
    print(f"   ✓ All filenames unique: {all_unique}")
    print(f"   ✓ All features finite: {all_finite}")
    print(f"   ✓ No ground-truth leakage: {no_gt_leakage}")
    print(f"   ✓ Summary file exists: {summary_path.exists()}")
    
    # Final status
    print(f"\n{'=' * 80}")
    if all([
        len(df_feat) == 1000,
        len(df_metrics) == 1000,
        len(feature_cols) == 37,
        all_unique,
        all_finite,
        no_gt_leakage,
        summary_path.exists()
    ]):
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("✅ STAGE A BATCH PROCESSING SUCCESSFULLY COMPLETED")
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
    print("=" * 80)

if __name__ == "__main__":
    main()

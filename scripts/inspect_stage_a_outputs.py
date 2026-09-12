"""Detailed inspection of Stage A outputs - read-only analysis."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("STAGE A OUTPUT INSPECTION REPORT")
    print("=" * 80)
    
    # Load CSVs
    feat_path = PROJECT_ROOT / "outputs" / "features" / "kvasir_stageA_features.csv"
    metrics_path = PROJECT_ROOT / "outputs" / "features" / "kvasir_stageA_segmentation_metrics.csv"
    
    df_feat = pd.read_csv(feat_path)
    df_metrics = pd.read_csv(metrics_path)
    
    # 1. Feature CSV Dimensions
    print(f"\n1. FEATURE CSV DIMENSIONS")
    print(f"   Rows × Columns: {df_feat.shape[0]} × {df_feat.shape[1]}")
    
    # 2. Complete list of feature columns
    print(f"\n2. COMPLETE LIST OF COLUMNS ({len(df_feat.columns)} total)")
    for i, col in enumerate(df_feat.columns, 1):
        marker = "→" if col.startswith(("shape_", "color_", "texture_")) else " "
        print(f"   {i:2d}. {marker} {col}")
    
    # Identify feature columns only
    feature_cols = [c for c in df_feat.columns if c.startswith(("shape_", "color_", "texture_"))]
    print(f"\n   Feature columns only: {len(feature_cols)}")
    
    # 3-4. Successful and failed segmentations
    print(f"\n3. SUCCESSFUL SEGMENTATIONS: {df_feat['segmentation_success'].sum()}")
    print(f"\n4. FAILED SEGMENTATIONS: {(~df_feat['segmentation_success']).sum()}")
    
    # 5-10. Segmentation metrics
    print(f"\n5. MEAN DICE: {df_metrics['dice'].mean():.4f}")
    print(f"\n6. MEDIAN DICE: {df_metrics['dice'].median():.4f}")
    print(f"\n7. MEAN IoU: {df_metrics['iou'].mean():.4f}")
    print(f"\n8. MEDIAN IoU: {df_metrics['iou'].median():.4f}")
    print(f"\n9. MEAN PRECISION: {df_metrics['precision'].mean():.4f}")
    print(f"\n10. MEAN RECALL: {df_metrics['recall'].mean():.4f}")
    
    # 11-12. NaN and Inf counts
    feature_data = df_feat[feature_cols]
    nan_count = feature_data.isna().sum().sum()
    inf_count = np.isinf(feature_data.values).sum()
    
    print(f"\n11. TOTAL NaN VALUES: {nan_count}")
    print(f"\n12. TOTAL Inf VALUES: {inf_count}")
    
    # 13. Min and max for each numerical feature
    print(f"\n13. NUMERICAL FEATURE RANGES (Min → Max)")
    print(f"    {'Feature':<35} {'Min':>15} {'Max':>15}")
    print(f"    {'-' * 67}")
    
    for col in sorted(feature_cols):
        min_val = feature_data[col].min()
        max_val = feature_data[col].max()
        print(f"    {col:<35} {min_val:>15.4f} {max_val:>15.4f}")
    
    # 14. Duplicate filenames
    duplicates = df_feat['filename'].duplicated().sum()
    print(f"\n14. DUPLICATE FILENAMES: {duplicates}")
    if duplicates == 0:
        print(f"    ✓ All {df_feat['filename'].nunique()} filenames are unique")
    else:
        print(f"    ✗ Found {duplicates} duplicate(s)")
    
    # 15. First 5 rows
    print(f"\n15. FIRST 5 ROWS OF FEATURE CSV")
    print(f"\n    Non-feature columns:")
    non_feature_cols = ['filename', 'segmentation_success']
    print(df_feat[non_feature_cols].head(5).to_string(index=True))
    
    print(f"\n    Sample of feature columns (first 8 features):")
    sample_features = sorted(feature_cols)[:8]
    print(df_feat[sample_features].head(5).to_string(index=True))
    
    print(f"\n    (Showing {len(sample_features)} of {len(feature_cols)} feature columns)")
    
    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nFeature Statistics:")
    print(df_feat[feature_cols].describe().to_string())
    
    print(f"\n{'=' * 80}")
    print("END OF INSPECTION REPORT")
    print("=" * 80)

if __name__ == "__main__":
    main()

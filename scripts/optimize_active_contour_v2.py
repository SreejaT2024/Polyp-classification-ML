"""Active Contour Error Analysis and Optimization - Version 2.

Tests more aggressive parameter variations and alternative initialization strategies.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm

from config.config import get_config
from src.data.kvasir_dataset import KvasirSEGDataset
from src.preprocessing.image_preprocess import PolypImagePreprocessor
from src.segmentation.active_contour import ActiveContourSegmenter
from src.segmentation.metrics import evaluate_segmentation


def select_representative_images(dataset: KvasirSEGDataset, metrics_df: pd.DataFrame) -> List[int]:
    """Select 10 representative images."""
    sorted_df = metrics_df.sort_values('dice', ascending=False).reset_index(drop=True)
    
    good = sorted_df[sorted_df['dice'] > 0.5].head(3)
    moderate = sorted_df[(sorted_df['dice'] >= 0.25) & (sorted_df['dice'] <= 0.35)].head(4)
    poor = sorted_df[sorted_df['dice'] < 0.2].head(3)
    
    selected_indices = []
    for _, row in pd.concat([good, moderate, poor]).iterrows():
        filename = row['filename']
        for idx in range(len(dataset)):
            if dataset[idx]['filename'] == filename:
                selected_indices.append(idx)
                break
    
    return selected_indices[:10]


def test_configuration(
    dataset: KvasirSEGDataset,
    preprocessor: PolypImagePreprocessor,
    indices: List[int],
    config_name: str,
    **segmenter_kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """Test a specific active contour configuration."""
    segmenter = ActiveContourSegmenter(**segmenter_kwargs)
    results = []
    
    for idx in indices:
        sample = dataset[idx]
        rgb_image = sample['image']
        gt_mask = sample['mask']
        filename = sample['filename']
        
        preprocessed = preprocessor.preprocess_for_segmentation(rgb_image)
        pred_mask, success, info = segmenter.segment(preprocessed, rgb_image=rgb_image)
        
        metrics = evaluate_segmentation(pred_mask, gt_mask)
        
        results.append({
            'config': config_name,
            'filename': filename,
            'dice': metrics['dice'],
            'iou': metrics['iou'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'seg_success': success,
        })
    
    results_df = pd.DataFrame(results)
    
    summary = {
        'config': config_name,
        'mean_dice': results_df['dice'].mean(),
        'median_dice': results_df['dice'].median(),
        'mean_iou': results_df['iou'].mean(),
        'median_iou': results_df['iou'].median(),
        'mean_precision': results_df['precision'].mean(),
        'mean_recall': results_df['recall'].mean(),
        'successful_count': results_df['seg_success'].sum(),
        **segmenter_kwargs
    }
    
    return results_df, summary


def main():
    print("=" * 80)
    print("ACTIVE CONTOUR OPTIMIZATION - VERSION 2 (AGGRESSIVE VARIATIONS)")
    print("=" * 80)
    
    config = get_config()
    dataset = KvasirSEGDataset(config=config)
    preprocessor = PolypImagePreprocessor()
    
    metrics_path = PROJECT_ROOT / "outputs" / "features" / "kvasir_stageA_segmentation_metrics.csv"
    metrics_df = pd.read_csv(metrics_path)
    
    print(f"\nCurrent Performance (1000 images):")
    print(f"  Mean Dice: {metrics_df['dice'].mean():.4f}")
    print(f"  Mean Precision: {metrics_df['precision'].mean():.4f}")
    print(f"  Mean Recall: {metrics_df['recall'].mean():.4f}")
    
    selected_indices = select_representative_images(dataset, metrics_df)
    
    print(f"\nSelected {len(selected_indices)} test images")
    
    # Test more aggressive variations
    configurations = {
        'Current': {
            'max_iterations': 80,
            'smoothing': 1,
            'lambda1': 1.0,
            'lambda2': 1.5,
            'init_type': 'otsu_seed',
        },
        'HighLambda2': {
            'max_iterations': 80,
            'smoothing': 1,
            'lambda1': 1.0,
            'lambda2': 5.0,  # Much higher penalty for expansion
            'init_type': 'otsu_seed',
        },
        'CenterDisk': {
            'max_iterations': 80,
            'smoothing': 1,
            'lambda1': 1.0,
            'lambda2': 1.5,
            'init_type': 'center_disk',  # Different initialization
        },
        'LowIterations': {
            'max_iterations': 30,  # Much fewer iterations
            'smoothing': 1,
            'lambda1': 1.0,
            'lambda2': 1.5,
            'init_type': 'otsu_seed',
        },
        'HighSmoothing': {
            'max_iterations': 80,
            'smoothing': 5,  # Much more smoothing
            'lambda1': 1.0,
            'lambda2': 1.5,
            'init_type': 'otsu_seed',
        },
        'Balanced': {
            'max_iterations': 80,
            'smoothing': 3,
            'lambda1': 1.0,
            'lambda2': 3.0,
            'init_type': 'otsu_seed',
        },
        'HighLambda1': {
            'max_iterations': 80,
            'smoothing': 1,
            'lambda1': 2.0,  # Higher weight for inside region
            'lambda2': 1.5,
            'init_type': 'otsu_seed',
        },
    }
    
    print("\n" + "=" * 80)
    print("TESTING CONFIGURATIONS")
    print("=" * 80)
    
    all_results = []
    all_summaries = []
    
    for config_name, config_kwargs in configurations.items():
        print(f"\nTesting {config_name}...")
        print(f"  Parameters: {config_kwargs}")
        
        results_df, summary = test_configuration(
            dataset, preprocessor, selected_indices, config_name, **config_kwargs
        )
        
        all_results.append(results_df)
        all_summaries.append(summary)
        
        print(f"  Mean Dice: {summary['mean_dice']:.4f}  "
              f"Mean Precision: {summary['mean_precision']:.4f}  "
              f"Mean Recall: {summary['mean_recall']:.4f}")
    
    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    
    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "stage_a_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "contour_optimization_v2_results.csv"
    combined_results.to_csv(results_path, index=False)
    
    summary_path = output_dir / "contour_optimization_v2_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    print("\n" + summary_df[['config', 'mean_dice', 'mean_iou', 'mean_precision', 'mean_recall']].to_string(index=False))
    
    # Find best config
    best_config = summary_df.loc[summary_df['mean_dice'].idxmax()]
    current_config = summary_df[summary_df['config'] == 'Current'].iloc[0]
    
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"\nConfiguration: {best_config['config']}")
    print(f"Mean Dice: {best_config['mean_dice']:.4f} (Current: {current_config['mean_dice']:.4f})")
    print(f"Mean IoU: {best_config['mean_iou']:.4f} (Current: {current_config['mean_iou']:.4f})")
    print(f"Mean Precision: {best_config['mean_precision']:.4f} (Current: {current_config['mean_precision']:.4f})")
    print(f"Mean Recall: {best_config['mean_recall']:.4f} (Current: {current_config['mean_recall']:.4f})")
    
    print(f"\n✓ Saved results: {results_path}")
    print(f"✓ Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

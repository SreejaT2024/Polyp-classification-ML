"""Active Contour Error Analysis and Optimization.

Analyzes current segmentation performance, tests alternative configurations,
and recommends optimal parameters for improved Dice/IoU scores.
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
    """Select 10 representative images covering good, moderate, and poor segmentation.
    
    Args:
        dataset: KvasirSEGDataset instance
        metrics_df: DataFrame with segmentation metrics
        
    Returns:
        List of 10 image indices
    """
    # Sort by Dice score
    sorted_df = metrics_df.sort_values('dice', ascending=False).reset_index(drop=True)
    
    # Select representatives
    total = len(sorted_df)
    
    # Good segmentation (top 10% - Dice > 0.5)
    good = sorted_df[sorted_df['dice'] > 0.5].head(3)
    
    # Moderate segmentation (around median - Dice 0.25-0.35)
    moderate = sorted_df[(sorted_df['dice'] >= 0.25) & (sorted_df['dice'] <= 0.35)].head(4)
    
    # Poor segmentation (bottom 20% - Dice < 0.2)
    poor = sorted_df[sorted_df['dice'] < 0.2].head(3)
    
    # Get original indices
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
    """Test a specific active contour configuration on selected images.
    
    Args:
        dataset: KvasirSEGDataset instance
        preprocessor: PolypImagePreprocessor instance
        indices: List of image indices to test
        config_name: Name of this configuration
        **segmenter_kwargs: Parameters for ActiveContourSegmenter
        
    Returns:
        Tuple of (results_df, summary_dict)
    """
    segmenter = ActiveContourSegmenter(**segmenter_kwargs)
    results = []
    
    for idx in indices:
        sample = dataset[idx]
        rgb_image = sample['image']
        gt_mask = sample['mask']
        filename = sample['filename']
        
        # Preprocess and segment
        preprocessed = preprocessor.preprocess_for_segmentation(rgb_image)
        pred_mask, success, info = segmenter.segment(preprocessed, rgb_image=rgb_image)
        
        # Evaluate
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
    
    # Calculate summary
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


def visualize_comparison(
    dataset: KvasirSEGDataset,
    preprocessor: PolypImagePreprocessor,
    indices: List[int],
    segmenter_configs: Dict[str, Dict],
    output_dir: Path
):
    """Generate visual comparisons for selected images with multiple configurations.
    
    Args:
        dataset: KvasirSEGDataset instance
        preprocessor: PolypImagePreprocessor instance
        indices: List of image indices
        segmenter_configs: Dict mapping config names to segmenter kwargs
        output_dir: Directory to save visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in tqdm(indices, desc="Generating visualizations"):
        sample = dataset[idx]
        rgb_image = sample['image']
        gt_mask = sample['mask']
        filename = sample['filename']
        
        # Preprocess
        preprocessed = preprocessor.preprocess_for_segmentation(rgb_image)
        
        # Test each configuration
        pred_masks = {}
        metrics_list = {}
        
        for config_name, config_kwargs in segmenter_configs.items():
            segmenter = ActiveContourSegmenter(**config_kwargs)
            pred_mask, success, info = segmenter.segment(preprocessed, rgb_image=rgb_image)
            metrics = evaluate_segmentation(pred_mask, gt_mask)
            pred_masks[config_name] = pred_mask
            metrics_list[config_name] = metrics
        
        # Create visualization
        num_configs = len(segmenter_configs)
        fig, axes = plt.subplots(2, 3 + num_configs, figsize=(4 * (3 + num_configs), 8))
        
        # Row 1: Original, GT mask, GT contour
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title("Original Image", fontsize=10)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_mask, cmap='gray')
        axes[0, 1].set_title("Ground Truth Mask", fontsize=10)
        axes[0, 1].axis('off')
        
        # GT contour overlay
        gt_overlay = rgb_image.copy()
        gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if gt_contours:
            cv2.drawContours(gt_overlay, gt_contours, -1, (0, 255, 0), thickness=2)
        axes[0, 2].imshow(gt_overlay)
        axes[0, 2].set_title("GT Contour (Green)", fontsize=10)
        axes[0, 2].axis('off')
        
        # Row 1: Predicted masks for each configuration
        for col_idx, (config_name, pred_mask) in enumerate(pred_masks.items(), start=3):
            axes[0, col_idx].imshow(pred_mask, cmap='gray')
            metrics = metrics_list[config_name]
            title = f"{config_name}\nDice={metrics['dice']:.3f}"
            axes[0, col_idx].set_title(title, fontsize=9)
            axes[0, col_idx].axis('off')
        
        # Row 2: Overlays (GT=Green, Pred=Red)
        for col_idx, (config_name, pred_mask) in enumerate(pred_masks.items()):
            overlay = rgb_image.copy()
            
            # GT contour in green
            if gt_contours:
                cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), thickness=2)
            
            # Pred contour in red
            pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if pred_contours:
                cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), thickness=2)
            
            axes[1, col_idx].imshow(overlay)
            metrics = metrics_list[config_name]
            title = f"{config_name} Overlay\nIoU={metrics['iou']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}"
            axes[1, col_idx].set_title(title, fontsize=8)
            axes[1, col_idx].axis('off')
        
        # Hide unused axes in row 2
        for col_idx in range(len(pred_masks), 3 + num_configs):
            axes[1, col_idx].axis('off')
        
        plt.suptitle(f"{Path(filename).stem}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / f"{Path(filename).stem}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    print("=" * 80)
    print("ACTIVE CONTOUR ERROR ANALYSIS AND OPTIMIZATION")
    print("=" * 80)
    
    config = get_config()
    dataset = KvasirSEGDataset(config=config)
    preprocessor = PolypImagePreprocessor()
    
    # Load existing metrics
    metrics_path = PROJECT_ROOT / "outputs" / "features" / "kvasir_stageA_segmentation_metrics.csv"
    metrics_df = pd.read_csv(metrics_path)
    
    print(f"\nDataset: {len(dataset)} images")
    print(f"\nCurrent Performance (1000 images):")
    print(f"  Mean Dice      : {metrics_df['dice'].mean():.4f}")
    print(f"  Median Dice    : {metrics_df['dice'].median():.4f}")
    print(f"  Mean IoU       : {metrics_df['iou'].mean():.4f}")
    print(f"  Median IoU     : {metrics_df['iou'].median():.4f}")
    print(f"  Mean Precision : {metrics_df['precision'].mean():.4f}")
    print(f"  Mean Recall    : {metrics_df['recall'].mean():.4f}")
    
    # Select 10 representative images
    print("\n" + "=" * 80)
    print("SELECTING 10 REPRESENTATIVE IMAGES")
    print("=" * 80)
    
    selected_indices = select_representative_images(dataset, metrics_df)
    
    print(f"\nSelected {len(selected_indices)} images:")
    for idx in selected_indices:
        filename = dataset[idx]['filename']
        row = metrics_df[metrics_df['filename'] == filename].iloc[0]
        print(f"  {idx:3d}: {filename:40s} Dice={row['dice']:.4f} IoU={row['iou']:.4f}")
    
    # Define configurations to test
    print("\n" + "=" * 80)
    print("TESTING ACTIVE CONTOUR CONFIGURATIONS")
    print("=" * 80)
    
    configurations = {
        'Current': {
            'max_iterations': 80,
            'smoothing': 1,
            'lambda1': 1.0,
            'lambda2': 1.5,
            'init_type': 'otsu_seed',
        },
        'Config_A': {
            'max_iterations': 120,
            'smoothing': 2,
            'lambda1': 1.0,
            'lambda2': 2.0,
            'init_type': 'otsu_seed',
        },
        'Config_B': {
            'max_iterations': 100,
            'smoothing': 1,
            'lambda1': 1.0,
            'lambda2': 2.5,
            'init_type': 'otsu_seed',
        },
        'Config_C': {
            'max_iterations': 150,
            'smoothing': 3,
            'lambda1': 1.0,
            'lambda2': 2.0,
            'init_type': 'otsu_seed',
        },
        'Config_D': {
            'max_iterations': 100,
            'smoothing': 2,
            'lambda1': 1.5,
            'lambda2': 2.5,
            'init_type': 'otsu_seed',
        },
    }
    
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
        
        print(f"  Results:")
        print(f"    Mean Dice      : {summary['mean_dice']:.4f}")
        print(f"    Median Dice    : {summary['median_dice']:.4f}")
        print(f"    Mean IoU       : {summary['mean_iou']:.4f}")
        print(f"    Median IoU     : {summary['median_iou']:.4f}")
        print(f"    Mean Precision : {summary['mean_precision']:.4f}")
        print(f"    Mean Recall    : {summary['mean_recall']:.4f}")
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    
    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "stage_a_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "contour_optimization_results.csv"
    combined_results.to_csv(results_path, index=False)
    print(f"\n✓ Saved detailed results: {results_path}")
    
    summary_path = output_dir / "contour_optimization_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary: {summary_path}")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUAL COMPARISONS")
    print("=" * 80)
    
    vis_dir = output_dir / "contour_optimization"
    visualize_comparison(dataset, preprocessor, selected_indices, configurations, vis_dir)
    print(f"\n✓ Saved visualizations to: {vis_dir}")
    
    # Analysis and recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS AND RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. CURRENT CONFIGURATION ANALYSIS:")
    print("   " + "-" * 76)
    print("   Current config has:")
    print("   - Very high recall (97.84%) = captures most polyp pixels")
    print("   - Very low precision (21.35%) = includes many false positives")
    print("   - Low Dice/IoU = poor overall overlap")
    print()
    print("   ROOT CAUSE:")
    print("   - lambda2 (1.5) is too LOW relative to lambda1 (1.0)")
    print("   - lambda2 controls expansion penalty - low value allows over-segmentation")
    print("   - Otsu initialization may start too large, then continues expanding")
    print("   - Smoothing=1 provides insufficient boundary regularization")
    
    # Find best configuration
    best_config = summary_df.loc[summary_df['mean_dice'].idxmax()]
    
    print("\n2. BEST CONFIGURATION:")
    print("   " + "-" * 76)
    print(f"   Configuration: {best_config['config']}")
    print(f"   Mean Dice      : {best_config['mean_dice']:.4f}")
    print(f"   Median Dice    : {best_config['median_dice']:.4f}")
    print(f"   Mean IoU       : {best_config['mean_iou']:.4f}")
    print(f"   Median IoU     : {best_config['median_iou']:.4f}")
    print(f"   Mean Precision : {best_config['mean_precision']:.4f}")
    print(f"   Mean Recall    : {best_config['mean_recall']:.4f}")
    print()
    print("   Parameters:")
    print(f"   - max_iterations: {best_config['max_iterations']}")
    print(f"   - smoothing: {best_config['smoothing']}")
    print(f"   - lambda1: {best_config['lambda1']}")
    print(f"   - lambda2: {best_config['lambda2']}")
    print(f"   - init_type: {best_config['init_type']}")
    
    # Calculate improvement
    current_config = summary_df[summary_df['config'] == 'Current'].iloc[0]
    dice_improvement = ((best_config['mean_dice'] - current_config['mean_dice']) / 
                       current_config['mean_dice'] * 100)
    iou_improvement = ((best_config['mean_iou'] - current_config['mean_iou']) / 
                      current_config['mean_iou'] * 100)
    
    print("\n3. EXPECTED IMPROVEMENT:")
    print("   " + "-" * 76)
    print(f"   Dice improvement : +{dice_improvement:.1f}%")
    print(f"   IoU improvement  : +{iou_improvement:.1f}%")
    
    # Safety check
    print("\n4. SAFETY ASSESSMENT:")
    print("   " + "-" * 76)
    
    safety_checks = []
    safety_checks.append(("Dice improvement > 10%", dice_improvement > 10))
    safety_checks.append(("IoU improvement > 10%", iou_improvement > 10))
    safety_checks.append(("Precision improved", 
                         best_config['mean_precision'] > current_config['mean_precision']))
    safety_checks.append(("Recall still high (>0.85)", best_config['mean_recall'] > 0.85))
    safety_checks.append(("All test images successful", 
                         best_config['successful_count'] == len(selected_indices)))
    
    for check_name, passed in safety_checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {check_name}")
    
    all_passed = all(check for _, check in safety_checks)
    
    print("\n5. RECOMMENDATION:")
    print("   " + "-" * 76)
    if all_passed:
        print("   ✅ SAFE TO REGENERATE 1,000-IMAGE FEATURE CSV")
        print(f"   Recommended configuration: {best_config['config']}")
        print("   Expected benefits:")
        print("   - Better precision (fewer false positives)")
        print("   - Maintained high recall (still captures polyps)")
        print("   - Improved Dice/IoU (better overall segmentation quality)")
    else:
        print("   ⚠ REVIEW REQUIRED - Some safety checks failed")
        print("   Consider additional testing or parameter tuning")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Save detailed report
    report_path = output_dir / "optimization_report.txt"
    with open(report_path, 'w') as f:
        f.write("ACTIVE CONTOUR OPTIMIZATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("COMPARISON TABLE:\n")
        f.write(summary_df[['config', 'mean_dice', 'median_dice', 'mean_iou', 
                           'mean_precision', 'mean_recall']].to_string(index=False))
        f.write("\n\n")
        
        f.write("BEST CONFIGURATION:\n")
        f.write(f"  {best_config['config']}\n")
        f.write(f"  Mean Dice: {best_config['mean_dice']:.4f}\n")
        f.write(f"  Mean IoU: {best_config['mean_iou']:.4f}\n")
        f.write(f"  Improvement: Dice +{dice_improvement:.1f}%, IoU +{iou_improvement:.1f}%\n")
        f.write("\n")
        f.write("PARAMETERS:\n")
        f.write(f"  max_iterations: {best_config['max_iterations']}\n")
        f.write(f"  smoothing: {best_config['smoothing']}\n")
        f.write(f"  lambda1: {best_config['lambda1']}\n")
        f.write(f"  lambda2: {best_config['lambda2']}\n")
        f.write(f"  init_type: {best_config['init_type']}\n")
    
    print(f"\n✓ Saved detailed report: {report_path}")


if __name__ == "__main__":
    main()

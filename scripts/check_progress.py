"""Quick progress checker for Stage A batch processing."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

def check_progress():
    """Check progress of batch processing."""
    output_dir = PROJECT_ROOT / "outputs" / "features"
    
    # Check for checkpoint files
    checkpoints = sorted(output_dir.glob("checkpoint_*.csv"))
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        df = pd.read_csv(latest_checkpoint)
        print(f"Latest checkpoint: {latest_checkpoint.name}")
        print(f"Samples processed: {len(df)} / 1000")
        print(f"Progress: {len(df) / 10.0:.1f}%")
    else:
        # Check main feature file
        feature_file = output_dir / "kvasir_stageA_features.csv"
        if feature_file.exists():
            df = pd.read_csv(feature_file)
            print(f"Feature file exists: {len(df)} rows")
        else:
            print("No progress files found yet")

if __name__ == "__main__":
    check_progress()

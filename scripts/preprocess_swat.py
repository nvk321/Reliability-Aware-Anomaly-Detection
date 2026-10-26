"""
Script to preprocess raw SWaT dataset into standardized training, validation,
and test sets with leak-free robust scaling and metadata tracking.
"""

import argparse
import sys
import os

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.swat_loader import SWaTLoader


def main():
    parser = argparse.ArgumentParser(description="Preprocess SWaT dataset")
    parser.add_argument("--raw_dir", type=str, default="reference/swat_data", help="Path to raw SWaT files")
    parser.add_argument("--processed_dir", type=str, default="data/processed/SWaT", help="Target output dir")
    parser.add_argument("--downsample_rate", type=int, default=5, help="Downsampling rate (5 or 10)")
    parser.add_argument("--train_ratio", type=float, default=0.80, help="Train/Val chronological split ratio")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if cached")

    args = parser.parse_args()

    loader = SWaTLoader(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        downsample_rate=args.downsample_rate,
        train_ratio=args.train_ratio,
    )

    train, val, test, test_labels = loader.load_and_preprocess(force_recompute=args.force)

    print("\n[SUCCESS] SWaT Preprocessing Summary:")
    print(f"  Training Split Shape:   {train.shape}")
    print(f"  Validation Split Shape: {val.shape}")
    print(f"  Test Split Shape:       {test.shape}")
    print(f"  Test Labels Shape:      {test_labels.shape} (Attacks: {test_labels.sum()} = {test_labels.mean()*100:.2f}%)")
    print(f"  Sensor Channels:        {len(loader.sensor_names)}")
    print(f"  Target Directory:       {args.processed_dir}")


if __name__ == "__main__":
    main()

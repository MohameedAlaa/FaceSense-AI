"""
FaceSense AI - Training Entrypoint Script
Run full model training via CLI driven by configs/config.yaml.
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.trainer import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train FaceSense AI Baseline Emotion Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs to run (useful for testing)",
    )
    args = parser.parse_args()

    print(f"Starting FaceSense AI Training Pipeline with config: {args.config}")
    trainer, results = run_training_pipeline(
        config_path=args.config,
        max_epochs=args.epochs,
    )
    print("\nTraining workflow successfully completed.")


if __name__ == "__main__":
    main()

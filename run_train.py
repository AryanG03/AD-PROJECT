"""
run_train.py
────────────
Entry-point: train baseline GRU and/or Neuro-CX model.

Usage:
  py run_train.py                        # trains both models
  py run_train.py --model baseline       # baseline only
  py run_train.py --model neurocx        # Neuro-CX only
  py run_train.py --model neurocx --epochs 5   # quick test run
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train Neuro-CX models")
    parser.add_argument(
        "--model", choices=["baseline", "neurocx", "both"], default="both",
        help="Which model to train (default: both)"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs from config")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stopping patience (default from config: 7)")
    args = parser.parse_args()

    models_to_train = (
        ["baseline", "neurocx"] if args.model == "both" else [args.model]
    )

    for m in models_to_train:
        print(f"\n{'#'*60}")
        print(f"# Training: {m.upper()}")
        print(f"{'#'*60}")
        ckpt = train(
            model_name=m,
            config_path=args.config,
            epochs_override=args.epochs,
            patience_override=args.patience,
        )
        print(f"Saved: {ckpt}")



if __name__ == "__main__":
    main()

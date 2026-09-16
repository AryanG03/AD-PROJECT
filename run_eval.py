"""
run_eval.py
───────────
Entry-point: evaluate both models and produce comparison.

Usage:
  py run_eval.py                     # evaluate both on test set
  py run_eval.py --split val         # use validation set instead
  py run_eval.py --model baseline    # only evaluate baseline
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluation.harness import evaluate_checkpoint, print_metrics
from evaluation.compare import run_comparison
import yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate Neuro-CX models")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--model", choices=["baseline", "neurocx", "both"], default="both")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to specific checkpoint (for single-model eval)")
    args = parser.parse_args()

    if args.model == "both":
        run_comparison(config_path=args.config, split=args.split)
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ckpt_dir = cfg["paths"]["checkpoints"]
        if args.checkpoint:
            ckpt = args.checkpoint
        elif args.model == "baseline":
            ckpt = os.path.join(ckpt_dir, "gru_baseline_best.pt")
        else:
            ckpt = os.path.join(ckpt_dir, "neuro_cx_best.pt")

        metrics = evaluate_checkpoint(ckpt, split=args.split, config_path=args.config)
        print_metrics(metrics)


if __name__ == "__main__":
    main()

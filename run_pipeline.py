"""
run_pipeline.py
───────────────
Entry-point: preprocess raw data → padded sequences → .parquet files.

Usage:
  py run_pipeline.py
  py run_pipeline.py --config config.yaml
  py run_pipeline.py --synthetic          # use purely synthetic data
"""

import sys
import os

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure src/ is on the path when running from neuro-cx/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline.preprocessor import run_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuro-CX preprocessing pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use purely synthetic data (ignore real CSV)")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, use_synthetic_only=args.synthetic)

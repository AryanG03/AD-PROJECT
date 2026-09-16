"""
compare.py
──────────
Runs both baseline and Neuro-CX checkpoints, produces a side-by-side
comparison table and saves a bar chart to logs/comparison.png.
"""

import os
import sys
import json
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.harness import evaluate_checkpoint, print_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_comparison(
    config_path: str = "config.yaml",
    split: str = "test",
    baseline_ckpt: str = None,
    neurocx_ckpt: str = None,
) -> dict:
    """
    Compare baseline vs Neuro-CX on the same split.

    Returns
    -------
    dict: {"baseline": metrics_dict, "neurocx": metrics_dict}
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = cfg["paths"]["checkpoints"]
    logs_dir = cfg["paths"]["logs"]
    os.makedirs(logs_dir, exist_ok=True)

    if baseline_ckpt is None:
        baseline_ckpt = os.path.join(ckpt_dir, "gru_baseline_best.pt")
    if neurocx_ckpt is None:
        neurocx_ckpt  = os.path.join(ckpt_dir, "neuro_cx_best.pt")

    results = {}

    for name, ckpt in [("baseline", baseline_ckpt), ("neurocx", neurocx_ckpt)]:
        if not os.path.exists(ckpt):
            print(f"[WARN] Checkpoint not found: {ckpt} — skipping {name}")
            continue
        m = evaluate_checkpoint(ckpt, split=split, config_path=config_path)
        print_metrics(m)
        results[name] = m

    if len(results) < 2:
        print("Need both checkpoints to produce comparison. Skipping chart.")
        return results

    # ── Side-by-side table ────────────────────────────────────────────────────
    metric_keys = [k for k in results["baseline"].keys()
                   if k not in ("model", "split", "n_eval")]

    print(f"\n{'═'*60}")
    print(f"{'Metric':<20} {'Baseline':>12} {'Neuro-CX':>12} {'Δ':>10}")
    print(f"{'─'*60}")
    for mk in metric_keys:
        b = results["baseline"][mk]
        n = results["neurocx"][mk]
        delta = n - b
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"{mk:<20} {b:>12.4f} {n:>12.4f} {arrow}{abs(delta):>8.4f}")
    print(f"{'═'*60}\n")

    # Save comparison JSON
    comparison = {
        "split":    split,
        "baseline": {k: v for k, v in results["baseline"].items()},
        "neurocx":  {k: v for k, v in results["neurocx"].items()},
    }
    json_path = os.path.join(logs_dir, "comparison.json")
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison JSON saved to {json_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    _save_comparison_chart(results, metric_keys, logs_dir)

    return results


def _save_comparison_chart(results: dict, metric_keys: list, logs_dir: str):
    """Save a grouped bar chart comparing all metrics."""
    baseline_vals = [results["baseline"].get(k, 0.0) for k in metric_keys]
    neurocx_vals  = [results["neurocx"].get(k, 0.0)  for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(metric_keys) * 1.2), 5))
    bars1 = ax.bar(x - width/2, baseline_vals, width,
                   label="GRU Baseline",  color="#4C72B0", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, neurocx_vals,  width,
                   label="Neuro-CX",      color="#DD8452", alpha=0.85, edgecolor="white")

    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Score",  fontsize=11)
    ax.set_title("Neuro-CX vs GRU Baseline — Evaluation Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(baseline_vals), max(neurocx_vals)) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#4C72B0")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#DD8452")

    plt.tight_layout()
    chart_path = os.path.join(logs_dir, "comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to {chart_path}")

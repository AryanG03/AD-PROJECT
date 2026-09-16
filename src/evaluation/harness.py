"""
harness.py
──────────
Evaluation harness: loads a checkpoint, runs it on val/test, returns metrics.
"""

import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.dataset import get_dataloader
from evaluation.metrics import compute_all_metrics
from model.neuro_cx_model import NeuroCXModel
from model.trainer import load_checkpoint


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    split: str = "test",
    config_path: str = "config.yaml",
    k_values: list | None = None,
) -> dict:
    """
    Load a checkpoint and evaluate it on val or test split.

    Parameters
    ----------
    checkpoint_path : str
    split : "val" or "test"
    config_path : str
    k_values : list of int, optional

    Returns
    -------
    dict: metric_name -> float value
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if k_values is None:
        k_values = cfg["evaluation"]["k_values"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ckpt_data = load_checkpoint(checkpoint_path, device)
    model.eval()
    model_name = model.MODEL_NAME

    data_path = cfg["data"]["test_file"] if split == "test" else cfg["data"]["val_file"]
    loader = get_dataloader(
        data_path,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    all_scores  = []
    all_targets = []
    is_neuro_cx = isinstance(model, NeuroCXModel)

    for batch in tqdm(loader, desc=f"Evaluating {model_name} on {split}"):
        item_seq   = batch["item_seq"].to(device)
        action_seq = batch["action_seq"].to(device)
        dwell_seq  = batch["dwell_seq"].to(device)
        weight_seq = batch["weight_seq"].to(device)
        targets    = batch["target_item"].to(device)

        if is_neuro_cx:
            logits, _ = model(item_seq, action_seq, dwell_seq, weight_seq)
        else:
            logits, _ = model(item_seq, action_seq, dwell_seq)

        all_scores.append(logits.cpu())
        all_targets.append(targets.cpu())

    scores  = torch.cat(all_scores,  dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = compute_all_metrics(scores, targets, k_values=k_values)
    metrics["model"]  = model_name
    metrics["split"]  = split
    metrics["n_eval"] = len(targets)

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print a metrics dict."""
    print(f"\n{'─'*50}")
    print(f"  Model : {metrics.get('model', '?')}")
    print(f"  Split : {metrics.get('split', '?')}  (n={metrics.get('n_eval', '?')})")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        if k not in ("model", "split", "n_eval"):
            print(f"  {k:<18} {v:.4f}")
    print(f"{'─'*50}\n")

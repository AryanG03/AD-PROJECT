"""
metrics.py
──────────
Ranking metrics for next-item recommendation evaluation.

Implements:
  - NDCG@k  (Normalised Discounted Cumulative Gain)
  - HitRate@k  (Recall@k)
  - Next-Item Accuracy  (Hit@1)

All functions operate on batched logits / ranked lists for efficiency.
"""

import numpy as np
import torch


def ndcg_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute mean NDCG@k over a batch.

    Parameters
    ----------
    scores  : (B, n_items) — raw logits or probabilities (higher = better)
    targets : (B,)         — ground-truth item indices
    k       : int

    Returns
    -------
    float : mean NDCG@k across the batch
    """
    B = scores.size(0)
    # Get top-k item indices for each sample
    top_k = torch.argsort(scores, dim=-1, descending=True)[:, :k]  # (B, k)

    ndcg_vals = []
    for i in range(B):
        target = targets[i].item()
        ranked = top_k[i].tolist()
        if target in ranked:
            rank = ranked.index(target) + 1      # 1-indexed
            dcg  = 1.0 / np.log2(rank + 1)
            idcg = 1.0 / np.log2(2)              # ideal DCG (rank 1)
            ndcg_vals.append(dcg / idcg)
        else:
            ndcg_vals.append(0.0)

    return float(np.mean(ndcg_vals))


def hit_rate_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute mean Hit Rate@k (Recall@k for single relevant item) over a batch.

    Returns
    -------
    float : fraction of samples where target is in the top-k predictions
    """
    top_k   = torch.argsort(scores, dim=-1, descending=True)[:, :k]  # (B, k)
    targets_ = targets.unsqueeze(1).expand_as(top_k)                  # (B, k)
    hits    = (top_k == targets_).any(dim=-1).float()                 # (B,)
    return float(hits.mean().item())


def accuracy_at_1(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Next-item prediction accuracy (Hit@1).
    """
    preds = scores.argmax(dim=-1)         # (B,)
    return float((preds == targets).float().mean().item())


def compute_all_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k_values: list[int] | None = None,
) -> dict:
    """
    Compute all metrics at multiple k values.

    Returns
    -------
    dict with keys like "NDCG@5", "HitRate@5", ..., "Accuracy@1"
    """
    if k_values is None:
        k_values = [5, 10, 20]
    results = {"Accuracy@1": accuracy_at_1(scores, targets)}
    for k in k_values:
        results[f"NDCG@{k}"]    = ndcg_at_k(scores, targets, k)
        results[f"HitRate@{k}"] = hit_rate_at_k(scores, targets, k)
    return results

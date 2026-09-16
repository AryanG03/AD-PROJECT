"""
inference.py
────────────
POST /api/inference  —  run model forward pass and return recommendations.
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from api.model_registry import get_device, get_model
from model.neuro_cx_model import NeuroCXModel

router = APIRouter(prefix="/api/inference", tags=["inference"])

ITEM_CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books", "Sports",
    "Beauty", "Toys", "Food & Beverages", "Automotive", "Gardening"
]
ITEMS_PER_CAT = 20

_INTERACTIONS_DF: pd.DataFrame | None = None
_ENCODERS: dict | None = None


def _item_name(item_id: int) -> str:
    cat_idx  = item_id // ITEMS_PER_CAT
    item_num = item_id % ITEMS_PER_CAT + 1
    cat = ITEM_CATEGORIES[cat_idx] if cat_idx < len(ITEM_CATEGORIES) else "Misc"
    return f"{cat} #{item_num:02d}"


def _load_data():
    global _INTERACTIONS_DF, _ENCODERS
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _INTERACTIONS_DF is None:
        _INTERACTIONS_DF = pd.read_parquet(os.path.join(root, "data", "interactions_sample.parquet"))
    if _ENCODERS is None:
        with open(os.path.join(root, "data", "encoders.json")) as f:
            _ENCODERS = json.load(f)


class InferenceRequest(BaseModel):
    customer_id: str
    model_name: str = "neuro_cx"   # "neuro_cx" | "baseline"
    step: int = 0                  # use events 0..step (inclusive)
    top_k: int = 10


def _build_tensors(customer_df: pd.DataFrame, step: int, max_seq_len: int, device: torch.device):
    """Build padded input tensors for events 0..step."""
    enc        = _ENCODERS
    aw         = enc.get("action_weights", {})
    ac         = enc.get("action_classes", [])
    item_map   = {int(k): v for k, v in enc.get("item_map", {}).items()}
    dwell_max  = enc.get("dwell_max", 600.0)

    rows = customer_df.iloc[:step + 1]

    def enc_action(a):
        a = str(a).strip()
        return ac.index(a) if a in ac else 0

    actions  = [enc_action(r["action_type"])                                  for _, r in rows.iterrows()]
    items    = [item_map.get(int(r["item_id"]), 0)                            for _, r in rows.iterrows()]
    dwells   = [min(float(r["dwell_time"]) / dwell_max, 1.0)                  for _, r in rows.iterrows()]
    weights  = [float(aw.get(str(r["action_type"]).strip(), 1.0))             for _, r in rows.iterrows()]

    actual_len = len(actions)
    if actual_len > max_seq_len:
        actions, items, dwells, weights = (x[-max_seq_len:] for x in (actions, items, dwells, weights))
        actual_len = max_seq_len

    pad = max_seq_len - actual_len
    actions  = [0]   * pad + actions
    items    = [0]   * pad + items
    dwells   = [0.0] * pad + dwells
    weights  = [0.0] * pad + weights

    return (
        torch.tensor([actions],  dtype=torch.long,    device=device),
        torch.tensor([items],    dtype=torch.long,    device=device),
        torch.tensor([dwells],   dtype=torch.float32, device=device),
        torch.tensor([weights],  dtype=torch.float32, device=device),
        actual_len,
    )


@router.post("")
@torch.no_grad()
def run_inference(req: InferenceRequest):
    """
    Run model inference for a customer at a given sequence step.
    Returns ranked recommendations, hidden state norm, and signal metadata.
    """
    _load_data()

    if req.model_name not in ("neuro_cx", "baseline"):
        raise HTTPException(status_code=400, detail="model_name must be 'neuro_cx' or 'baseline'")

    df = _INTERACTIONS_DF[
        _INTERACTIONS_DF["customer_id"] == req.customer_id
    ].sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{req.customer_id}' not found")

    n_events = len(df)
    step = min(req.step, n_events - 1)

    model, ckpt = get_model(req.model_name)
    device      = get_device()
    enc         = _ENCODERS
    max_seq_len = enc.get("max_seq_len", 50)

    action_seq, item_seq, dwell_seq, weight_seq, _actual_len = _build_tensors(df, step, max_seq_len, device)

    is_ncx = isinstance(model, NeuroCXModel)

    if is_ncx:
        try:
            logits, hidden, _all_hiddens = model(
                item_seq, action_seq, dwell_seq, weight_seq, return_all_hidden=True
            )
        except TypeError:
            logits, hidden = model(item_seq, action_seq, dwell_seq, weight_seq)
    else:
        logits, hidden = model(item_seq, action_seq, dwell_seq)

    probs    = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    top_k    = min(req.top_k, len(probs))
    top_idx  = np.argsort(probs)[::-1][:top_k].tolist()
    top_sc   = probs[top_idx].tolist()

    recommendations = [
        {"rank": i + 1, "item_id": iid, "name": _item_name(iid), "score": round(float(sc), 6), "pct": round(float(sc) * 100, 3)}
        for i, (iid, sc) in enumerate(zip(top_idx, top_sc))
    ]

    # Hidden state norm (first 64 dims for viz)
    last_h   = hidden[-1].squeeze(0).cpu().numpy()
    h_norm   = float(np.linalg.norm(last_h))
    h_sample = last_h[:64].tolist()

    # Signal metadata for current step
    current_event = df.iloc[step]
    action_str    = str(current_event["action_type"]).strip()
    aw            = enc.get("action_weights", {})
    w_val         = float(aw.get(action_str, 1.0))

    # Decay factor (Neuro-CX only)
    decay_val = None
    if is_ncx:
        decay_hl  = ckpt.get("model_config", {}).get("decay_half_life", 5.0)
        lam       = math.log(2) / decay_hl
        decay_val = round(math.exp(-lam / max(w_val, 0.1)), 4)

    return {
        "customer_id":    req.customer_id,
        "model_name":     req.model_name,
        "step":           step,
        "n_events":       n_events,
        "current_action": action_str,
        "signal_weight":  w_val,
        "decay_factor":   decay_val,
        "hidden_norm":    round(h_norm, 4),
        "hidden_sample":  h_sample,
        "recommendations": recommendations,
    }

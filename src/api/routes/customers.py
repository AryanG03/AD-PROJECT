"""
customers.py
────────────
/api/customers  —  list customer IDs and fetch individual sequences.
"""

import json
import os

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/customers", tags=["customers"])

_INTERACTIONS_DF: pd.DataFrame | None = None
_ENCODERS: dict | None = None


def _load_data():
    global _INTERACTIONS_DF, _ENCODERS

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    inter_path = os.path.join(root, "data", "interactions_sample.parquet")
    enc_path   = os.path.join(root, "data", "encoders.json")

    if _INTERACTIONS_DF is None:
        if not os.path.exists(inter_path):
            raise FileNotFoundError(f"Interactions file not found: {inter_path}")
        _INTERACTIONS_DF = pd.read_parquet(inter_path)

    if _ENCODERS is None:
        if not os.path.exists(enc_path):
            raise FileNotFoundError(f"Encoders file not found: {enc_path}")
        with open(enc_path) as f:
            _ENCODERS = json.load(f)


ITEM_CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books", "Sports",
    "Beauty", "Toys", "Food & Beverages", "Automotive", "Gardening"
]
ITEMS_PER_CAT = 20


def _item_id_to_name(item_id: int) -> str:
    cat_idx  = item_id // ITEMS_PER_CAT
    item_num = item_id % ITEMS_PER_CAT + 1
    cat = ITEM_CATEGORIES[cat_idx] if cat_idx < len(ITEM_CATEGORIES) else "Misc"
    return f"{cat} #{item_num:02d}"


ACTION_ICONS = {
    "purchase":    "🛒",
    "view":        "👁",
    "add_to_cart": "🛍",
    "review":      "⭐",
    "bounce":      "↩",
}


@router.get("")
def list_customers(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """Return a paginated list of customer IDs."""
    _load_data()
    all_ids = sorted(_INTERACTIONS_DF["customer_id"].unique().tolist())
    total   = len(all_ids)
    start   = (page - 1) * page_size
    end     = start + page_size
    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "customers": all_ids[start:end],
    }


@router.get("/{customer_id}")
def get_customer(customer_id: str):
    """Return a customer's full interaction history as a list of events."""
    _load_data()

    df = _INTERACTIONS_DF[
        _INTERACTIONS_DF["customer_id"] == customer_id
    ].sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")

    events = []
    for _, row in df.iterrows():
        action = str(row["action_type"]).strip()
        events.append({
            "step":        int(row.get("sequence_position", _)),
            "action":      action,
            "action_icon": ACTION_ICONS.get(action, "•"),
            "item_id":     int(row["item_id"]),
            "item_name":   _item_id_to_name(int(row["item_id"])),
            "dwell_time":  round(float(row["dwell_time"]), 1),
            "timestamp":   str(row["timestamp"]),
            "signal_weight": float(
                _ENCODERS.get("action_weights", {}).get(action, 1.0)
            ) if _ENCODERS else 1.0,
        })

    return {
        "customer_id": customer_id,
        "n_events":    len(events),
        "events":      events,
    }

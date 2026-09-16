"""
synthetic_generator.py
──────────────────────
Generates a realistic synthetic customer interaction dataset that mimics
e-commerce clickstream sequences. Used as a standalone fallback when no real
dataset is available, and as a supplement to the real CSV in testing.

Schema produced:
  customer_id | timestamp | action_type | item_id | dwell_time | sequence_position
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


# ── Constants ─────────────────────────────────────────────────────────────────

ACTION_TYPES = ["view", "add_to_cart", "purchase", "review", "bounce"]

# Realistic action-transition probabilities (Markov-style)
# P(next_action | current_action)
TRANSITION_PROBS = {
    "view":        {"view": 0.40, "add_to_cart": 0.25, "bounce": 0.25, "purchase": 0.05, "review": 0.05},
    "add_to_cart": {"view": 0.20, "add_to_cart": 0.10, "bounce": 0.10, "purchase": 0.50, "review": 0.10},
    "purchase":    {"view": 0.40, "add_to_cart": 0.15, "bounce": 0.10, "purchase": 0.10, "review": 0.25},
    "review":      {"view": 0.50, "add_to_cart": 0.15, "bounce": 0.20, "purchase": 0.10, "review": 0.05},
    "bounce":      {"view": 0.60, "add_to_cart": 0.10, "bounce": 0.20, "purchase": 0.05, "review": 0.05},
}

# Mean dwell time in seconds per action type
DWELL_MEANS = {
    "view": 45, "add_to_cart": 15, "purchase": 120, "review": 180, "bounce": 5
}

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books", "Sports",
    "Beauty", "Toys", "Food & Beverages", "Automotive", "Gardening"
]

ITEMS_PER_CATEGORY = 20  # 10 categories × 20 items = 200 distinct items


def _category_item_ids() -> dict:
    """Return mapping: category → list of item_ids."""
    return {
        cat: list(range(i * ITEMS_PER_CATEGORY, (i + 1) * ITEMS_PER_CATEGORY))
        for i, cat in enumerate(CATEGORIES)
    }


def _sample_action(current_action: str, rng: np.random.Generator) -> str:
    probs = TRANSITION_PROBS[current_action]
    actions = list(probs.keys())
    weights = [probs[a] for a in actions]
    return rng.choice(actions, p=weights)


def _sample_dwell(action: str, rng: np.random.Generator) -> float:
    mean = DWELL_MEANS[action]
    return float(np.clip(rng.exponential(mean), 1.0, 600.0))


def generate_synthetic_dataset(
    n_customers: int = 500,
    min_seq_len: int = 5,
    max_seq_len: int = 15,
    seed: int = 42,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Generate a synthetic interaction dataset.

    Parameters
    ----------
    n_customers : int
        Number of unique customers to simulate.
    min_seq_len : int
        Minimum interactions per customer.
    max_seq_len : int
        Maximum interactions per customer.
    seed : int
        Random seed for reproducibility.
    start_date : str
        ISO date string; first possible interaction timestamp.

    Returns
    -------
    pd.DataFrame with columns:
        customer_id, timestamp, action_type, item_id, dwell_time, sequence_position
    """
    rng = np.random.default_rng(seed)
    cat_items = _category_item_ids()
    base_dt = datetime.fromisoformat(start_date)

    rows = []
    for cust_idx in range(n_customers):
        customer_id = f"SYNTH_{cust_idx:05d}"
        seq_len = int(rng.integers(min_seq_len, max_seq_len + 1))

        # Give each synthetic customer a preferred category (neuroplasticity analogy:
        # strong repeated signals reinforce that category's representation)
        pref_category = rng.choice(CATEGORIES)
        pref_items = cat_items[pref_category]
        other_items = [
            item for cat, items in cat_items.items()
            if cat != pref_category for item in items
        ]

        # Random starting timestamp within the past year
        offset_days = int(rng.integers(0, 365))
        current_dt = base_dt + timedelta(days=offset_days)

        action = "view"  # every session starts with a view
        for pos in range(seq_len):
            # 70% chance of interacting with preferred category
            if rng.random() < 0.70:
                item_id = int(rng.choice(pref_items))
            else:
                item_id = int(rng.choice(other_items))

            dwell = _sample_dwell(action, rng)
            rows.append({
                "customer_id": customer_id,
                "timestamp": current_dt,
                "action_type": action,
                "item_id": item_id,
                "dwell_time": round(dwell, 2),
                "sequence_position": pos,
            })

            # Advance time by dwell + small gap
            gap_seconds = float(rng.uniform(10, 120))
            current_dt += timedelta(seconds=dwell + gap_seconds)

            if pos < seq_len - 1:
                action = _sample_action(action, rng)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(n_customers=200, seed=42)
    print(f"Generated {len(df):,} rows for {df['customer_id'].nunique()} customers")
    print(df.head(10).to_string(index=False))
    print("\nAction distribution:")
    print(df["action_type"].value_counts())

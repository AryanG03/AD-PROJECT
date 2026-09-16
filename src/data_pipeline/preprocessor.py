"""
preprocessor.py
───────────────
Transforms the raw e-commerce CSV (or synthetic data) into padded integer
sequences ready for the GRU model.

Pipeline steps:
  1. Load real CSV  →  synthesise per-customer interaction sequences
  2. Clean / deduplicate
  3. Encode categoricals (LabelEncoder → integer IDs)
  4. Normalise numeric features (dwell_time, sequence_position)
  5. Pad / truncate to max_seq_len
  6. 70/15/15 train/val/test split (stratified by customer if possible)
  7. Save as .parquet

Output schema per row:
  customer_id | action_seq (list[int]) | item_seq (list[int])
  | dwell_seq (list[float]) | weight_seq (list[float])
  | target_item (int) | seq_len (int)
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import yaml

# Make src importable when run from neuro-cx/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.synthetic_generator import (
    generate_synthetic_dataset,
    ACTION_TYPES,
    CATEGORIES,
    ITEMS_PER_CATEGORY,
    TRANSITION_PROBS,
    DWELL_MEANS,
)


# ── Action weight mapping (reinforcement signal strength) ────────────────────

DEFAULT_ACTION_WEIGHTS = {
    "purchase":    3.0,
    "review":      1.5,
    "add_to_cart": 1.5,
    "view":        1.0,
    "bounce":      0.3,
}


# ── Real CSV → Sequence synthesis ────────────────────────────────────────────

def _parse_amount(val: str) -> float:
    """Parse '$1,234.56 ' → 1234.56."""
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _category_to_item_range(category: str, all_categories: list) -> list:
    """Return item IDs for a given category (deterministic mapping)."""
    idx = all_categories.index(category) if category in all_categories else 0
    return list(range(idx * ITEMS_PER_CATEGORY, (idx + 1) * ITEMS_PER_CATEGORY))


def _synthesize_sequences_from_csv(df_raw: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Convert cross-sectional customer data into per-customer interaction sequences.

    Each real CSV row represents one customer profile + one purchase event.
    We back-fill earlier synthetic events to create a plausible journey:
      - Number of prior interactions  ∝  Frequency_of_Purchase
      - Dominant category             =  Purchase_Category from real data
      - Purchase intent maps to action mix (impulsive → fewer views before purchase)
      - Time anchor                   =  Time_of_Purchase from real data

    The resulting sequences retain real customer attribute distributions while
    providing the sequential structure needed by the GRU.
    """
    rng = np.random.default_rng(seed)
    all_categories = CATEGORIES

    # Map real categories to our canonical list
    cat_mapping = {}
    for rc in df_raw["Purchase_Category"].str.strip().unique():
        # fuzzy match
        best = None
        for cc in all_categories:
            if cc.lower() in rc.lower() or rc.lower() in cc.lower():
                best = cc
                break
        cat_mapping[rc] = best if best else rng.choice(all_categories)

    rows = []
    for _, record in df_raw.iterrows():
        cid = str(record["Customer_ID"]).strip()

        # Parse anchor timestamp
        try:
            anchor_dt = pd.to_datetime(record["Time_of_Purchase"], dayfirst=False)
        except Exception:
            anchor_dt = pd.Timestamp("2024-06-01")

        pref_cat = cat_mapping.get(str(record["Purchase_Category"]).strip(), rng.choice(all_categories))
        pref_items = _category_to_item_range(pref_cat, all_categories)
        other_items = [
            item for cat in all_categories
            if cat != pref_cat
            for item in _category_to_item_range(cat, all_categories)
        ]

        # Determine sequence length from Frequency_of_Purchase (1–12 → 3–14 events)
        freq = int(record.get("Frequency_of_Purchase", 4))
        seq_len = max(3, min(14, freq + 2))

        # Intent-driven action bias
        intent = str(record.get("Purchase_Intent", "Need-based")).lower()
        if "impulsive" in intent:
            # Short funnel: view → purchase fast
            action_seq = (["view"] * max(1, seq_len // 3)
                          + ["add_to_cart"] * 1
                          + ["purchase"])
        elif "wants" in intent:
            action_seq = (["view"] * max(2, seq_len // 2)
                          + ["add_to_cart"] * 1
                          + ["view"] * 1
                          + ["purchase"])
        else:  # need-based
            action_seq = (["view"] * max(3, seq_len // 2)
                          + ["add_to_cart"] * 1
                          + ["view"] * 1
                          + ["purchase"]
                          + ["review"] * min(2, seq_len - 5))

        # Trim/extend to seq_len
        while len(action_seq) < seq_len:
            action_seq.insert(0, "view")
        action_seq = action_seq[-seq_len:]

        # Build the actual rows, working backwards from the anchor timestamp
        dwell_time_research = float(record.get("Time_Spent_on_Product_Research(hours)", 1.0)) * 3600
        current_dt = anchor_dt

        event_rows = []
        for pos, action in enumerate(reversed(action_seq)):
            if action == "purchase":
                dwell = float(np.clip(rng.normal(120, 30), 30, 300))
            elif action == "view":
                dwell = float(np.clip(rng.exponential(dwell_time_research / max(1, seq_len)), 10, 600))
            else:
                dwell = float(np.clip(rng.exponential(DWELL_MEANS[action]), 5, 600))

            # Item selection: prefer category items, occasionally explore
            if rng.random() < 0.75:
                item_id = int(rng.choice(pref_items))
            else:
                item_id = int(rng.choice(other_items))

            # Rating signal: high-rated products generate reviews
            if action == "review":
                product_rating = int(record.get("Product_Rating", 3))
                item_id = int(rng.choice(pref_items))  # always preferred for reviews

            event_rows.append({
                "customer_id": cid,
                "timestamp": current_dt,
                "action_type": action,
                "item_id": item_id,
                "dwell_time": round(dwell, 2),
                "sequence_position": 0,  # will be recomputed after sort
            })

            # Step backward in time
            gap = float(rng.uniform(300, 3600 * 24))  # 5 min – 1 day gap
            current_dt = current_dt - timedelta(seconds=dwell + gap)

        # Reverse so events are chronological
        event_rows = list(reversed(event_rows))
        for i, row in enumerate(event_rows):
            row["sequence_position"] = i

        rows.extend(event_rows)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    return df


# ── Encoding + Feature engineering ──────────────────────────────────────────

def encode_interactions(df: pd.DataFrame, action_weights: dict) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical features and add signal weight column.
    Returns (encoded_df, encoders_dict).
    """
    encoders = {}

    # Encode action_type
    le_action = LabelEncoder()
    le_action.fit(list(action_weights.keys()))
    df["action_id"] = le_action.transform(df["action_type"].str.strip())
    encoders["action"] = le_action

    # Encode item_id (already int, but ensure contiguous from 0)
    all_items = sorted(df["item_id"].unique())
    item_map = {v: i for i, v in enumerate(all_items)}
    df["item_id_enc"] = df["item_id"].map(item_map)
    encoders["item_map"] = item_map
    encoders["n_items"] = len(all_items)
    encoders["n_actions"] = len(le_action.classes_)

    # Normalise dwell_time to [0,1]
    dwell_max = df["dwell_time"].quantile(0.99)
    df["dwell_norm"] = (df["dwell_time"] / dwell_max).clip(0, 1)
    encoders["dwell_max"] = float(dwell_max)

    # Signal reinforcement weight
    df["signal_weight"] = df["action_type"].map(action_weights).fillna(1.0)

    return df, encoders


# ── Sequence building + Padding ──────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, max_seq_len: int = 50) -> pd.DataFrame:
    """
    Group by customer, sort by timestamp, build fixed-length padded sequences.
    Each row in the output represents one training example (all interactions
    up to the last one, with the last item as the target).

    Output columns:
      customer_id | action_seq | item_seq | dwell_seq | weight_seq
      | target_item | seq_len
    """
    records = []

    for cid, grp in df.groupby("customer_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)

        actions = grp["action_id"].tolist()
        items   = grp["item_id_enc"].tolist()
        dwells  = grp["dwell_norm"].tolist()
        weights = grp["signal_weight"].tolist()

        # We need at least 2 events (1 input + 1 target)
        if len(actions) < 2:
            continue

        # Use all-but-last as input sequence, last item as target
        seq_actions = actions[:-1]
        seq_items   = items[:-1]
        seq_dwells  = dwells[:-1]
        seq_weights = weights[:-1]
        target_item = items[-1]

        actual_len = len(seq_actions)

        # Truncate to max_seq_len
        if actual_len > max_seq_len:
            seq_actions = seq_actions[-max_seq_len:]
            seq_items   = seq_items[-max_seq_len:]
            seq_dwells  = seq_dwells[-max_seq_len:]
            seq_weights = seq_weights[-max_seq_len:]
            actual_len  = max_seq_len

        # Pad to max_seq_len with zeros (left-pad so recent events align right)
        pad_len = max_seq_len - actual_len
        seq_actions = [0] * pad_len + seq_actions
        seq_items   = [0] * pad_len + seq_items
        seq_dwells  = [0.0] * pad_len + seq_dwells
        seq_weights = [0.0] * pad_len + seq_weights

        records.append({
            "customer_id":  cid,
            "action_seq":   seq_actions,
            "item_seq":     seq_items,
            "dwell_seq":    seq_dwells,
            "weight_seq":   seq_weights,
            "target_item":  target_item,
            "seq_len":      actual_len,
        })

    return pd.DataFrame(records)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(config_path: str = "config.yaml", use_synthetic_only: bool = False):
    """
    Full preprocessing pipeline. Saves train/val/test .parquet files and
    an encoders.json file to the processed_dir.

    Parameters
    ----------
    config_path : str
        Path to config.yaml (relative to cwd or absolute).
    use_synthetic_only : bool
        If True, skip the real CSV and use purely synthetic data.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    feat_cfg  = cfg["features"]
    train_cfg = cfg["training"]

    action_weights = cfg["features"]["action_weights"]
    max_seq_len    = data_cfg["max_seq_len"]
    seed           = train_cfg["seed"]

    os.makedirs(data_cfg["processed_dir"], exist_ok=True)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("=" * 60)
    print("Neuro-CX Data Pipeline")
    print("=" * 60)

    if not use_synthetic_only and os.path.exists(data_cfg["raw_csv"]):
        print(f"[1/5] Loading real CSV: {data_cfg['raw_csv']}")
        df_raw = pd.read_csv(data_cfg["raw_csv"])
        print(f"      {len(df_raw):,} customer profiles loaded.")
        print("[2/5] Synthesising interaction sequences from real attributes...")
        df_events = _synthesize_sequences_from_csv(df_raw, seed=seed)
        print(f"      {len(df_events):,} interaction events generated for "
              f"{df_events['customer_id'].nunique()} customers.")
    else:
        print("[1/5] Real CSV not found — using pure synthetic dataset.")
        df_events = generate_synthetic_dataset(n_customers=500, seed=seed)
        print(f"      {len(df_events):,} events for {df_events['customer_id'].nunique()} customers.")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    print("[2/5] Cleaning...")
    before = len(df_events)
    df_events = df_events.drop_duplicates(
        subset=["customer_id", "timestamp", "action_type", "item_id"]
    )
    df_events = df_events.dropna(subset=["customer_id", "action_type", "item_id"])
    df_events["dwell_time"] = pd.to_numeric(df_events["dwell_time"], errors="coerce").fillna(1.0)
    df_events["item_id"]    = pd.to_numeric(df_events["item_id"], errors="coerce").fillna(0).astype(int)
    print(f"      Removed {before - len(df_events)} duplicate/null rows -> {len(df_events):,} remain.")

    # ── Step 3: Encode ────────────────────────────────────────────────────────
    print("[3/5] Encoding features...")
    df_events, encoders = encode_interactions(df_events, action_weights)
    print(f"      {encoders['n_items']} unique items, {encoders['n_actions']} action types.")

    # ── Step 4: Build sequences ───────────────────────────────────────────────
    print(f"[4/5] Building sequences (max_len={max_seq_len})...")
    df_seqs = build_sequences(df_events, max_seq_len=max_seq_len)
    print(f"      {len(df_seqs):,} training examples built.")

    # ── Step 5: Split + save ──────────────────────────────────────────────────
    print("[5/5] Splitting and saving...")
    train_ratio = data_cfg["train_ratio"]
    val_ratio   = data_cfg["val_ratio"]

    # Shuffle deterministically
    df_seqs = df_seqs.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df_seqs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    df_train = df_seqs.iloc[:n_train]
    df_val   = df_seqs.iloc[n_train:n_train + n_val]
    df_test  = df_seqs.iloc[n_train + n_val:]

    df_train.to_parquet(data_cfg["train_file"], index=False)
    df_val.to_parquet(data_cfg["val_file"],     index=False)
    df_test.to_parquet(data_cfg["test_file"],   index=False)

    print(f"      Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
    print(f"      Saved to {data_cfg['processed_dir']}")

    # Save encoder metadata for use by model + dashboard
    enc_path = os.path.join(data_cfg["processed_dir"], "encoders.json")
    encoder_data = {
        "n_items":    encoders["n_items"],
        "n_actions":  encoders["n_actions"],
        "dwell_max":  encoders["dwell_max"],
        "item_map":   {str(k): v for k, v in encoders["item_map"].items()},
        "action_classes": list(encoders["action"].classes_),
        "action_weights": action_weights,
        "max_seq_len": max_seq_len,
    }
    with open(enc_path, "w") as f:
        json.dump(encoder_data, f, indent=2)
    print(f"      Encoder metadata saved to {enc_path}")

    # Save a sample of raw interactions for the dashboard
    raw_sample_path = os.path.join(data_cfg["processed_dir"], "interactions_sample.parquet")
    df_events.to_parquet(raw_sample_path, index=False)
    print(f"      Raw interactions saved to {raw_sample_path}")

    print("\n✓ Pipeline complete.\n")
    return df_train, df_val, df_test, encoders


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Neuro-CX data preprocessing pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data only")
    args = parser.parse_args()
    run_pipeline(config_path=args.config, use_synthetic_only=args.synthetic)

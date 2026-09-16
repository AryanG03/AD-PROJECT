"""
test_data_pipeline.py
─────────────────────
Neha's Days 16–18 Deliverable: Unit tests for the data pipeline.

Tests:
  - Schema validation: all required columns present + correct types
  - Padding: sequences are exactly max_seq_len elements long
  - DataLoader: correct tensor shapes and dtypes
  - Encoder metadata: all required keys present
  - clean_data: deduplication and type coercion work correctly
  - sequence_builder: builds correct target items and sequence lengths

Run with:
  py -m pytest tests/test_data_pipeline.py -v
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest
import torch

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA_DIR       = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_FILE     = os.path.join(DATA_DIR, "train.parquet")
VAL_FILE       = os.path.join(DATA_DIR, "val.parquet")
TEST_FILE      = os.path.join(DATA_DIR, "test.parquet")
SAMPLE_FILE    = os.path.join(DATA_DIR, "sample_data.parquet")
ENCODERS_FILE  = os.path.join(DATA_DIR, "encoders.json")
CLEANED_FILE   = os.path.join(DATA_DIR, "cleaned_events.parquet")

MAX_SEQ_LEN = 50
REQUIRED_SEQ_COLS = ["customer_id", "action_seq", "item_seq", "dwell_seq",
                     "weight_seq", "target_item", "seq_len"]
REQUIRED_ENC_KEYS = ["n_items", "n_actions", "dwell_max", "item_map",
                     "action_classes", "action_weights", "max_seq_len"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_df():
    assert os.path.exists(TRAIN_FILE), f"train.parquet not found. Run run_pipeline.py first."
    return pd.read_parquet(TRAIN_FILE)


@pytest.fixture(scope="module")
def encoders():
    assert os.path.exists(ENCODERS_FILE), f"encoders.json not found."
    with open(ENCODERS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_df():
    assert os.path.exists(SAMPLE_FILE), f"sample_data.parquet not found."
    return pd.read_parquet(SAMPLE_FILE)


@pytest.fixture(scope="module")
def cleaned_df():
    assert os.path.exists(CLEANED_FILE), f"cleaned_events.parquet not found."
    return pd.read_parquet(CLEANED_FILE)


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestSchema:
    """Validate that all parquet files conform to the frozen schema contract."""

    @pytest.mark.parametrize("filepath", [TRAIN_FILE, VAL_FILE, TEST_FILE, SAMPLE_FILE])
    def test_required_columns_present(self, filepath):
        """All required sequence columns must be present."""
        df = pd.read_parquet(filepath)
        for col in REQUIRED_SEQ_COLS:
            assert col in df.columns, f"Missing column '{col}' in {os.path.basename(filepath)}"

    def test_no_empty_splits(self, train_df):
        """Train/val/test splits must be non-empty."""
        assert len(train_df) > 0, "train.parquet is empty"
        for fp in [VAL_FILE, TEST_FILE]:
            df = pd.read_parquet(fp)
            assert len(df) > 0, f"{os.path.basename(fp)} is empty"

    def test_split_sizes_are_reasonable(self, train_df):
        """Train should be the largest split."""
        val_df  = pd.read_parquet(VAL_FILE)
        test_df = pd.read_parquet(TEST_FILE)
        assert len(train_df) > len(val_df), "Train should be larger than val"
        assert len(train_df) > len(test_df), "Train should be larger than test"

    def test_no_customer_id_overlap_between_splits(self):
        """No customer should appear in more than one split."""
        train_ids = set(pd.read_parquet(TRAIN_FILE)["customer_id"].astype(str))
        val_ids   = set(pd.read_parquet(VAL_FILE)["customer_id"].astype(str))
        test_ids  = set(pd.read_parquet(TEST_FILE)["customer_id"].astype(str))
        assert train_ids.isdisjoint(val_ids),  "Customer overlap between train and val"
        assert train_ids.isdisjoint(test_ids), "Customer overlap between train and test"
        assert val_ids.isdisjoint(test_ids),   "Customer overlap between val and test"


# ── Padding tests ─────────────────────────────────────────────────────────────

class TestPadding:
    """Validate padding and sequence length fields."""

    def _parse_list(self, val):
        """Parse list column (may be stored as Python list or string repr)."""
        import ast
        if isinstance(val, str):
            return ast.literal_eval(val)
        return list(val)

    def test_sequences_are_exactly_max_seq_len(self, train_df):
        """Every sequence list must have exactly max_seq_len elements."""
        for col in ["action_seq", "item_seq", "dwell_seq", "weight_seq"]:
            for val in train_df[col].head(20):
                seq = self._parse_list(val)
                assert len(seq) == MAX_SEQ_LEN, \
                    f"Column '{col}' has length {len(seq)}, expected {MAX_SEQ_LEN}"

    def test_seq_len_within_bounds(self, train_df):
        """seq_len must be between 1 and MAX_SEQ_LEN inclusive."""
        assert train_df["seq_len"].between(1, MAX_SEQ_LEN).all(), \
            "seq_len out of [1, MAX_SEQ_LEN] range"

    def test_dwell_seq_in_unit_range(self, train_df):
        """Dwell values must be normalised to [0, 1]."""
        for val in train_df["dwell_seq"].head(20):
            seq = self._parse_list(val)
            arr = np.array(seq, dtype=float)
            assert (arr >= 0.0).all() and (arr <= 1.0 + 1e-6).all(), \
                f"dwell_seq out of [0,1]: min={arr.min():.4f}, max={arr.max():.4f}"

    def test_weight_seq_non_negative(self, train_df):
        """Signal weights must be >= 0."""
        for val in train_df["weight_seq"].head(20):
            seq = self._parse_list(val)
            arr = np.array(seq, dtype=float)
            assert (arr >= 0.0).all(), f"weight_seq contains negative values"

    def test_item_seq_non_negative(self, train_df):
        """Item IDs (including padding=0) must be non-negative integers."""
        for val in train_df["item_seq"].head(20):
            seq = self._parse_list(val)
            arr = np.array(seq, dtype=int)
            assert (arr >= 0).all(), "item_seq contains negative values"


# ── Encoder tests ─────────────────────────────────────────────────────────────

class TestEncoders:
    """Validate encoders.json metadata."""

    def test_required_keys_present(self, encoders):
        for key in REQUIRED_ENC_KEYS:
            assert key in encoders, f"Missing key '{key}' in encoders.json"

    def test_n_items_positive(self, encoders):
        assert encoders["n_items"] > 0

    def test_n_actions_equals_5(self, encoders):
        assert encoders["n_actions"] == 5, \
            f"Expected 5 action types, got {encoders['n_actions']}"

    def test_action_classes_correct(self, encoders):
        expected = sorted(["purchase", "review", "add_to_cart", "view", "bounce"])
        actual   = sorted(encoders["action_classes"])
        assert actual == expected, f"Action classes mismatch: {actual}"

    def test_action_weights_all_present(self, encoders):
        for action in ["purchase", "review", "add_to_cart", "view", "bounce"]:
            assert action in encoders["action_weights"], \
                f"Missing weight for action '{action}'"

    def test_purchase_weight_highest(self, encoders):
        weights = encoders["action_weights"]
        assert weights["purchase"] == max(weights.values()), \
            "Purchase should have the highest signal weight"

    def test_max_seq_len_correct(self, encoders):
        assert encoders["max_seq_len"] == MAX_SEQ_LEN


# ── DataLoader tests ──────────────────────────────────────────────────────────

class TestDataLoader:
    """Validate the PyTorch DataLoader produces correct tensor shapes."""

    def test_batch_shapes(self):
        from data_pipeline.dataset import get_dataloader
        loader = get_dataloader(TRAIN_FILE, batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        assert batch["item_seq"].shape    == (8, MAX_SEQ_LEN), \
            f"item_seq shape wrong: {batch['item_seq'].shape}"
        assert batch["action_seq"].shape  == (8, MAX_SEQ_LEN)
        assert batch["dwell_seq"].shape   == (8, MAX_SEQ_LEN)
        assert batch["weight_seq"].shape  == (8, MAX_SEQ_LEN)
        assert batch["target_item"].shape == (8,)
        assert batch["seq_len"].shape     == (8,)

    def test_batch_dtypes(self):
        from data_pipeline.dataset import get_dataloader
        loader = get_dataloader(TRAIN_FILE, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        assert batch["item_seq"].dtype    == torch.long,    "item_seq should be int64"
        assert batch["action_seq"].dtype  == torch.long,    "action_seq should be int64"
        assert batch["dwell_seq"].dtype   == torch.float32, "dwell_seq should be float32"
        assert batch["weight_seq"].dtype  == torch.float32, "weight_seq should be float32"
        assert batch["target_item"].dtype == torch.long,    "target_item should be int64"

    def test_target_item_in_vocab(self):
        from data_pipeline.dataset import get_dataloader
        with open(ENCODERS_FILE) as f:
            enc = json.load(f)
        n_items = enc["n_items"]

        loader = get_dataloader(TRAIN_FILE, batch_size=32, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert (batch["target_item"] >= 0).all()
        assert (batch["target_item"] < n_items).all(), \
            "target_item out of vocab range"


# ── Clean data tests ──────────────────────────────────────────────────────────

class TestCleanData:
    """Validate the cleaned_events.parquet output from clean_data.py."""

    REQUIRED_COLS = ["customer_id", "timestamp", "action_type", "item_id",
                     "dwell_time", "sequence_position"]

    def test_required_columns(self, cleaned_df):
        for col in self.REQUIRED_COLS:
            assert col in cleaned_df.columns, f"Missing '{col}' in cleaned_events.parquet"

    def test_no_null_critical_fields(self, cleaned_df):
        for col in ["customer_id", "action_type", "item_id"]:
            assert cleaned_df[col].isna().sum() == 0, f"Nulls found in '{col}'"

    def test_dwell_time_in_range(self, cleaned_df):
        assert (cleaned_df["dwell_time"] >= 1.0).all(), "dwell_time below 1s"
        assert (cleaned_df["dwell_time"] <= 3600.0).all(), "dwell_time above 3600s"

    def test_action_types_valid(self, cleaned_df):
        valid = {"purchase", "review", "add_to_cart", "view", "bounce"}
        actual = set(cleaned_df["action_type"].unique())
        assert actual.issubset(valid), f"Invalid action types: {actual - valid}"

    def test_timestamps_are_datetime(self, cleaned_df):
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df["timestamp"]), \
            "timestamp column is not datetime"

    def test_no_duplicate_events(self, cleaned_df):
        dupes = cleaned_df.duplicated(
            subset=["customer_id", "timestamp", "action_type", "item_id"]
        ).sum()
        assert dupes == 0, f"{dupes} duplicate events found"


# ── Sample data tests ─────────────────────────────────────────────────────────

class TestSampleData:
    """Validate sample_data.parquet (teammates' mock-dev data)."""

    def test_sample_is_subset_of_train(self, sample_df, train_df):
        """Sample data must come from the training split."""
        sample_ids = set(sample_df["customer_id"].astype(str))
        train_ids  = set(train_df["customer_id"].astype(str))
        assert sample_ids.issubset(train_ids), \
            "sample_data contains customers not in train split"

    def test_sample_size(self, sample_df):
        assert 100 <= len(sample_df) <= 300, \
            f"Sample size {len(sample_df)} outside expected range [100, 300]"

    def test_sample_has_same_schema(self, sample_df):
        for col in REQUIRED_SEQ_COLS:
            assert col in sample_df.columns, \
                f"Missing column '{col}' in sample_data.parquet"

"""
dataset.py
──────────
PyTorch Dataset and DataLoader for Neuro-CX sequential interaction data.

Each sample contains:
  - action_seq   : (max_seq_len,)  int64   — encoded action type at each step
  - item_seq     : (max_seq_len,)  int64   — encoded item ID at each step
  - dwell_seq    : (max_seq_len,)  float32 — normalised dwell time at each step
  - weight_seq   : (max_seq_len,)  float32 — reinforcement signal weight at each step
  - seq_len      : ()              int64   — actual (unpadded) sequence length
  - target_item  : ()              int64   — next item to predict
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class InteractionSequenceDataset(Dataset):
    """
    Loads a .parquet file of padded customer interaction sequences and serves
    them as PyTorch tensors.
    """

    def __init__(self, parquet_path: str):
        """
        Parameters
        ----------
        parquet_path : str
            Path to train.parquet, val.parquet, or test.parquet.
        """
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Parquet file not found: {parquet_path}\n"
                f"Run `py run_pipeline.py` first to generate processed data."
            )

        df = pd.read_parquet(parquet_path)
        self.n_samples = len(df)

        # Parse list columns (stored as Python lists in parquet)
        def to_array(col, dtype):
            vals = df[col].tolist()
            out = []
            for v in vals:
                if isinstance(v, str):
                    v = ast.literal_eval(v)
                out.append(np.array(v, dtype=dtype))
            return np.stack(out, axis=0)

        self.action_seqs  = to_array("action_seq",  np.int64)
        self.item_seqs    = to_array("item_seq",    np.int64)
        self.dwell_seqs   = to_array("dwell_seq",   np.float32)
        self.weight_seqs  = to_array("weight_seq",  np.float32)
        self.seq_lens     = df["seq_len"].values.astype(np.int64)
        self.target_items = df["target_item"].values.astype(np.int64)
        self.customer_ids = df["customer_id"].tolist()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "action_seq":  torch.tensor(self.action_seqs[idx],  dtype=torch.long),
            "item_seq":    torch.tensor(self.item_seqs[idx],    dtype=torch.long),
            "dwell_seq":   torch.tensor(self.dwell_seqs[idx],   dtype=torch.float32),
            "weight_seq":  torch.tensor(self.weight_seqs[idx],  dtype=torch.float32),
            "seq_len":     torch.tensor(self.seq_lens[idx],     dtype=torch.long),
            "target_item": torch.tensor(self.target_items[idx], dtype=torch.long),
            "customer_id": self.customer_ids[idx],
        }


def get_dataloader(
    parquet_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Convenience factory for creating a DataLoader.

    Parameters
    ----------
    parquet_path : str
        Path to the .parquet file.
    batch_size : int
    shuffle : bool
        True for train, False for val/test.
    num_workers : int
        Set to 0 on Windows (multiprocessing issues with default spawn).
    pin_memory : bool
        True if CUDA is available.

    Returns
    -------
    DataLoader
    """
    dataset = InteractionSequenceDataset(parquet_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )


if __name__ == "__main__":
    # Quick smoke-test
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    loader = get_dataloader("data/train.parquet", batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("Batch keys:  ", list(batch.keys()))
    print("action_seq:  ", batch["action_seq"].shape)
    print("item_seq:    ", batch["item_seq"].shape)
    print("target_item: ", batch["target_item"])
    print("seq_len:     ", batch["seq_len"])

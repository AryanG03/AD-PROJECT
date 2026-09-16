"""
gru_baseline.py
───────────────
Standard GRU-based next-item recommender.
No reinforcement weighting, no forgetting/decay.
Used as the ablation baseline for comparison with Neuro-CX.

Architecture:
  [item_emb + action_emb + dwell_feat]  →  GRU  →  FC  →  logits over all items
"""


import torch
import torch.nn.functional as F
from torch import nn


class GRUBaseline(nn.Module):
    """
    Vanilla GRU sequential recommender.

    Parameters
    ----------
    n_items : int
        Vocabulary size for items (including padding idx 0).
    n_actions : int
        Number of distinct action types.
    embedding_dim : int
        Embedding size for both item and action embeddings.
    hidden_size : int
        GRU hidden state dimension.
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout probability (applied between GRU layers and before output FC).
    """

    MODEL_NAME = "gru_baseline"

    def __init__(
        self,
        n_items: int,
        n_actions: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_items     = n_items
        self.n_actions   = n_actions
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Embeddings
        self.item_emb   = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.action_emb = nn.Embedding(n_actions + 1, embedding_dim // 2, padding_idx=0)

        # Dwell time feature projection
        self.dwell_proj = nn.Linear(1, embedding_dim // 4)

        # Input size = item_emb + action_emb + dwell_proj
        input_size = embedding_dim + embedding_dim // 2 + embedding_dim // 4

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, n_items)

    def forward(
        self,
        item_seq: torch.Tensor,       # (B, T)
        action_seq: torch.Tensor,     # (B, T)
        dwell_seq: torch.Tensor,      # (B, T)
        weight_seq: torch.Tensor | None = None,  # ignored in baseline
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        logits : (B, n_items) — unnormalized scores for next-item prediction
        hidden : (num_layers, B, hidden_size) — final GRU hidden state
        """
        _B, _T = item_seq.shape

        # Build input features
        item_e   = self.item_emb(item_seq)               # (B, T, emb_dim)
        action_e = self.action_emb(action_seq)            # (B, T, emb_dim//2)
        dwell_e  = self.dwell_proj(dwell_seq.unsqueeze(-1))  # (B, T, emb_dim//4)

        x = torch.cat([item_e, action_e, dwell_e], dim=-1)   # (B, T, input_size)
        x = self.dropout(x)

        out, hidden = self.gru(x, hidden)                 # (B, T, hidden), (layers, B, H)

        # Take hidden state at the last real time-step (rightmost non-pad)
        last_out = out[:, -1, :]                          # (B, hidden)
        logits   = self.output_layer(self.dropout(last_out))  # (B, n_items)

        return logits, hidden

    @torch.no_grad()
    def predict(
        self,
        item_seq: torch.Tensor,
        action_seq: torch.Tensor,
        dwell_seq: torch.Tensor,
        top_k: int = 20,
        device: torch.device | None = None,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Inference API: returns ranked item IDs and final hidden state.

        Parameters
        ----------
        item_seq, action_seq, dwell_seq : 1-D or 2-D tensors
            Single sequence or batch. 1-D inputs are unsqueezed to (1, T).
        top_k : int
            Number of top items to return.
        device : torch.device, optional

        Returns
        -------
        ranked_items : list[int] of length top_k
        hidden_state : (num_layers, 1, hidden_size) tensor
        """
        self.eval()
        if device is not None:
            item_seq   = item_seq.to(device)
            action_seq = action_seq.to(device)
            dwell_seq  = dwell_seq.to(device)

        if item_seq.dim() == 1:
            item_seq   = item_seq.unsqueeze(0)
            action_seq = action_seq.unsqueeze(0)
            dwell_seq  = dwell_seq.unsqueeze(0)

        logits, hidden = self.forward(item_seq, action_seq, dwell_seq)
        scores         = F.softmax(logits, dim=-1)[0]     # (n_items,)
        top_k_idx      = torch.argsort(scores, descending=True)[:top_k].tolist()

        return top_k_idx, hidden

    def config_dict(self) -> dict:
        """Return constructor args for checkpoint saving."""
        return {
            "model_name":    self.MODEL_NAME,
            "n_items":       self.n_items,
            "n_actions":     self.n_actions,
            "embedding_dim": self.item_emb.embedding_dim,
            "hidden_size":   self.hidden_size,
            "num_layers":    self.num_layers,
            "dropout":       self.dropout.p,
        }


def build_baseline(cfg: dict, n_items: int, n_actions: int) -> "GRUBaseline":
    """Factory: build GRUBaseline from config dict."""
    m = cfg["model"]
    return GRUBaseline(
        n_items=n_items,
        n_actions=n_actions,
        embedding_dim=m["embedding_dim"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
    )

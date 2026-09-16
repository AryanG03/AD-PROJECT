"""
neuro_cx_model.py
─────────────────
The Neuro-CX GRU model with two key mechanisms inspired by neuroplasticity:

  1. REINFORCEMENT GATE
     Strong signals (purchases, reviews) amplify the hidden-state update
     more than weak signals (views, bounces). This is implemented as a
     per-step scaling of the GRU's candidate hidden state before it is
     written into memory.

     h_candidate_scaled = tanh(W·x + U·(α * h_prev))
     where α = f(signal_weight) ∈ [0.3, 3.0]

     Mechanistically, we inject the signal weight as an additional feature
     AND as a gating scalar on the GRU output, so stronger signals pull the
     hidden state further in the direction of the new input.

  2. FORGETTING / DECAY
     Between steps, if no strong signal is present, the hidden state
     decays exponentially toward zero:

       h_t = decay(t) * h_GRU + (1 - decay(t)) * h_prev_decayed

     decay(signal_weight) = exp(-λ * (1 / signal_weight))
     where λ = log(2) / decay_half_life

     A high-weight event (purchase) → decay ≈ 1.0 (no forgetting).
     A low-weight event (bounce)    → decay < 0.5 (significant forgetting).

NOTE: This is an architectural analogy for adaptive memory, not a
biological model. The "neuroplasticity" framing describes the mathematical
intent of the mechanisms.

Architecture:
  [item_emb + action_emb + dwell_feat + weight_feat]
      → ReinforcedGRUCell (per-step)
      → Decay gate
      → Output FC → logits over all items
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class NeuroCXModel(nn.Module):
    """
    GRU-based recommender with reinforcement weighting and forgetting decay.

    Parameters
    ----------
    n_items : int
    n_actions : int
    embedding_dim : int
    hidden_size : int
    num_layers : int
    dropout : float
    decay_half_life : float
        Steps at which a mid-strength signal has decayed to 50%.
    min_decay : float
        Floor value for the decay scalar (prevents complete state erasure).
    """

    MODEL_NAME = "neuro_cx"

    def __init__(
        self,
        n_items: int,
        n_actions: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        decay_half_life: float = 10.0,
        min_decay: float = 0.1,
    ):
        super().__init__()
        self.n_items        = n_items
        self.n_actions      = n_actions
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.decay_lambda   = math.log(2.0) / max(decay_half_life, 1.0)
        self.min_decay      = min_decay

        # Embeddings
        self.item_emb   = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.action_emb = nn.Embedding(n_actions + 1, embedding_dim // 2, padding_idx=0)

        # Dwell time + signal weight projections
        self.dwell_proj  = nn.Linear(1, embedding_dim // 4)
        self.weight_proj = nn.Linear(1, embedding_dim // 4)  # extra: weight as feature

        # Input size = item_emb + action_emb + dwell + weight
        input_size = embedding_dim + embedding_dim // 2 + embedding_dim // 4 + embedding_dim // 4

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Reinforcement gate ────────────────────────────────────────────────
        # Learns a per-step scaling factor from the signal weight
        # Output ∈ (0, 3] so strong signals can amplify, weak ones attenuate
        self.reinforce_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),          # (0, 1) — will be rescaled to (0, max_weight)
        )

        # ── Decay gate ────────────────────────────────────────────────────────
        # Learns to modulate the analytical decay with context
        self.decay_gate = nn.Sequential(
            nn.Linear(hidden_size + 1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_size),
            nn.Sigmoid(),          # per-dimension decay mask
        )

        self.dropout      = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, n_items)

        # Scale factor for reinforcement gate output
        self.register_buffer("max_reinforce", torch.tensor(3.0))

    def _analytical_decay(self, signal_weight: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample decay scalar from signal weight.

        decay = max(min_decay, exp(-λ / w))

        High w (purchase=3.0) → decay ≈ 1.0 (strong reinforcement, minimal forgetting)
        Low  w (bounce=0.3)   → decay ≈ exp(-λ/0.3) (significant forgetting)

        Parameters
        ----------
        signal_weight : (B,) tensor

        Returns
        -------
        decay : (B, 1) tensor in [min_decay, 1.0]
        """
        w = signal_weight.clamp(min=0.1)
        d = torch.exp(-self.decay_lambda / w)
        d = d.clamp(min=self.min_decay, max=1.0)
        return d.unsqueeze(-1)   # (B, 1)

    def forward(
        self,
        item_seq: torch.Tensor,       # (B, T)
        action_seq: torch.Tensor,     # (B, T)
        dwell_seq: torch.Tensor,      # (B, T)
        weight_seq: torch.Tensor,     # (B, T)  — signal weights
        hidden: torch.Tensor | None = None,
        return_all_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with reinforcement and forgetting.

        Parameters
        ----------
        return_all_hidden : bool
            If True, also return all intermediate hidden states (for dashboard).

        Returns
        -------
        logits : (B, n_items)
        hidden : (num_layers, B, hidden_size)
        all_hiddens (optional) : list of (B, hidden_size) tensors, length T
        """
        B, T = item_seq.shape

        # Build input features (same as baseline + weight feature)
        item_e   = self.item_emb(item_seq)                      # (B, T, emb)
        action_e = self.action_emb(action_seq)                  # (B, T, emb//2)
        dwell_e  = self.dwell_proj(dwell_seq.unsqueeze(-1))     # (B, T, emb//4)
        weight_e = self.weight_proj(weight_seq.unsqueeze(-1))   # (B, T, emb//4)

        x = torch.cat([item_e, action_e, dwell_e, weight_e], dim=-1)  # (B, T, input_size)
        x = self.dropout(x)

        # ── Step-wise processing (necessary for per-step decay) ───────────────
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers, B, self.hidden_size,
                device=item_seq.device, dtype=x.dtype
            )

        all_hiddens = []
        for t in range(T):
            x_t = x[:, t:t+1, :]           # (B, 1, input_size)
            w_t = weight_seq[:, t]          # (B,)

            # Standard GRU step
            _out_t, hidden = self.gru(x_t, hidden)  # out_t: (B, 1, H)

            # ── Reinforcement gate: scale hidden-state update by signal strength
            reinforce_scalar = self.reinforce_gate(w_t.unsqueeze(-1))  # (B, 1)
            reinforce_scalar = reinforce_scalar * self.max_reinforce    # scale to (0, 3]

            # Blend current GRU output with gated amplification
            # h_top = last layer hidden state
            h_top = hidden[-1]  # (B, H)
            h_reinforced = h_top * reinforce_scalar  # element-wise amplification

            # ── Decay gate: reduce influence of stale memories ────────────────
            analytic_d = self._analytical_decay(w_t)   # (B, 1)
            gate_input  = torch.cat([h_reinforced, w_t.unsqueeze(-1)], dim=-1)  # (B, H+1)
            learned_d   = self.decay_gate(gate_input)  # (B, H)

            # Combined decay: analytical × learned (both in [0,1])
            decay = analytic_d * learned_d              # (B, H)

            # Apply decay: new_h = decay * reinforced_h + (1 - decay) * prev_h_top
            # We only update the top GRU layer (others are memory sub-layers)
            h_new = decay * h_reinforced + (1 - decay) * hidden[-1]

            # Write back
            hidden = hidden.clone()
            hidden[-1] = h_new

            all_hiddens.append(h_new.detach().clone())

        # Final output from top hidden state
        last_h = hidden[-1]                             # (B, H)
        logits  = self.output_layer(self.dropout(last_h))  # (B, n_items)

        if return_all_hidden:
            return logits, hidden, all_hiddens

        return logits, hidden

    @torch.no_grad()
    def predict(
        self,
        item_seq: torch.Tensor,
        action_seq: torch.Tensor,
        dwell_seq: torch.Tensor,
        weight_seq: torch.Tensor,
        top_k: int = 20,
        device: torch.device | None = None,
        return_hidden_history: bool = False,
    ) -> tuple:
        """
        Inference API matching GRUBaseline.predict() signature (+ weight_seq).

        Returns
        -------
        ranked_items : list[int] of length top_k
        hidden_state : final hidden state tensor
        [hidden_history] : list of per-step hidden states (if return_hidden_history=True)
        """
        self.eval()
        if device is not None:
            item_seq   = item_seq.to(device)
            action_seq = action_seq.to(device)
            dwell_seq  = dwell_seq.to(device)
            weight_seq = weight_seq.to(device)

        if item_seq.dim() == 1:
            item_seq   = item_seq.unsqueeze(0)
            action_seq = action_seq.unsqueeze(0)
            dwell_seq  = dwell_seq.unsqueeze(0)
            weight_seq = weight_seq.unsqueeze(0)

        result = self.forward(
            item_seq, action_seq, dwell_seq, weight_seq,
            return_all_hidden=return_hidden_history
        )

        if return_hidden_history:
            logits, hidden, all_hiddens = result
        else:
            logits, hidden = result
            all_hiddens = None

        scores     = F.softmax(logits, dim=-1)[0]
        top_k_idx  = torch.argsort(scores, descending=True)[:top_k].tolist()

        if return_hidden_history:
            return top_k_idx, hidden, all_hiddens
        return top_k_idx, hidden

    def config_dict(self) -> dict:
        return {
            "model_name":      self.MODEL_NAME,
            "n_items":         self.n_items,
            "n_actions":       self.n_actions,
            "embedding_dim":   self.item_emb.embedding_dim,
            "hidden_size":     self.hidden_size,
            "num_layers":      self.num_layers,
            "dropout":         self.dropout.p,
            "decay_half_life": math.log(2.0) / self.decay_lambda,
            "min_decay":       self.min_decay,
        }


def build_neuro_cx(cfg: dict, n_items: int, n_actions: int) -> "NeuroCXModel":
    """Factory: build NeuroCXModel from config dict."""
    m = cfg["model"]
    return NeuroCXModel(
        n_items=n_items,
        n_actions=n_actions,
        embedding_dim=m["embedding_dim"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        decay_half_life=m.get("decay_half_life", 10.0),
        min_decay=m.get("min_decay", 0.1),
    )

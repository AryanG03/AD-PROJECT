"""gru_baseline.pyi — public API type stubs."""
import torch
from torch import nn

class GRUBaseline(nn.Module):
    MODEL_NAME: str
    n_items: int
    hidden_size: int
    num_layers: int

    def __init__(
        self,
        n_items: int,
        n_actions: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        embed_dim: int = 32,
    ) -> None: ...

    def forward(
        self,
        item_seq: torch.Tensor,
        action_seq: torch.Tensor,
        dwell_seq: torch.Tensor,
        weight_seq: torch.Tensor,
    ) -> torch.Tensor: ...

    def predict(
        self,
        item_seq: torch.Tensor,
        action_seq: torch.Tensor,
        dwell_seq: torch.Tensor,
        weight_seq: torch.Tensor,
        top_k: int = 20,
        device: torch.device | None = None,
    ) -> tuple[list[int], list[float]]: ...

    def config_dict(self) -> dict[str, object]: ...

def build_baseline(
    n_items: int,
    n_actions: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.3,
) -> GRUBaseline: ...

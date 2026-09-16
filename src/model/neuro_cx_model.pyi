"""neuro_cx_model.pyi — public API type stubs."""
import torch
from torch import nn

class NeuroCXModel(nn.Module):
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
        hidden: torch.Tensor | None = None,
        return_all_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def predict(
        self,
        item_seq: torch.Tensor,
        action_seq: torch.Tensor,
        dwell_seq: torch.Tensor,
        weight_seq: torch.Tensor,
        top_k: int = 20,
        device: torch.device | None = None,
        return_hidden_history: bool = False,
    ) -> tuple: ...

    def config_dict(self) -> dict[str, object]: ...

def build_neuro_cx(
    n_items: int,
    n_actions: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.3,
) -> NeuroCXModel: ...

"""
model_registry.py
─────────────────
Singleton that loads both model checkpoints once at startup
and provides a thread-safe accessor for route handlers.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.trainer import load_checkpoint

_MODELS: dict = {}
_DEVICE: torch.device | None = None


def startup(ckpt_dir: str, device: torch.device):
    """Load both checkpoints into memory. Call once at app startup."""
    global _DEVICE
    _DEVICE = device

    baseline_path = os.path.join(ckpt_dir, "gru_baseline_best.pt")
    neurocx_path  = os.path.join(ckpt_dir, "neuro_cx_best.pt")

    for name, path in [("baseline", baseline_path), ("neuro_cx", neurocx_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}\nRun `py run_train.py` first.")
        model, ckpt_data = load_checkpoint(path, device)
        model.eval()
        _MODELS[name] = {"model": model, "ckpt": ckpt_data}
        print(f"  Loaded {name}: epoch={ckpt_data['epoch']}, val_loss={ckpt_data['val_loss']:.4f}")


def get_model(name: str):
    """Return (model, ckpt_data) for 'baseline' or 'neuro_cx'."""
    if name not in _MODELS:
        raise KeyError(f"Model '{name}' not loaded. Valid: {list(_MODELS.keys())}")
    entry = _MODELS[name]
    return entry["model"], entry["ckpt"]


def get_device() -> torch.device:
    return _DEVICE


def model_info() -> dict:
    """Return a summary dict of loaded models."""
    result = {}
    for name, entry in _MODELS.items():
        cfg = entry["ckpt"].get("model_config", {})
        result[name] = {
            "epoch":       entry["ckpt"]["epoch"],
            "val_loss":    round(entry["ckpt"]["val_loss"], 6),
            "hidden_size": cfg.get("hidden_size", "?"),
            "num_layers":  cfg.get("num_layers", "?"),
            "model_name":  cfg.get("model_name", name),
        }
    return result

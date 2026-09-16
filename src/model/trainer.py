"""
trainer.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Training loop for both GRUBaseline and NeuroCXModel.

Features:
  - torch.cuda.amp mixed precision (GradScaler)
  - ReduceLROnPlateau scheduler
  - Gradient clipping
  - Checkpoint saving (weights + config dict)
  - CSV loss log written to logs/
"""

import csv
import json
import os
import sys
import time

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.dataset import get_dataloader
from model.gru_baseline import GRUBaseline, build_baseline
from model.neuro_cx_model import NeuroCXModel, build_neuro_cx

# â”€â”€ Loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model_name: str,
    weight_seq: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross-entropy loss. For NeuroCX, optionally upweight purchase samples.
    """
    loss = nn.functional.cross_entropy(logits, targets, reduction="none")  # (B,)
    # Weight by mean batch signal â€” batches with more purchases get higher loss weight
    if weight_seq is not None and model_name == "neuro_cx":
        batch_weights = weight_seq.mean(dim=-1).clamp(min=0.5, max=3.0)  # (B,)
        loss = (loss * batch_weights).mean()
    else:
        loss = loss.mean()
    return loss


# â”€â”€ Training loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_epoch(model, loader, optimizer, scaler, device, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    is_neuro_cx = isinstance(model, NeuroCXModel)

    for batch in loader:
        item_seq   = batch["item_seq"].to(device)
        action_seq = batch["action_seq"].to(device)
        dwell_seq  = batch["dwell_seq"].to(device)
        weight_seq = batch["weight_seq"].to(device)
        targets    = batch["target_item"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            if is_neuro_cx:
                logits, _ = model(item_seq, action_seq, dwell_seq, weight_seq)
            else:
                logits, _ = model(item_seq, action_seq, dwell_seq)

            loss = compute_loss(
                logits, targets,
                model_name=model.MODEL_NAME,
                weight_seq=weight_seq if is_neuro_cx else None,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, device, use_amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    is_neuro_cx = isinstance(model, NeuroCXModel)

    for batch in loader:
        item_seq   = batch["item_seq"].to(device)
        action_seq = batch["action_seq"].to(device)
        dwell_seq  = batch["dwell_seq"].to(device)
        weight_seq = batch["weight_seq"].to(device)
        targets    = batch["target_item"].to(device)

        with autocast(enabled=use_amp):
            if is_neuro_cx:
                logits, _ = model(item_seq, action_seq, dwell_seq, weight_seq)
            else:
                logits, _ = model(item_seq, action_seq, dwell_seq)

            loss = compute_loss(logits, targets, model_name=model.MODEL_NAME)

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# â”€â”€ Save / Load checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save_checkpoint(model, epoch: int, val_loss: float, checkpoint_dir: str, encoder_meta: dict):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{model.MODEL_NAME}_best.pt")
    torch.save({
        "epoch":        epoch,
        "val_loss":     val_loss,
        "model_state":  model.state_dict(),
        "model_config": model.config_dict(),
        "encoder_meta": encoder_meta,
    }, path)
    return path


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load a checkpoint and reconstruct the model.

    Returns
    -------
    model : GRUBaseline or NeuroCXModel
    meta  : dict with encoder_meta and training info
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]

    if cfg["model_name"] == "gru_baseline":
        model = GRUBaseline(
            n_items=cfg["n_items"],
            n_actions=cfg["n_actions"],
            embedding_dim=cfg["embedding_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    else:
        model = NeuroCXModel(
            n_items=cfg["n_items"],
            n_actions=cfg["n_actions"],
            embedding_dim=cfg["embedding_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            decay_half_life=cfg.get("decay_half_life", 10.0),
            min_decay=cfg.get("min_decay", 0.1),
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, ckpt


# â”€â”€ Full training run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train(
    model_name: str = "neurocx",
    config_path: str = "config.yaml",
    epochs_override: int | None = None,
    patience_override: int | None = None,
):
    """
    Train a model end-to-end.

    Parameters
    ----------
    model_name : str
        "baseline" or "neurocx"
    config_path : str
    epochs_override : int, optional
        Override cfg epochs (useful for quick smoke-tests).
    patience_override : int, optional
        Early-stopping patience (epochs without improvement before stopping).
        Defaults to cfg training.early_stop_patience.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]

    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Neuro-CX Trainer â€” model: {model_name}  device: {device}")
    print(f"{'='*60}")

    # Load encoder metadata to get n_items / n_actions
    enc_path = os.path.join(data_cfg["processed_dir"], "encoders.json")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(
            f"Encoders not found at {enc_path}. Run `py run_pipeline.py` first."
        )
    with open(enc_path) as f:
        enc_meta = json.load(f)

    n_items   = enc_meta["n_items"]
    n_actions = enc_meta["n_actions"]

    # Build model
    if model_name == "baseline":
        model = build_baseline(cfg, n_items, n_actions)
    else:
        model = build_neuro_cx(cfg, n_items, n_actions)

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # DataLoaders
    num_workers = 0  # Windows-safe default
    train_loader = get_dataloader(data_cfg["train_file"], batch_size=train_cfg["batch_size"],
                                  shuffle=True,  num_workers=num_workers)
    val_loader   = get_dataloader(data_cfg["val_file"],   batch_size=train_cfg["batch_size"],
                                  shuffle=False, num_workers=num_workers)

    optimizer = Adam(model.parameters(),
                     lr=train_cfg["learning_rate"],
                     weight_decay=train_cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler    = GradScaler(enabled=train_cfg["use_amp"] and device.type == "cuda")
    use_amp   = train_cfg["use_amp"] and device.type == "cuda"

    epochs  = epochs_override   if epochs_override   is not None else train_cfg["epochs"]
    patience = patience_override if patience_override is not None else train_cfg.get("early_stop_patience", 7)

    # Logging
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    log_path = os.path.join(cfg["paths"]["logs"], f"{model.MODEL_NAME}_training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    best_val_loss  = float("inf")
    best_ckpt_path = None
    no_improve     = 0          # early-stopping counter
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val_loss   = eval_epoch(model, val_loader, device, use_amp)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}  [{elapsed:.0f}s]")

        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{lr_now:.2e}", f"{elapsed:.1f}"])
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            no_improve     = 0
            best_ckpt_path = save_checkpoint(
                model, epoch, val_loss, cfg["paths"]["checkpoints"], enc_meta
            )
            print(f"  âœ“ New best checkpoint saved: {best_ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping triggered (no improvement for {patience} epochs).")
                break

    log_file.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {best_ckpt_path}")
    print(f"Log: {log_path}\n")
    return best_ckpt_path


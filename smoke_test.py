import sys, os
sys.path.insert(0, os.path.join("src"))
from model.trainer import load_checkpoint
from model.neuro_cx_model import NeuroCXModel
import torch

for name, path in [("baseline","checkpoints/gru_baseline_best.pt"), ("neurocx","checkpoints/neuro_cx_best.pt")]:
    model, ckpt = load_checkpoint(path, torch.device("cpu"))
    cfg = ckpt["model_config"]
    n_items = cfg["n_items"]
    n_actions = cfg["n_actions"]
    hidden = cfg["hidden_size"]
    print(f"{name}: {n_items} items, {n_actions} actions, hidden={hidden}")
    seq = torch.zeros(50, dtype=torch.long)
    act = torch.zeros(50, dtype=torch.long)
    dwl = torch.zeros(50, dtype=torch.float32)
    wgt = torch.ones(50, dtype=torch.float32)
    if isinstance(model, NeuroCXModel):
        items, h = model.predict(seq, act, dwl, wgt, top_k=5)
    else:
        items, h = model.predict(seq, act, dwl, top_k=5)
    print(f"  Top-5 items: {items}")

print("All imports and inference OK!")

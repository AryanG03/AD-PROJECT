# Neuro-CX

> **An adaptive customer-experience research prototype using GRU-based sequential modelling with neuroplasticity-inspired reinforcement and forgetting mechanisms.**
>
> MSc Data Science Project · Research Prototype · Not for production use.

---

## Architecture Overview

```
Customer Interaction Sequence
  [view → add_to_cart → view → purchase → review]
              │
              ▼
    ┌─────────────────────┐
    │  Item Embedding      │  (64-dim)
    │  Action Embedding    │  (32-dim)
    │  Dwell Projection    │  (16-dim)
    │  Weight Projection   │  (16-dim, Neuro-CX only)
    └─────────┬───────────┘
              │  128-dim input
              ▼
    ┌─────────────────────┐
    │  2-layer GRU        │  hidden_size=128
    │                     │
    │  + Reinforcement    │  ← purchase signals amplify state update
    │    Gate (learned)   │     α(w) ∈ (0, 3.0]
    │                     │
    │  + Forgetting Decay │  ← stale memories decay exponentially
    │    Analytical × Gated    decay = exp(-λ/w) × learned_gate
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  FC → n_items logits│  next-item prediction
    └─────────────────────┘
```

**Key mechanisms (architectural analogy — no biological data involved):**
- **Reinforcement gate**: Strong signals (purchase=3.0, review=1.5) amplify the GRU hidden-state update, mimicking how repeated or important experiences strengthen memory traces.
- **Forgetting decay**: Exponential decay term reduces the influence of older interactions unless reinforced. Configurable via `decay_half_life` in `config.yaml`.

---

## Hardware Requirements

| Component | Minimum | Tested On |
|---|---|---|
| GPU | Any CUDA GPU | NVIDIA RTX 3050 (4–6GB) |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 512 GB |
| Python | 3.10+ | 3.11.5 |
| PyTorch | 2.0+ | 2.0+ |

---

## Setup

### 1. Check CUDA

```powershell
# Check GPU visibility
nvidia-smi

# Check CUDA in Python
py -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 2. Install dependencies

```powershell
cd neuro-cx
py -m pip install -r requirements.txt
```

> **PyTorch with CUDA** — if `pip install torch` installs a CPU-only build, use the official selector:
> ```
> py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Place the raw dataset (optional)

The pipeline will automatically use `../Ecommerce_Consumer_Behavior_Analysis_Data.csv`
(one directory above `neuro-cx/`) if it exists. Otherwise it falls back to a
built-in synthetic generator.

---

## Running the Prototype

All commands are run from inside the `neuro-cx/` directory.

### Step 1 — Preprocess data

```powershell
py run_pipeline.py
```

What this does:
- Reads real CSV (or generates synthetic data)
- Synthesises per-customer sequential interaction events
- Encodes features, pads/truncates to max_seq_len=50
- Saves `data/train.parquet`, `data/val.parquet`, `data/test.parquet`
- Saves `data/encoders.json` (vocabulary sizes, label mappings)

Expected output:
```
[1/5] Loading real CSV ...  1000 customer profiles loaded.
[2/5] Synthesising interaction sequences ... 8423 events for 1000 customers.
[3/5] Encoding features ... 200 unique items, 5 action types.
[4/5] Building sequences (max_len=50) ... 997 training examples built.
[5/5] Splitting and saving ... Train: 697 | Val: 150 | Test: 150
✓ Pipeline complete.
```

---

### Step 2 — Train models

```powershell
# Train both baseline GRU and Neuro-CX (default: 20 epochs each)
py run_train.py

# Train only one model
py run_train.py --model baseline
py run_train.py --model neurocx

# Quick test (1 epoch)
py run_train.py --epochs 1
```

Training logs are saved to `logs/<model>_training_log.csv`.
Best checkpoints are saved to `checkpoints/<model>_best.pt`.

Expected training time on RTX 3050:
- Baseline GRU: ~2–4 min / 20 epochs
- Neuro-CX: ~5–8 min / 20 epochs (step-wise processing is slower)

---

### Step 3 — Evaluate

```powershell
# Compare both models on test set
py run_eval.py

# Evaluate only one model
py run_eval.py --model baseline
py run_eval.py --model neurocx --split val
```

Output:
- Console table with NDCG@5/10/20, HitRate@5/10/20, Accuracy@1
- Side-by-side comparison with Δ column
- `logs/comparison.png` — bar chart
- `logs/comparison.json` — JSON results (read by dashboard)

---

### Step 4 — Launch dashboard

```powershell
py run_app.py
```

Open **http://localhost:8501** in your browser.

Dashboard features:
| Panel | Description |
|---|---|
| Sidebar | Switch between Neuro-CX and Baseline, adjust k |
| Customer Selector | Pick any customer from the dataset |
| Sequence Stepper | Step through events one at a time (◀ / ▶) |
| Recommendations | Live ranked item list, updates each step |
| Hidden State Heatmap | 64-dim state vector as colour map over time |
| Mean Activation Line | Shows reinforcement spikes on purchase events |
| Metrics Panel | Comparison table + chart from evaluation |

---

## Configuration

Edit `config.yaml` to tune any hyperparameter:

```yaml
model:
  hidden_size: 128        # GRU hidden dim
  embedding_dim: 64       # item/action embedding size
  decay_half_life: 10.0   # steps for state to decay 50% (Neuro-CX)
  min_decay: 0.1          # floor to prevent full state erasure

training:
  epochs: 20
  batch_size: 32          # safe for RTX 3050 4GB with mixed precision
  use_amp: true           # torch.cuda.amp mixed precision

features:
  action_weights:
    purchase:    3.0      # strong reinforcement
    review:      1.5
    add_to_cart: 1.5
    view:        1.0
    bounce:      0.3      # weak signal → faster forgetting
```

---

## Project Structure

```
neuro-cx/
├── config.yaml                     # All hyperparameters
├── requirements.txt
├── run_pipeline.py                 # Step 1: preprocess
├── run_train.py                    # Step 2: train
├── run_eval.py                     # Step 3: evaluate
├── run_app.py                      # Step 4: dashboard
│
├── data/
│   ├── train.parquet               # generated by pipeline
│   ├── val.parquet
│   ├── test.parquet
│   ├── encoders.json               # vocabulary + feature metadata
│   └── interactions_sample.parquet # raw events for dashboard
│
├── src/
│   ├── data_pipeline/
│   │   ├── synthetic_generator.py  # standalone synthetic data generator
│   │   ├── preprocessor.py         # real CSV → sequences → parquet
│   │   └── dataset.py              # PyTorch Dataset + DataLoader
│   │
│   ├── model/
│   │   ├── gru_baseline.py         # vanilla GRU (ablation)
│   │   ├── neuro_cx_model.py       # GRU + reinforcement + forgetting
│   │   └── trainer.py              # training loop + checkpoint I/O
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # NDCG@k, HitRate@k, Accuracy@1
│   │   ├── harness.py              # checkpoint evaluation harness
│   │   └── compare.py              # side-by-side comparison + chart
│   │
│   └── app/
│       └── dashboard.py            # Streamlit interactive demo
│
├── checkpoints/
│   ├── gru_baseline_best.pt
│   └── neuro_cx_best.pt
│
├── logs/
│   ├── gru_baseline_training_log.csv
│   ├── neuro_cx_training_log.csv
│   ├── comparison.json
│   └── comparison.png
│
└── notebooks/
    └── 01_eda.ipynb
```

---

## What "done" looks like

1. `py run_pipeline.py` completes and produces `data/*.parquet` files.
2. `py run_train.py` trains both models, saves checkpoints with non-NaN val loss.
3. `py run_eval.py` outputs a comparison table where Neuro-CX shows improvement in at least some metrics over the baseline.
4. `py run_app.py` launches the dashboard where:
   - Recommendations visibly change as you step through interactions.
   - The hidden-state heatmap shows brighter/more active regions after purchase events.
   - After several low-weight steps (views/bounces), the heatmap shows state drift toward neutral (forgetting).
   - The metrics panel shows the comparison chart.

---

## Constraints & Non-Goals

- Research prototype — no auth, no scaling, no deployment infrastructure.
- The "neuroplasticity" framing is an **architectural analogy** in comments and docs only. No biological or neurological data is used.
- Sequences derived from a real CSV are clearly labelled as derived/synthetic in code.
- Accuracy is secondary to demonstrating the adaptive behaviour end-to-end.

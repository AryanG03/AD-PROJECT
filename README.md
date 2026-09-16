# Neuro-CX

> **An adaptive customer-experience system using GRU-based sequential modelling with neuroplasticity-inspired reinforcement and forgetting mechanisms.**

MSc Data Science Project · Research Prototype

---

## Overview

Neuro-CX extends a standard GRU recommender with two mechanisms inspired by biological memory:

- **Reinforcement Gate** — amplifies hidden-state updates from high-signal events (purchases, reviews)
- **Forgetting Gate** — attenuates stale, low-signal interactions (bounces, casual views)

Together they allow the model to build sharper, more temporally coherent customer preference representations.

## Results

| Metric | GRU Baseline | Neuro-CX | Δ |
|--------|-------------|---------|---|
| Accuracy@1 | 0.0067 | **0.0267** | +0.0200 ▲ |
| NDCG@10 | 0.0914 | **0.1008** | +0.0095 ▲ |
| HitRate@20 | 0.3600 | **0.4133** | +0.0533 ▲ |
| NDCG@20 | 0.1248 | **0.1509** | +0.0262 ▲ |

Neuro-CX outperforms the GRU baseline on **6 of 7 ranking metrics** on the held-out test set.

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + uvicorn |
| Model | PyTorch GRU (65,561 parameters) |
| Frontend | Vanilla HTML / CSS / JS |
| Data | 1,000 customer profiles · 8,945 interaction events |
| Tests | pytest · 31 unit tests (all passing) |

## Run Locally

`ash
# Install dependencies
pip install -r requirements.txt

# Start the app
py app.py

# Open in browser
http://localhost:8000
`

## Project Structure

`
neuro-cx/
├── app.py                    # FastAPI entry point
├── frontend/                 # HTML/CSS/JS SPA
│   ├── index.html
│   └── static/css/ js/
├── src/
│   ├── api/                  # REST API routes
│   ├── model/                # GRU Baseline + Neuro-CX
│   ├── data_pipeline/        # Preprocessing + DataLoader
│   ├── evaluation/           # Metrics + ablation
│   └── app/                  # Streamlit prototype
├── checkpoints/              # Trained model weights
├── data/                     # Parquet datasets + encoders
├── logs/                     # Evaluation results
└── tests/                    # 31 unit tests
`

## Team

| Member | Role |
|--------|------|
| Neha | Data pipeline, schema contract, unit tests |
| Aryan | Model architecture, training, hyperparameter tuning |
| Diya | Evaluation metrics, ablation study |
| Shravani | App, API design, deployment |

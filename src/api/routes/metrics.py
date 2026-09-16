"""
metrics.py
──────────
GET /api/metrics   — overall test-set evaluation results
GET /api/ablation  — ablation study results
GET /api/models    — loaded model info
"""

import csv
import json
import os

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["metrics"])

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LOGS_DIR = os.path.join(ROOT, "logs")


@router.get("/metrics")
def get_metrics():
    """Return the overall test-set comparison between Neuro-CX and GRU Baseline."""
    path = os.path.join(LOGS_DIR, "comparison.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="comparison.json not found. Run `py run_eval.py`.")
    with open(path) as f:
        data = json.load(f)

    baseline = data.get("baseline", {})
    neurocx  = data.get("neurocx", {})

    # Only include keys where both sides are numeric
    rows = []
    for k in baseline:
        try:
            bv = float(baseline[k])
            nv = float(neurocx.get(k, 0))
        except (TypeError, ValueError):
            continue
        rows.append({
            "metric":   k,
            "baseline": round(bv, 6),
            "neurocx":  round(nv, 6),
            "delta":    round(nv - bv, 6),
            "winner":   "neurocx" if nv >= bv else "baseline",
        })

    return {
        "rows":          rows,
        "neurocx_wins":  sum(1 for r in rows if r["winner"] == "neurocx"),
        "total_metrics": len(rows),
    }


@router.get("/ablation")
def get_ablation():
    """Return the ablation study results parsed from CSV."""
    path = os.path.join(LOGS_DIR, "ablation_results.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="ablation_results.csv not found. Run ablation.py.")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (float(v) if k != "variant" else v) for k, v in row.items()})
    return {"variants": rows}


@router.get("/models")
def get_model_info():
    """Return info about loaded models."""
    from api.model_registry import model_info
    return model_info()

"""
app.py
──────
Neuro-CX FastAPI application entry point.

Launch:
  py app.py                    # default: http://localhost:8000
  py app.py --port 8080        # custom port
"""

import argparse
import os
import sys
from contextlib import asynccontextmanager

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.model_registry import startup as load_models
from api.routes.customers import router as customers_router
from api.routes.inference import router as inference_router
from api.routes.metrics import router as metrics_router

# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("\n  Neuro-CX API starting...")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    print(f"  Device: {device}")
    load_models(ckpt_dir, device)
    print("  Ready.\n")
    yield

app = FastAPI(
    title="Neuro-CX API",
    description="Neuroplasticity-inspired GRU recommendation system -- REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ─────────────────────────────────────────────────────────────────
app.include_router(customers_router)
app.include_router(inference_router)
app.include_router(metrics_router)

# ── Static frontend ────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
STATIC_DIR   = os.path.join(FRONTEND_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def serve_index():
    index = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))}



# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"\n  Neuro-CX App  -> http://localhost:{args.port}")
    print(f"  API docs      -> http://localhost:{args.port}/docs\n")

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

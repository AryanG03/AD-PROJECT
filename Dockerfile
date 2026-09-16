# ──────────────────────────────────────────────────────────────
# Neuro-CX Dockerfile
# Builds a self-contained image with backend + frontend + models
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir fastapi uvicorn python-multipart

COPY app.py          .
COPY config.yaml     .
COPY src/            src/
COPY frontend/       frontend/
COPY checkpoints/    checkpoints/
COPY data/           data/
COPY logs/           logs/

ENV PORT=8000
EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port $PORT

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py          .
COPY config.yaml     .
COPY src/            src/
COPY frontend/       frontend/
COPY checkpoints/    checkpoints/
COPY data/           data/
COPY logs/           logs/

# HF Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port 7860

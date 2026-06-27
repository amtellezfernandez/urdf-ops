# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS trainer

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md LICENSE NOTICE ./
COPY backend ./backend/

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/outputs /app/.cache/huggingface

ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

CMD ["python3", "/app/backend/scripts/train_policy.py", "--help"]

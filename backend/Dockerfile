# ── Base: CUDA runtime for GPU-accelerated RL ─────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Force matplotlib to use the non-interactive Agg backend (no display in Docker)
    MPLBACKEND=Agg

WORKDIR /workspace

# System deps needed by OpenCV (headless), PyTorch, and CARLA client
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-venv \
        libpng16-16 \
        libjpeg-turbo8 \
        libtiff5 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip once
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies (layer-cached until requirements.txt changes)
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy full project (run.py at root, backend/ and simulator/ as packages)
COPY . /workspace

# PYTHONPATH must include the workspace root so that
#   `from backend.src.config import Config` and
#   `from simulator.envs.carla_env import ...` both resolve.
ENV PYTHONPATH="/workspace:/workspace/backend:/workspace/simulator"

# Pre-create runtime directories so volume mounts land cleanly
RUN mkdir -p /workspace/models /workspace/checkpoints

# FastAPI telemetry port
EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=10s --start-period=45s --retries=6 \
    CMD curl -fs http://localhost:8000/health || exit 1

# multi_agent_main.py starts both the FastAPI telemetry server and the RL loop
ENTRYPOINT ["python", "/workspace/backend/src/multi_agent_main.py"]
CMD ["--mode", "auto", "--algorithm", "ppo"]

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libpng16-16 \
    libjpeg8 \
    libtiff5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy project files
COPY . .

# Set PYTHONPATH to include the project root for absolute imports
ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python", "src/multi_agent_main.py"]

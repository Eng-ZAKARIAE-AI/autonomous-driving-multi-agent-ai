# AI Backend

This directory contains the core AI logic for the autonomous driving multi-agent system, including Reinforcement Learning (RL) agents, training pipelines, and telemetry services.

## Structure

- `src/`: Main source code.
  - `agents/`: Implementations of RL algorithms (PPO, SAC).
  - `training/`: Training loops and trainer logic.
  - `perception/`: Camera and sensor data processing.
  - `reward/`: Reward function definitions.
  - `evaluation/`: Metrics and evaluation scripts.
  - `visualization/`: Plotting and visualization utilities.
  - `utils/`: Common helper functions and utilities.
  - `scripts/`: Entry points for training, inference, and visualization.
  - `multi_agent_main.py`: Main entry point with FastAPI/WebSocket telemetry.
- `config/`: YAML configuration files for CARLA and RL parameters.
- `training_logs/`: Directory for storing training progress and logs.

## Getting Started

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (recommended for training)
- CARLA Simulator (0.9.15)

### Installation
You can install the dependencies using:
```bash
pip install -r requirements.txt
```
Or using the `pyproject.toml`:
```bash
pip install .
```

### Running the Backend
To start the telemetry server and the AI loop:
```bash
python src/multi_agent_main.py
```

### Training an Agent
```bash
python src/scripts/train.py --algorithm ppo --episodes 200
```

## Docker
A `Dockerfile` is provided for containerized execution with NVIDIA GPU support.

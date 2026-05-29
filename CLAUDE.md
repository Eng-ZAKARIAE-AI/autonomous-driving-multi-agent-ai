# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous driving simulator using multi-agent reinforcement learning. The system connects a CARLA simulator to PPO/SAC RL agents and exposes a real-time telemetry dashboard via FastAPI + React.

## Commands

### Backend (Python)

```bash
# Set up Python path (required for local dev)
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend:$(pwd)/simulator

# Install dependencies
pip install -r backend/requirements.txt

# Training
python run.py --mode train --algorithm sac --episodes 100
python run.py --mode train --algorithm ppo --episodes 100

# Inference (requires a trained model checkpoint)
python run.py --mode infer --algorithm sac

# Evaluation (stochastic, works without a trained model)
python run.py --mode evaluate --algorithm sac --episodes 5

# Auto mode: trains if no model found, otherwise infers
python run.py --mode auto --algorithm ppo

# Custom config override
python run.py --mode train --algorithm ppo --config backend/config/config.yaml

# Standalone scripts
python backend/src/scripts/train.py --algorithm ppo --episodes 50
python backend/src/scripts/infer.py --algorithm sac --episodes 5 --model-path models/sac_agent.pth

# Verify CARLA connection
python test_carla_connection.py
```

### Frontend (React + TypeScript)

```bash
cd frontend/web
npm install
npm run dev      # dev server at http://localhost:3000
npm run build
npm run lint
```

### Docker (full stack)

```bash
# Start all services: CARLA (port 2000), backend (port 8000), frontend (port 3000)
docker compose up --build

# Start individual services
docker compose up carla-sim
docker compose up backend
docker compose up frontend
```

## Architecture

### Data Flow

```
CARLA Simulator
  → Raw observations (speed, lane_offset, camera, collision)
  → StateBuilder: ImageEncoder (84×84 RGB → 128D) + VectorEncoder (5D kinematics → 64D) → 192D state
  → RL Agent (PPO or SAC): outputs [steer, throttle, brake] ∈ [-1, 1]
  → RewardFunction: weighted multi-objective reward
  → Trainer: logs CSV, saves checkpoints, broadcasts WebSocket telemetry
  → FastAPI (port 8000): /ws/telemetry, /telemetry
  → React Dashboard (port 3000)
```

### Key Files

| File | Role |
|------|------|
| `run.py` | Main CLI entry point |
| `backend/src/multi_agent_main.py` | FastAPI server + WebSocket telemetry broadcaster |
| `backend/src/training/trainer.py` | Training/eval/infer loop orchestrator |
| `simulator/envs/carla_env.py` | Gymnasium-compatible CARLA environment |
| `backend/src/agents/sac/sac_agent.py` | SAC agent (recommended for production) |
| `backend/src/agents/ppo/ppo_agent.py` | PPO agent |
| `backend/src/agents/common.py` | Shared `ImageEncoder` and `VectorEncoder` networks |
| `backend/src/models/state_builder.py` | Fuses visual + kinematic observations |
| `backend/src/reward/reward_function.py` | Multi-objective weighted reward |
| `backend/src/config.py` | `DEFAULT_CONFIG` dict + YAML override logic |

### Two Code Paths (Important)

There are two overlapping implementations:
- **Recommended:** `run.py` → `Trainer` → `CarlaGymEnv` + agents in `backend/src/agents/`
- **Legacy (deprecated):** `backend/src/system.py` + `backend/src/envs/carla_env.py`

Always use the main path. The legacy path is kept for reference only.

### Agent Architecture

Both PPO and SAC agents share:
- `ImageEncoder`: 3 Conv layers → Flatten → 2 FC layers → 128D
- `VectorEncoder`: 2 FC layers → 64D
- Input state: `{image: (3, 84, 84), vector: (5,)}`
- Output action: `[steer, throttle, brake]` (continuous, clipped to [-1, 1])

SAC uses `GaussianPolicy` (actor) + dual `QNetwork` critics with automatic entropy tuning. PPO uses GAE and clipped surrogate objective.

### Configuration

Config loads in priority order: built-in defaults (`DEFAULT_CONFIG` in `config.py`) → `backend/config/config.yaml` → env vars (`CARLA_HOST`, `CARLA_PORT`).

Key defaults: CARLA on `localhost:2000`, map `Town03`, target speed `16 m/s`, episodes `200`, batch size `64`, replay buffer `100k`, model saved to `models/<algorithm>_agent.pth`.

### Frontend

React 19 + TypeScript + Tailwind + Recharts + Vite. Pages in `frontend/web/src/`:
- `Dashboard.tsx`, `Training.tsx`, `Agents.tsx`, `Simulation.tsx`, `Telemetry.tsx`, `Settings.tsx`
- `useTelemetry.ts` hook subscribes to `ws://localhost:8000/ws/telemetry`

### Known Issues

- `requirements.txt`, `backend/config/config.yaml`, `backend/src/training/trainer.py`, and `backend/src/envs/carla_env.py` may contain git merge markers that need cleanup.
- No automated test suite exists; `tests/` is empty.
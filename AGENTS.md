# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project reality check (read first)
- There are **two overlapping implementations** in `src/`:
  - A Gym-style RL pipeline used by `run.py` (`src/training/trainer.py` + `src/environment/carla_env.py` + `src/rl/*`).
  - A legacy multi-agent orchestrator (`src/system.py` + `src/agents/*` + `src/envs/carla_env.py`).
- `run.py` is the main entrypoint referenced by `README.md` and should be treated as the default workflow.
- Several files currently contain unresolved merge markers (`<<<<<<<`, `=======`, `>>>>>>>`), notably:
  - `requirements.txt`
  - `config/config.yaml`
  - `src/training/trainer.py`
  - `src/envs/carla_env.py`
  These must be cleaned before reliable development/testing.

## Setup and runtime commands
- Create and activate a virtual environment (PowerShell):
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1`
- Install dependencies:
  - `pip install -r requirements.txt`
- Start CARLA (required before train/infer/evaluate):
  - Local CARLA server on `localhost:2000`, or
  - Docker: `docker compose up carla`

## Core development commands
- Train with PPO:
  - `python run.py --mode train --algorithm ppo`
- Train with SAC:
  - `python run.py --mode train --algorithm sac`
- Inference:
  - `python run.py --mode infer --algorithm ppo`
- Evaluate:
  - `python run.py --mode evaluate --algorithm sac --episodes 5`
- Use a custom config:
  - `python run.py --mode train --algorithm ppo --config config/config.yaml`
- Alternate script entrypoints (same trainer stack):
  - `python src/scripts/train.py --algorithm ppo --episodes 50`
  - `python src/scripts/infer.py --algorithm sac --episodes 5 --model-path models/sac_agent.pth`

## Build/lint/test status
- No repository-defined lint command is currently configured (no `ruff`/`flake8`/`pylint` config found).
- `tests/` currently contains only `.gitkeep` (no implemented automated tests yet).
- Fast smoke checks during development:
  - `python run.py --mode evaluate --algorithm ppo --episodes 1`
  - `python run.py --mode infer --algorithm ppo --episodes 1`
- If adding pytest-based tests, use:
  - All tests: `python -m pytest tests`
  - Single test: `python -m pytest tests/path/to/test_file.py::test_name`

## High-level architecture
### 1) Main RL training/evaluation path (default)
- `run.py` parses CLI args, loads config via `src/config.py`, and delegates to `Trainer`.
- `src/training/trainer.py` owns episode loops, logging (`checkpoints/training_log.csv`), checkpoint saving, and metric aggregation.
- `Trainer` uses `src/environment/carla_env.py::CarlaGymEnv`:
  - Handles CARLA connection/reset/step lifecycle.
  - Spawns ego vehicle + traffic + sensors.
  - Produces raw observation dict (`speed`, `lane_offset`, `goal_distance`, `goal_angle`, `camera`, `collision`).
- `StateBuilder` (`src/state_representation/state_builder.py`) converts raw observations into model input:
  - `image`: processed RGB tensor from `src/perception/camera.py`
  - `vector`: normalized kinematic/navigation features
- `RewardFunction` (`src/reward/reward_function.py`) computes weighted reward (speed, lane, progress, comfort, collision, success).
- Agent backend is selected by `--algorithm`:
  - PPO: `src/rl/ppo/ppo_agent.py`
  - SAC: `src/rl/sac/sac_agent.py`
  Both consume the same state structure (`{"image", "vector"}`).

### 2) Legacy multi-agent orchestrator path (still present)
- `src/system.py` composes `PerceptionAgent`, `PlanningAgent`, `ControlAgent`, and `DecisionAgent`.
- Uses `src/envs/carla_env.py` (different environment wrapper and observation contract).
- Entrypoints `src/train.py` and `src/infer.py` use this stack.
- Treat this as legacy/parallel code unless intentionally working on orchestrator-specific behavior.

## Important code navigation hints
- Config defaults live in `src/config.py` and are overridden by `config/config.yaml`.
- Checkpoints and logs are written under `checkpoints/` by default.
- Model path switching logic by algorithm is handled in `run.py` and `Trainer`.
- `src_backup/` is archival code; do not treat it as active runtime path unless explicitly restoring legacy behavior.

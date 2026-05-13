# Autonomous Driving Multi-Agent AI (Multi-Service Architecture)

## Overview

This project provides a professional, enterprise-grade software stack for training and deploying autonomous driving agents in the CARLA simulator. It has been refactored into a **multi-service architecture** to ensure modularity, scalability, and ease of deployment.

## Key Features

- **Multi-Service Architecture**: Decoupled Backend (AI), Simulator (CARLA), and Frontend (Dashboard).
- **Advanced RL Algorithms**: Production-ready implementations of **PPO** and **SAC** for continuous control.
- **Real-Time Telemetry**: Integrated **FastAPI** server in the backend providing live data via **WebSockets**.
- **Containerization**: Fully dockerized environment with NVIDIA GPU support for both CARLA and the AI agents.
- **Gym-Compatible Environment**: A robust wrapper for CARLA following the Gymnasium interface.

---

## Project Structure

```text
.
├── backend/                # Core AI logic and Telemetry API
│   ├── src/                # Python source code
│   │   ├── agents/         # PPO and SAC agent implementations
│   │   ├── models/         # State representation and encoders
│   │   ├── training/       # Training and evaluation pipelines
│   │   └── multi_agent_main.py # Entry point (FastAPI + AI Loop)
│   ├── config/             # YAML configurations
│   └── Dockerfile          # GPU-optimized backend image
├── simulator/              # CARLA integration layer
│   ├── envs/               # CarlaGymEnv and sensor management
│   └── carla_simulation.py # Simulation utility scripts
├── frontend/               # Visualization layer
│   └── web/                # React dashboard (FastAPI integration)
├── docker-compose.yml      # Orchestrates all services
└── requirements.txt        # Backend dependencies
```

---

## Getting Started

### Prerequisites

- **Docker & Docker Compose**
- **NVIDIA Container Toolkit** (for GPU acceleration)
- **CARLA 0.9.15** (managed automatically via Docker)

### Installation & Deployment

The recommended way to run the project is using Docker Compose:

```bash
docker compose up --build
```

This command will:
1. Start the **CARLA 0.9.15** simulator.
2. Build and launch the **AI Backend** (training/inference).
3. Launch the **Web Frontend** dashboard.

---

## Telemetry & Monitoring

The AI Backend includes a FastAPI server that broadcasts live telemetry data:

- **WebSocket Endpoint**: `ws://localhost:8000/ws/telemetry`
- **REST Endpoint**: `http://localhost:8000/telemetry`

Telemetry data includes:
- Current Speed (km/h)
- Lane Offset (m)
- Collision status
- Instantaneous Reward
- Control Actions (throttle, steer, brake)

---

## Usage (Local Development)

If you prefer to run the backend locally (without Docker):

1. **Set PYTHONPATH**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/backend:$(pwd)/simulator
   ```

2. **Launch the Backend**:
   ```bash
   python backend/src/multi_agent_main.py --mode train --algorithm sac
   ```

### Arguments:
- `--mode {train,infer,evaluate,auto}`: Execution mode.
- `--algorithm {ppo,sac}`: Choice of RL algorithm.
- `--config path/to/config.yaml`: Custom configuration path.
- `--port 8000`: Port for the telemetry server.

---

## Architecture Details

For a deep dive into the system design, algorithms, and safety features (CBF), please refer to [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# 1. System Architecture Diagram

```mermaid
graph TD
    subgraph "Frontend Layer"
        Dashboard[React Web Dashboard]
    end

    subgraph "Backend Layer (AI & API)"
        FastAPI[FastAPI Telemetry Server]
        RL_Agent[RL Agent (SAC/PPO)]
        Trainer[Trainer Loop]
        StateBuilder[State Builder / Encoder]
    end

    subgraph "Simulation Layer"
        CARLA[CARLA Simulator 0.9.15]
        GymEnv[CarlaGymEnv (Gymnasium)]
    end

    %% Interactions
    Dashboard <-->|WebSocket/REST| FastAPI
    FastAPI <-->|Shared State| Trainer
    Trainer <-->|Optimize| RL_Agent
    Trainer <-->|Step/Reset| GymEnv
    GymEnv <-->|CARLA API| CARLA
    GymEnv -->|Raw Obs| StateBuilder
    StateBuilder -->|Fused State| RL_Agent
```

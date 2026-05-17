# Slide 4: Global Architecture

## System Overview: Multi-Service Design

The system is divided into three primary layers:

1.  **Perception Layer:**
    - Handles raw sensor data (84x84 Grayscale/Color images).
    - Extracts vector states (Speed, GPS, IMU, Lane Offset).

2.  **Decision Making Layer:**
    - Core RL agents (SAC/PPO) processing fused state representations.
    - Continuous action space: Steering, Throttle, and Brake.

3.  **Simulation & Telemetry Layer:**
    - **CARLA 0.9.15:** High-fidelity simulation environment.
    - **FastAPI + WebSockets:** Real-time data broadcasting to the frontend dashboard.

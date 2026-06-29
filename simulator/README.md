# CARLA Simulator Integration

This directory contains the interface and environment wrappers for the CARLA simulator.

## Structure
- `envs/`: Gymnasium-like environment wrappers.
  - `carla_env.py`: The main `CarlaGymEnv` class that handles spawning actors, sensors, and stepping the simulation.
- `carla_simulation.py`: A utility script to test the connection to the CARLA server.

## Features
- **Gym-like Interface**: Easy integration with RL libraries.
- **Auto-Launch**: Can automatically start the CARLA simulator if configured.
- **Synchronous Mode**: Supports fixed time steps for reproducible RL training.
- **Sensor Management**: Handles RGB cameras and collision sensors.

## Usage
To test your connection to a running CARLA server:
```bash
python carla_simulation.py
```

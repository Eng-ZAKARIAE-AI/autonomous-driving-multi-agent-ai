"""Perception agent that extracts features from the CARLA environment state."""

import numpy as np
from typing import Dict, Any


class PerceptionAgent:
    """Perception agent for obstacle and lane information."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def observe(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'speed': float(observation['speed']),
            'lane_offset': float(observation['lane_offset']),
            'goal_distance': float(observation['goal_distance']),
            'nearest_obstacle_distance': float(observation['nearest_obstacle_distance']),
            'nearest_obstacle_angle': float(observation['nearest_obstacle_angle']),
            'collision': bool(observation['collision'])
        }

    def vectorize(self, perception: Dict[str, Any]) -> np.ndarray:
        return np.array([
            perception['speed'] / 30.0,
            np.tanh(perception['lane_offset'] / 5.0),
            np.tanh(perception['goal_distance'] / 100.0),
            np.tanh(perception['nearest_obstacle_distance'] / 50.0),
            np.tanh(perception['nearest_obstacle_angle'] / np.pi),
        ], dtype=np.float32)

"""Multi-agent system orchestrator for CARLA."""

import numpy as np
from typing import Dict, Any

from .config import config
from .envs.carla_env import CarlaEnv
from .agents.perception_agent import PerceptionAgent
from .agents.planning_agent import PlanningAgent
from .agents.control_agent import ControlAgent
from .agents.decision_agent import DecisionAgent


class MultiAgentSystem:
    def __init__(self, training: bool = True):
        self.config = config
        self.env = CarlaEnv(self.config)
        self.perception = PerceptionAgent(self.config)
        self.planning = PlanningAgent(self.config)
        self.control = ControlAgent(self.config)
        self.decision = DecisionAgent(self.config, training=training)
        self.current_plan: Dict[str, Any] = {}
        self.last_observation: Dict[str, Any] = {}

    def initialize(self) -> bool:
        return self.env.initialize()

    def reset(self) -> np.ndarray:
        self.last_observation = self.env.reset()
        perception = self.perception.observe(self.last_observation)
        self.current_plan = self.planning.plan(perception)
        return self._state_vector(perception)

    def step(self, action: Dict[str, float]) -> (np.ndarray, float, bool, Dict[str, Any]):
        if not self.last_observation:
            raise RuntimeError('System has not been reset.')

        control_action = self.control.compute_control(self.last_observation, action, self.current_plan)
        observation, reward, done, info = self.env.step(control_action)
        self.last_observation = observation
        perception = self.perception.observe(observation)
        self.current_plan = self.planning.plan(perception)
        state_vector = self._state_vector(perception)
        return state_vector, reward, done, info

    def _state_vector(self, perception: Dict[str, Any]) -> np.ndarray:
        return np.array([
            perception['speed'] / 30.0,
            np.tanh(perception['lane_offset'] / 5.0),
            np.tanh(perception['goal_distance'] / 100.0),
            np.tanh(perception['nearest_obstacle_distance'] / 50.0),
            np.tanh(perception['nearest_obstacle_angle'] / np.pi),
            perception['collision'] and 1.0 or 0.0
        ], dtype=np.float32)

    def close(self) -> None:
        self.env.close()

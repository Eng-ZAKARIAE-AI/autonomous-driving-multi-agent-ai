"""Reward function for autonomous driving in CARLA."""

from typing import Dict, Any


class RewardFunction:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_action = {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0}
        self.last_goal_distance = None

    def reset(self) -> None:
        self.last_action = {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0}
        self.last_goal_distance = None

    def compute(self, observation: Dict[str, Any], action: Dict[str, float]) -> float:
        weights = self.config.get('reward', {})

        speed = observation['speed']
        goal_distance = observation['goal_distance']
        lane_offset = abs(observation['lane_offset'])
        collision = observation['collision']

        target_speed = float(self.config['agents']['planning'].get('target_speed', 10.0))

        speed_weight = float(weights.get('speed_weight', self.config['simulation'].get('speed_reward', 1.0)))
        lane_weight = float(weights.get('lane_weight', self.config['simulation'].get('lane_weight', 1.0)))
        progress_weight = float(weights.get('progress_weight', self.config['simulation'].get('progress_reward', 0.3)))
        comfort_weight = float(weights.get('comfort_weight', 0.1))
        collision_penalty_config = float(weights.get('collision_penalty', self.config['simulation'].get('collision_penalty', -10.0)))
        success_reward_value = float(weights.get('success_reward', 20.0))

        # =========================
        # SPEED
        # =========================
        speed_error = abs(speed - target_speed)
        speed_reward = max(0.0, 1.0 - speed_error / (target_speed + 1e-6))

        # =========================
        # LANE
        # =========================
        lane_norm = float(self.config['simulation'].get('lane_normalizer', 5.0))
        lane_reward = max(0.0, 1.0 - lane_offset / lane_norm)

        # =========================
        # PROGRESS
        # =========================
        progress_reward = 0.0
        if self.last_goal_distance is not None:
            progress_reward = max(
                -1.0,
                min(1.0, (self.last_goal_distance - goal_distance) / (self.last_goal_distance + 1e-6))
            )

        # =========================
        # COMFORT
        # =========================
        throttle_diff = action.get('throttle', 0.0) - self.last_action['throttle']
        steer_diff = action.get('steer', 0.0) - self.last_action['steer']
        brake_diff = action.get('brake', 0.0) - self.last_action['brake']

        comfort_penalty = -(throttle_diff ** 2 + steer_diff ** 2 + brake_diff ** 2)

        # =========================
        # COLLISION
        # =========================
        collision_penalty = collision_penalty_config if collision else 0.0

        # =========================
        # SUCCESS
        # =========================
        goal_threshold = float(self.config['simulation'].get('goal_threshold', 5.0))
        success_reward = success_reward_value if goal_distance < goal_threshold else 0.0

        # =========================
        # FINAL REWARD
        # =========================
        reward = (
            speed_weight * speed_reward +
            lane_weight * lane_reward +
            progress_weight * progress_reward +
            comfort_weight * comfort_penalty +
            collision_penalty +
            success_reward
        )

        # update memory
        self.last_action = {
            'throttle': float(action.get('throttle', 0.0)),
            'steer': float(action.get('steer', 0.0)),
            'brake': float(action.get('brake', 0.0))
        }

        self.last_goal_distance = goal_distance

        return float(reward)
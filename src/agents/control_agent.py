"""Control agent for low-level throttle, brake, and steering commands."""

import numpy as np
from typing import Dict, Any


class ControlAgent:
    """Translates high-level desired speed and steering into vehicle controls."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.speed_error_integral = 0.0

    def compute_control(self, observation: Dict[str, Any], action: Dict[str, float], plan: Dict[str, Any]) -> Dict[str, float]:
        desired_speed = float(plan['target_speed']) + float(action['speed_adjust']) * 3.0
        desired_speed = np.clip(desired_speed, 0.0, 25.0)

        current_speed = float(observation['speed'])
        speed_error = desired_speed - current_speed
        self.speed_error_integral += speed_error * self.config['carla']['fixed_delta_seconds']

        throttle = np.clip(speed_error * self.config['agents']['control']['speed_gain'], 0.0, self.config['agents']['control']['max_throttle'])
        brake = np.clip(-speed_error * self.config['agents']['control']['speed_gain'], 0.0, self.config['agents']['control']['max_brake'])

        steer = float(action['steer']) * float(self.config['agents']['control']['max_steer'])
        steer = np.clip(steer, -1.0, 1.0)

        if brake > 0.0:
            throttle = 0.0

        return {
            'throttle': float(throttle),
            'brake': float(brake),
            'steer': float(steer)
        }

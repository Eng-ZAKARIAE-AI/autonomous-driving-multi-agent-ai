"""Simple planning agent for target speed and safe following."""

from typing import Dict, Any


class PlanningAgent:
    """Planning agent that defines a safe target speed."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def plan(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        target_speed = float(self.config['agents']['planning']['target_speed'])
        obstacle_distance = float(perception['nearest_obstacle_distance'])

        if obstacle_distance < float(self.config['agents']['planning']['obstacle_distance']):
            target_speed = float(self.config['agents']['planning']['slow_speed'])

        return {
            'target_speed': float(target_speed)
        }

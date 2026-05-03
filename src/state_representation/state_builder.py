"""Build multi-modal state representations for RL."""

import numpy as np
from typing import Dict, Any

from src.perception.camera import CameraProcessor


class StateBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        camera_cfg = self.config['camera']
        self.camera_processor = CameraProcessor(width=int(camera_cfg['input_width']),
                                                height=int(camera_cfg['input_height']))

    def build_state(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        image = self.camera_processor.process(observation.get('camera'))
        vector = np.array([
            observation['speed'] / float(self.config['simulation']['speed_normalizer']),
            observation['lane_offset'] / float(self.config['simulation']['lane_normalizer']),
            observation['goal_distance'] / float(self.config['simulation']['distance_normalizer']),
            observation['goal_angle'] / np.pi,
            float(observation['collision']),
        ], dtype=np.float32)

        return {
            'image': image,
            'vector': vector
        }

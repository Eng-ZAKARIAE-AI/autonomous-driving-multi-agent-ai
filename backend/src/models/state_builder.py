"""Build multi-modal state representations for RL."""

import numpy as np
from typing import Dict, Any

from backend.src.perception.camera import CameraProcessor


class StateBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        camera_cfg = self.config['camera']
        self.camera_processor = CameraProcessor(width=int(camera_cfg['input_width']),
                                                height=int(camera_cfg['input_height']))

    def build_state(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        image = self.camera_processor.process(observation.get('camera'))
        
        # Normalize vector features
        speed_norm = float(self.config['simulation'].get('speed_normalizer', 30.0))
        lane_norm = float(self.config['simulation'].get('lane_normalizer', 4.0))
        dist_norm = float(self.config['simulation'].get('distance_normalizer', 100.0))
        
        vector = np.array([
            observation['speed'] / speed_norm,
            observation['lane_offset'] / lane_norm,
            observation['goal_distance'] / dist_norm,
            observation.get('goal_angle', 0.0) / np.pi,
            float(observation.get('collision', False)),
        ], dtype=np.float32)

        return {
            'image': image,
            'vector': vector
        }

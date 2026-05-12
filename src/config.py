"""Configuration loader for the CARLA multi-agent RL project."""
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    'carla': {
        'host': 'localhost',
        'port': 2000,
        'timeout': 30.0,
        'map': 'Town03',
        'synchronous': False,
        'fixed_delta_seconds': 0.05,
        'ego_filter': 'vehicle.tesla.model3',
        'traffic_vehicles': 8,
        'pedestrians': 6,
        'auto_launch': False,
        'carla_root': None,
        'launch_command': None,
        'connect_retries': 10,
        'retry_delay': 1.0
    },
    'simulation': {
        'max_episode_steps': 400,
        'target_speed': 16.0,
        'smoothness_weight': 0.05,
        'lane_weight': 1.0,
        'collision_penalty': -100.0,
        'speed_reward': 1.0,
        'progress_reward': 0.3,
        'goal_threshold': 5.0,
        'speed_normalizer': 30.0,
        'lane_normalizer': 4.0,
        'distance_normalizer': 100.0
    },
    'reward': {
        'speed_weight': 1.0,
        'lane_weight': 1.0,
        'progress_weight': 0.3,
        'comfort_weight': 0.1,
        'collision_penalty': -100.0,
        'success_reward': 20.0
    },
    'camera': {
        'width': 84,
        'height': 84,
        'fov': 90,
        'channels': 3,
        'input_width': 84,
        'input_height': 84
    },
    'state': {
        'vector_dim': 5
    },
    'agents': {
        'decision': {
            'state_dim': 6,
            'action_dim': 3,
            'hidden_dim': 128
        },
        'planning': {
            'target_speed': 16.0,
            'slow_speed': 6.0,
            'obstacle_distance': 15.0,
            'acceleration_step': 3.0
        },
        'control': {
            'steer_gain': 0.8,
            'speed_gain': 1.0,
            'max_steer': 0.8,
            'max_throttle': 1.0,
            'max_brake': 0.7
        }
    },
    'training': {
        'episodes': 200,
        'save_every': 20,
        'checkpoint_dir': 'checkpoints',
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'ppo_epochs': 4,
        'batch_size': 64,
        'action_std': 0.5,
        'sac_alpha': 0.2,
        'target_smoothing_tau': 0.005,
        'replay_size': 100000,
        'model_path': 'models/ppo_agent.pth',
        'seed': 42
    },
    'evaluation': {
        'episodes': 5
    }
}


class Config:
    """Configuration manager."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'

        self.path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config = DEFAULT_CONFIG.copy()
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
            self._merge(config, user_config)
        return config

    def _merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._merge(base[key], value)
            else:
                base[key] = value

    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split('.')
        pointer = self._config
        try:
            for key in keys:
                pointer = pointer[key]
            return pointer
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        return self._config.get(key)

    def __getattr__(self, key: str) -> Any:
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"Config has no attribute '{key}'")


config = Config()

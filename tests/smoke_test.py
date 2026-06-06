"""
Smoke tests: verify project structure and core imports without a CARLA server.
Run with:  python -m pytest tests/smoke_test.py -v
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.absolute()
for p in [str(ROOT), str(ROOT / "backend"), str(ROOT / "simulator")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Stub out external C-extensions that are unavailable in CI ─────────────────
_MOCKS = {
    "carla": MagicMock(),
    "cv2": MagicMock(),
    "ray": MagicMock(),
    "ray.rllib": MagicMock(),
}
for mod, mock in _MOCKS.items():
    sys.modules.setdefault(mod, mock)

# Ensure cv2.resize returns a valid numpy-shaped array
import numpy as np
sys.modules["cv2"].resize = lambda img, size, **kw: np.zeros((*size[::-1], 3), dtype=np.uint8)
sys.modules["cv2"].INTER_AREA = 1


class TestConfig(unittest.TestCase):
    def test_default_config_loads(self):
        from backend.src.config import Config
        cfg = Config()
        self.assertIn("carla", cfg._config)
        self.assertIn("training", cfg._config)
        self.assertEqual(cfg["carla"]["host"], "localhost")

    def test_env_var_override(self):
        from backend.src.config import Config
        with patch.dict("os.environ", {"CARLA_HOST": "carla-sim", "CARLA_PORT": "2001"}):
            cfg = Config()
        self.assertEqual(cfg["carla"]["host"], "carla-sim")
        self.assertEqual(cfg["carla"]["port"], 2001)


class TestAgentNetworks(unittest.TestCase):
    def test_image_encoder_forward(self):
        import torch
        from backend.src.agents.common import ImageEncoder
        enc = ImageEncoder(input_channels=3, feature_dim=128)
        x = torch.zeros(1, 3, 84, 84)
        out = enc(x)
        self.assertEqual(out.shape, (1, 128))

    def test_vector_encoder_forward(self):
        import torch
        from backend.src.agents.common import VectorEncoder
        enc = VectorEncoder(input_dim=5, feature_dim=64)
        x = torch.zeros(1, 5)
        out = enc(x)
        self.assertEqual(out.shape, (1, 64))

    def test_ppo_agent_select_action(self):
        import torch
        from backend.src.config import Config
        from backend.src.agents.ppo.ppo_agent import PPOAgent
        cfg = Config()
        agent = PPOAgent(cfg._config)
        state = {
            "image": np.zeros((3, 84, 84), dtype=np.float32),
            "vector": np.zeros(5, dtype=np.float32),
        }
        result = agent.select_action(state, deterministic=True)
        self.assertIn("action", result)
        self.assertEqual(len(result["action"]), 3)

    def test_sac_agent_select_action(self):
        from backend.src.config import Config
        from backend.src.agents.sac.sac_agent import SACAgent
        cfg = Config()
        agent = SACAgent(cfg._config)
        state = {
            "image": np.zeros((3, 84, 84), dtype=np.float32),
            "vector": np.zeros(5, dtype=np.float32),
        }
        result = agent.select_action(state, deterministic=True)
        self.assertIn("action", result)
        self.assertEqual(len(result["action"]), 3)


class TestRewardFunction(unittest.TestCase):
    def test_no_collision_positive_reward(self):
        from backend.src.config import Config
        from backend.src.reward.reward_function import RewardFunction
        cfg = Config()
        rf = RewardFunction(cfg._config)
        obs = {
            "speed": 16.0,
            "lane_offset": 0.0,
            "goal_distance": 30.0,
            "collision": False,
        }
        action = {"steer": 0.0, "throttle": 0.5, "brake": 0.0}
        reward = rf.compute(obs, action)
        self.assertGreater(reward, 0.0)

    def test_collision_gives_penalty(self):
        from backend.src.config import Config
        from backend.src.reward.reward_function import RewardFunction
        cfg = Config()
        rf = RewardFunction(cfg._config)
        obs = {
            "speed": 0.0,
            "lane_offset": 0.0,
            "goal_distance": 50.0,
            "collision": True,
        }
        action = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
        reward = rf.compute(obs, action)
        self.assertLess(reward, 0.0)


class TestStateBuilder(unittest.TestCase):
    def test_build_state_shape(self):
        from backend.src.config import Config
        from backend.src.models.state_builder import StateBuilder
        cfg = Config()
        sb = StateBuilder(cfg._config)
        obs = {
            "speed": 10.0,
            "lane_offset": 0.1,
            "goal_distance": 20.0,
            "goal_angle": 0.2,
            "collision": False,
            "camera": None,  # CameraProcessor handles None → zeros
        }
        state = sb.build_state(obs)
        self.assertIn("image", state)
        self.assertIn("vector", state)
        self.assertEqual(state["image"].shape, (3, 84, 84))
        self.assertEqual(state["vector"].shape, (5,))


class TestMetrics(unittest.TestCase):
    def test_empty_results(self):
        from backend.src.evaluation.metrics import compute_metrics
        m = compute_metrics([])
        self.assertEqual(m["collision_rate"], 0.0)

    def test_full_results(self):
        from backend.src.evaluation.metrics import compute_metrics
        results = [
            {"reward": 10.0, "steps": 100, "collision": False, "avg_speed": 15.0, "lane_deviation": 0.1},
            {"reward": -5.0, "steps": 50,  "collision": True,  "avg_speed": 5.0,  "lane_deviation": 0.5},
        ]
        m = compute_metrics(results)
        self.assertAlmostEqual(m["collision_rate"], 0.5)
        self.assertAlmostEqual(m["average_reward"], 2.5)
        self.assertAlmostEqual(m["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

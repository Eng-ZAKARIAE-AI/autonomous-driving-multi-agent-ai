"""
Smoke test to verify the integrity of the project structure and imports.
"""

import sys
import unittest
from unittest.mock import MagicMock
from pathlib import Path

# Mock carla before imports
mock_carla = MagicMock()
sys.modules["carla"] = mock_carla

# Add project root to sys.path
root_dir = Path(__file__).parent.parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

class TestProjectIntegrity(unittest.TestCase):
    def test_imports(self):
        """Verify that essential modules can be imported without error."""
        try:
            from backend.src.config import Config
            from backend.src.training.trainer import Trainer
            from simulator.envs.carla_env import CarlaGymEnv
            from backend.src.agents.ppo.ppo_agent import PPOAgent
            from backend.src.agents.sac.sac_agent import SACAgent
            print("✅ Essential modules imported successfully.")
        except ImportError as e:
            self.fail(f"❌ Import failed: {e}")

    def test_config_loading(self):
        """Verify that the configuration can be loaded correctly."""
        from backend.src.config import Config
        try:
            config = Config()
            self.assertIsNotNone(config)
            self.assertIn('carla', config._config)
            print("✅ Configuration loaded successfully.")
        except Exception as e:
            self.fail(f"❌ Config loading failed: {e}")

    def test_trainer_instantiation(self):
        """Verify that the Trainer can be instantiated (requires config but not necessarily CARLA)."""
        from backend.src.config import Config
        from backend.src.training.trainer import Trainer
        try:
            config = Config()
            # We don't initialize the environment here to avoid requiring a CARLA server
            trainer = Trainer(config, algorithm='ppo')
            self.assertIsNotNone(trainer)
            self.assertEqual(trainer.algorithm, 'ppo')
            print("✅ Trainer instantiated successfully.")
        except Exception as e:
            self.fail(f"❌ Trainer instantiation failed: {e}")

if __name__ == '__main__':
    unittest.main()

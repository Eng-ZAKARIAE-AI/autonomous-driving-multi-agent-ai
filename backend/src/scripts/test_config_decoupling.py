"""Verify config decoupling: perception_camera vs rl_encoder vs StateBuilder."""
import numpy as np
import yaml

with open("backend/config/config.yaml") as f:
    cfg = yaml.safe_load(f)

assert "perception_camera" in cfg, "MISSING: perception_camera"
assert "rl_encoder" in cfg, "MISSING: rl_encoder"
assert cfg["perception_camera"]["width"] == 640
assert cfg["perception_camera"]["height"] == 640
assert cfg["rl_encoder"]["width"] == 84
assert cfg["rl_encoder"]["height"] == 84
print("[PASS] perception_camera = 640x640, rl_encoder = 84x84")

from backend.src.models.state_builder import StateBuilder
sb = StateBuilder(cfg)
assert sb.img_w == 84, f"StateBuilder should use rl_encoder=84, got {sb.img_w}"
assert sb.img_h == 84
print("[PASS] StateBuilder reads from rl_encoder (84x84)")

obs = {
    "camera": np.zeros((640, 640, 3), dtype="uint8"),
    "speed": 5.0, "lane_offset": 0.1,
    "goal_distance": 50.0, "goal_angle": 0.2, "collision": False,
}
state = sb.build_state(obs)
assert state["image"].shape == (3, 84, 84), f"Bad shape: {state['image'].shape}"
print("[PASS] StateBuilder resizes 640x640 CARLA frame to (3, 84, 84) RL tensor")

# Verify carla_env.py uses perception_camera key
with open("simulator/envs/carla_env.py", encoding="utf-8") as f:
    src = f.read()
assert "perception_camera" in src
assert "config.get('perception_camera'" in src
print("[PASS] carla_env.py uses perception_camera key for CARLA blueprint setup")

print("[OK] Config decoupling verified.")

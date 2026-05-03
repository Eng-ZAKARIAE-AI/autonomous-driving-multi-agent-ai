"""Inference runner for the CARLA multi-agent RL system."""

from pathlib import Path
from typing import Optional
import cv2

from .config import config
from .system import MultiAgentSystem


def run_inference(model_path: Optional[str] = None, episodes: Optional[int] = None) -> None:
    model_path = model_path or config['training']['model_path']
    episodes = int(episodes or 5)
    model_path = Path(model_path)
    
    camera_output_dir = Path('results/camera_frames')
    camera_output_dir.mkdir(parents=True, exist_ok=True)

    system = MultiAgentSystem(training=False)
    print('🚀 Starting CARLA inference session...')
    if not system.initialize():
        raise RuntimeError('CARLA environment initialization failed. Make sure the server is running.')

    if model_path.exists():
        system.decision.load(str(model_path))
        print(f"✅ Loaded policy from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    for episode in range(1, episodes + 1):
        state = system.reset()
        episode_reward = 0.0
        step = 0
        episode_dir = camera_output_dir / f'episode_{episode:03d}'
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Episode {episode} started")

        while True:
            decision = system.decision.select_action(state, deterministic=True)
            action = {
                'steer': float(decision['action'][0]),
                'speed_adjust': float(decision['action'][1])
            }
            state, reward, done, info = system.step(action)
            
            # Save camera frame (non-blocking)
            camera_image = system.env.camera_image
            if camera_image is not None and step % 5 == 0:  # Save every 5th frame to reduce I/O
                frame_path = episode_dir / f'frame_{step:06d}.png'
                cv2.imwrite(str(frame_path), cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR))
            
            if step == 0:
                print(f"  Step 0: action={action}, ego_speed={system.env.ego.get_velocity() if system.env.ego else 'N/A'}")
            
            episode_reward += reward
            step += 1
            if done:
                break

        print(f"Inference episode {episode}/{episodes} | reward={episode_reward:.2f} | steps={step} | frames saved to {episode_dir}")

    print(f"✅ All camera frames saved to {camera_output_dir}")
    system.close()

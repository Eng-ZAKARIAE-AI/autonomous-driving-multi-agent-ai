"""Entry point for running inference in CARLA."""

import argparse
from pathlib import Path
import numpy as np
from backend.src.config import Config
from simulator.envs.carla_env import CarlaGymEnv
from backend.src.agents.ppo.ppo_agent import PPOAgent
from backend.src.agents.sac.sac_agent import SACAgent


def main():
    parser = argparse.ArgumentParser(description='Run inference in CARLA with trained RL agent.')
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo', help='RL algorithm to load')
    parser.add_argument('--episodes', type=int, default=5, help='Number of inference episodes')
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    config = Config(args.config) if args.config else Config()
    env = CarlaGymEnv(config)
    if not env.initialize():
        raise RuntimeError('CARLA environment initialization failed.')

    algorithm = args.algorithm.lower()
    agent = PPOAgent(config) if algorithm == 'ppo' else SACAgent(config)
    default_model_path = Path(config['training']['model_path'])
    if args.model_path is None and default_model_path.name in ('ppo_agent.pth', 'sac_agent.pth'):
        model_path = str(default_model_path.with_name(f'{algorithm}_agent.pth'))
    else:
        model_path = args.model_path or config['training']['model_path']
    agent.load(model_path)

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        step = 0

        while True:
            action_output = agent.select_action(state, deterministic=True)
            throttle_raw = float(np.clip(action_output['action'][1], -1.0, 1.0))
            brake_raw = float(np.clip(action_output['action'][2], -1.0, 1.0))
            throttle = float(max(0.0, min(1.0, (throttle_raw + 1.0) / 2.0)))
            brake = float(max(0.0, min(1.0, (brake_raw + 1.0) / 2.0)))
            if throttle > brake:
                brake = 0.0
            else:
                throttle = 0.0
            action = {
                'steer': float(action_output['action'][0]),
                'throttle': throttle,
                'brake': brake
            }
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            if done:
                break

        print(f"Inference episode {episode}/{args.episodes} | reward={episode_reward:.2f} | steps={step}")

    env.close()


if __name__ == '__main__':
    main()

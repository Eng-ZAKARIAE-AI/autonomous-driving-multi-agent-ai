"""Entry point for training RL agents in CARLA."""

import argparse
from backend.src.config import Config
from backend.src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train SAC or PPO agent in CARLA.')
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo', help='RL algorithm')
    parser.add_argument('--episodes', type=int, default=None, help='Number of training episodes')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    config = Config(args.config) if args.config else Config()
    trainer = Trainer(config, algorithm=args.algorithm)
    trainer.train(episodes=args.episodes)


if __name__ == '__main__':
    main()

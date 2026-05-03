"""Root entrypoint for the CARLA multi-agent RL project."""

import argparse
from pathlib import Path

from src.config import Config
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Run or train the CARLA multi-agent RL system.')
    parser.add_argument('--mode', choices=['auto', 'train', 'infer', 'evaluate'], default='auto', help='Execution mode.')
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo', help='RL algorithm.')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes to train or infer.')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the model file.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(args.config) if args.config else Config()
    base_model_path = Path(config['training']['model_path'])
    if args.model_path is None and base_model_path.name in ('ppo_agent.pth', 'sac_agent.pth'):
        model_path = base_model_path.with_name(f'{args.algorithm}_agent.pth')
    else:
        model_path = Path(args.model_path or config['training']['model_path'])
    trainer = Trainer(config, algorithm=args.algorithm)

    if args.mode == 'train':
        trainer.train(episodes=args.episodes)
        return

    if args.mode == 'evaluate':
        trainer.agent.load(str(model_path))
        metrics = trainer.evaluate(episodes=int(args.episodes or config['evaluation']['episodes']))
        print('Evaluation results:')
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')
        return

    if args.mode == 'infer':
        trainer.agent.load(str(model_path))
        metrics = trainer.evaluate(episodes=int(args.episodes or config['evaluation']['episodes']), deterministic=True)
        print('Inference results:')
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')
        return

    print('🔍 Auto mode: checking for existing trained policy...')
    if model_path.exists():
        print('✅ Found trained model. Launching inference.')
        trainer.agent.load(str(model_path))
        metrics = trainer.evaluate(episodes=int(args.episodes or config['evaluation']['episodes']), deterministic=True)
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')
    else:
        print('⚠️ No trained model found. Starting training first.')
        trainer.train(episodes=args.episodes)
        print('✅ Training finished. Starting inference.')
        trainer.agent.load(str(model_path))
        metrics = trainer.evaluate(episodes=int(args.episodes or config['evaluation']['episodes']), deterministic=True)
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')


if __name__ == '__main__':
    main()

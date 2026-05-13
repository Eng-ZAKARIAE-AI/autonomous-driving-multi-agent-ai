"""
Main entry point for the Autonomous Driving Multi-Agent AI system.
Unified pipeline for training, evaluation, and inference.
"""

import argparse
import sys
from pathlib import Path

# Ensure the root directory is in sys.path
root_dir = Path(__file__).parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.src.config import Config
from backend.src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Autonomous Driving RL Pipeline')
    parser.add_argument('--mode', choices=['train', 'infer', 'evaluate', 'auto'], default='auto',
                        help='Execution mode: train, infer (deterministic eval), evaluate (stochastic eval), or auto.')
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo',
                        help='Reinforcement Learning algorithm to use.')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes for the specified mode.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a custom YAML configuration file.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a specific model checkpoint to load.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Override model path if provided
    if args.model_path:
        config['training']['model_path'] = args.model_path
    
    print(f"🛠️  Starting system in {args.mode} mode using {args.algorithm.upper()}...")
    
    trainer = Trainer(config, algorithm=args.algorithm)
    
    if args.mode == 'train':
        trainer.train(episodes=args.episodes)
    elif args.mode == 'evaluate':
        # Load model if it exists
        model_path = Path(config['training']['model_path'])
        if model_path.exists():
            trainer.agent.load(str(model_path))
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"⚠️  No model found at {model_path}, evaluating with random initialization.")
        
        episodes = args.episodes or config['evaluation']['episodes']
        trainer.evaluate(episodes=int(episodes), deterministic=False)
        
    elif args.mode == 'infer':
        model_path = Path(config['training']['model_path'])
        if not model_path.exists():
            print(f"❌ Error: Model not found at {model_path}. Inference requires a trained model.")
            sys.exit(1)
            
        trainer.agent.load(str(model_path))
        print(f"✅ Loaded model from {model_path}")
        
        episodes = args.episodes or 1
        trainer.evaluate(episodes=int(episodes), deterministic=True)
        
    else: # auto mode
        model_path = Path(config['training']['model_path'])
        if model_path.exists():
            print(f"✨ Model found, starting inference...")
            trainer.agent.load(str(model_path))
            trainer.evaluate(episodes=args.episodes or 1, deterministic=True)
        else:
            print(f"✨ No model found, starting training...")
            trainer.train(episodes=args.episodes)

if __name__ == '__main__':
    main()

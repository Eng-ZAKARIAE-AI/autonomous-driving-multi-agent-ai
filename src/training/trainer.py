"""Training pipeline for CARLA autonomous driving agents."""

import csv
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
<<<<<<< HEAD
from src.config import config
=======
>>>>>>> clean-branch
from src.environment.carla_env import CarlaGymEnv
from src.rl.ppo.ppo_agent import PPOAgent
from src.rl.sac.sac_agent import SACAgent
from src.evaluation.metrics import compute_metrics
from src.visualization.plots import plot_training_history


class Trainer:
    def __init__(self, config: Dict[str, Any], algorithm: str = 'ppo'):
        self.config = config
        self.algorithm = algorithm.lower()
<<<<<<< HEAD
=======

        # Fix model path
>>>>>>> clean-branch
        if self.algorithm == 'sac' and Path(config['training']['model_path']).name == 'ppo_agent.pth':
            config['training']['model_path'] = str(Path(config['training']['model_path']).with_name('sac_agent.pth'))
        elif self.algorithm == 'ppo' and Path(config['training']['model_path']).name == 'sac_agent.pth':
            config['training']['model_path'] = str(Path(config['training']['model_path']).with_name('ppo_agent.pth'))
<<<<<<< HEAD
        self.env = CarlaGymEnv(config)
        self.agent = PPOAgent(config) if self.algorithm == 'ppo' else SACAgent(config)
        self.save_every = int(config['training']['save_every'])
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.checkpoint_dir / 'training_log.csv'
=======

        self.env = CarlaGymEnv(config)
        self.agent = PPOAgent(config) if self.algorithm == 'ppo' else SACAgent(config)

        self.save_every = int(config['training']['save_every'])
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.checkpoint_dir / 'training_log.csv'

>>>>>>> clean-branch
        self.history = {
            'episode': [],
            'reward': [],
            'avg_speed': [],
            'lane_deviation': []
        }
<<<<<<< HEAD
        self._prepare_logging()
        self._set_seed(int(self.config['training']['seed']))

    def _prepare_logging(self) -> None:
        if not self.log_path.exists():
            with open(self.log_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['episode', 'reward', 'steps', 'collision', 'avg_speed', 'lane_deviation', 'elapsed'])

    def _set_seed(self, seed: int) -> None:
=======

        self._prepare_logging()
        self._set_seed(int(self.config['training']['seed']))

    def _prepare_logging(self):
        if not self.log_path.exists():
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'reward', 'steps', 'collision', 'avg_speed', 'lane_deviation', 'elapsed'])

    def _set_seed(self, seed: int):
>>>>>>> clean-branch
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

<<<<<<< HEAD
    def train(self, episodes: Optional[int] = None) -> None:
        episodes = int(episodes or self.config['training']['episodes'])
        if not self.env.initialize():
            raise RuntimeError('CARLA environment initialization failed.')

        for episode in range(1, episodes + 1):
            start = time.time()
            state = self.env.reset()
=======
    def train(self, episodes: Optional[int] = None):
        episodes = int(episodes or self.config['training']['episodes'])
        max_steps = int(self.config['training'].get('max_steps', 500))

        print("🚀 Initializing CARLA...")
        if not self.env.initialize():
            raise RuntimeError("CARLA init failed")

        for episode in range(1, episodes + 1):
            print(f"\n🚀 EPISODE {episode}/{episodes}")

            # Reset sécurisé
            try:
                state = self.env.reset()
            except Exception as e:
                print("❌ Reset error:", e)
                break

>>>>>>> clean-branch
            episode_reward = 0.0
            step = 0
            collision = False
            speeds = []
            lane_deviations = []

<<<<<<< HEAD
            while True:
                action_output = self.agent.select_action(state, deterministic=False)
                raw_action = action_output['action']
                env_action = self._format_action(raw_action)
                next_state, reward, done, info = self.env.step(env_action)

                episode_reward += reward
                speeds.append(info['speed'])
                lane_deviations.append(abs(info['lane_offset']))
                collision = collision or bool(info['collision'])
=======
            done = False  # ✅ CORRECTION CRITIQUE

            start = time.time()

            while not done and step < max_steps:
                try:
                    action_output = self.agent.select_action(state, deterministic=False)
                    raw_action = action_output['action']
                    env_action = self._format_action(raw_action)

                    next_state, reward, done, info = self.env.step(env_action)

                except Exception as e:
                    print("❌ Step error:", e)
                    break

                episode_reward += reward
                speeds.append(info.get('speed', 0.0))
                lane_deviations.append(abs(info.get('lane_offset', 0.0)))
                collision = collision or bool(info.get('collision', False))
>>>>>>> clean-branch
                step += 1

                if self.algorithm == 'sac':
                    self.agent.store_transition(state, raw_action, reward, next_state, done)
                    self.agent.update()
                else:
<<<<<<< HEAD
                    self.agent.store_transition(state, raw_action, action_output['log_prob'], action_output['value'], reward, done)

                state = next_state
                if done:
                    break

            if self.algorithm == 'ppo':
                self.agent.update(next_value=0.0)
=======
                    self.agent.store_transition(
                        state,
                        raw_action,
                        action_output['log_prob'],
                        action_output['value'],
                        reward,
                        done
                    )

                state = next_state

            # PPO update
            if self.algorithm == 'ppo':
                try:
                    self.agent.update(next_value=0.0)
                except Exception as e:
                    print("⚠️ PPO update error:", e)
>>>>>>> clean-branch

            elapsed = time.time() - start
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            avg_lane = float(np.mean(lane_deviations)) if lane_deviations else 0.0
<<<<<<< HEAD
=======

>>>>>>> clean-branch
            self.history['episode'].append(episode)
            self.history['reward'].append(episode_reward)
            self.history['avg_speed'].append(avg_speed)
            self.history['lane_deviation'].append(avg_lane)
<<<<<<< HEAD
            self._write_log(episode, episode_reward, step, collision, avg_speed, avg_lane, elapsed)

            if episode % self.save_every == 0 or episode == episodes:
                self.agent.save(str(self.checkpoint_dir / f'{self.algorithm}_episode_{episode}.pt'))

            print(f"Episode {episode}/{episodes} | reward={episode_reward:.2f} | steps={step} | collision={collision} | speed={avg_speed:.2f} | lane={avg_lane:.2f} | time={elapsed:.1f}s")

        self._save_training_plot()

    def _save_training_plot(self) -> None:
        plot_path = self.checkpoint_dir / 'training_history.png'
        try:
            plot_training_history(str(plot_path), self.history)
            print(f"✅ Saved training history plot to {plot_path}")
        except Exception as exc:
            print(f"⚠️ Failed to save training plot: {exc}")

    def evaluate(self, episodes: int = 5, deterministic: bool = True) -> Dict[str, Any]:
        results = []
        if not self.env.initialize():
            raise RuntimeError('CARLA environment initialization failed.')

        for episode in range(1, episodes + 1):
            state = self.env.reset()
=======

            self._write_log(episode, episode_reward, step, collision, avg_speed, avg_lane, elapsed)

            # Save model
            if episode % self.save_every == 0 or episode == episodes:
                try:
                    self.agent.save(str(self.checkpoint_dir / f"{self.algorithm}_episode_{episode}.pt"))
                except Exception as e:
                    print("⚠️ Save error:", e)

            print(f"Episode {episode} | reward={episode_reward:.2f} | steps={step} | collision={collision}")

        print("\n✅ TRAINING DONE")
        self._save_training_plot()

    def _save_training_plot(self):
        plot_path = self.checkpoint_dir / 'training_history.png'
        try:
            plot_training_history(str(plot_path), self.history)
            print(f"✅ Plot saved: {plot_path}")
        except Exception as e:
            print("⚠️ Plot error:", e)

    def evaluate(self, episodes: int = 5, deterministic: bool = True):
        results = []

        if not self.env.initialize():
            raise RuntimeError("CARLA init failed")

        for episode in range(1, episodes + 1):
            state = self.env.reset()

>>>>>>> clean-branch
            episode_reward = 0.0
            step = 0
            collision = False
            speeds = []
            lane_deviations = []

<<<<<<< HEAD
            while True:
                action_output = self.agent.select_action(state, deterministic=deterministic)
                action = self._format_action(action_output['action'])
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                speeds.append(info['speed'])
                lane_deviations.append(abs(info['lane_offset']))
                collision = collision or bool(info['collision'])
                step += 1
                if done:
                    break
=======
            done = False

            while not done:
                action_output = self.agent.select_action(state, deterministic=deterministic)
                action = self._format_action(action_output['action'])

                state, reward, done, info = self.env.step(action)

                episode_reward += reward
                speeds.append(info.get('speed', 0.0))
                lane_deviations.append(abs(info.get('lane_offset', 0.0)))
                collision = collision or bool(info.get('collision', False))
                step += 1
>>>>>>> clean-branch

            results.append({
                'reward': episode_reward,
                'steps': step,
                'collision': collision,
                'avg_speed': float(np.mean(speeds)) if speeds else 0.0,
                'lane_deviation': float(np.mean(lane_deviations)) if lane_deviations else 0.0
            })
<<<<<<< HEAD
            print(f"Eval {episode}/{episodes} | reward={episode_reward:.2f} | steps={step} | collision={collision}")
=======

            print(f"Eval {episode} | reward={episode_reward:.2f}")
>>>>>>> clean-branch

        return compute_metrics(results)

    def _format_action(self, action):
<<<<<<< HEAD
        steer = float(np.clip(action[0], -1.0, 1.0)) if len(action) > 0 else 0.0
        throttle_raw = float(np.clip(action[1], -1.0, 1.0)) if len(action) > 1 else -1.0
        brake_raw = float(np.clip(action[2], -1.0, 1.0)) if len(action) > 2 else -1.0

        throttle = float(np.clip((throttle_raw + 1.0) / 2.0, 0.0, 1.0))
        brake = float(np.clip((brake_raw + 1.0) / 2.0, 0.0, 1.0))
=======
        steer = float(np.clip(action[0], -1, 1)) if len(action) > 0 else 0.0
        throttle_raw = float(np.clip(action[1], -1, 1)) if len(action) > 1 else -1.0
        brake_raw = float(np.clip(action[2], -1, 1)) if len(action) > 2 else -1.0

        throttle = (throttle_raw + 1) / 2
        brake = (brake_raw + 1) / 2
>>>>>>> clean-branch

        if throttle > brake:
            brake = 0.0
        else:
            throttle = 0.0

<<<<<<< HEAD
        return {
            'steer': steer,
            'throttle': throttle,
            'brake': brake
        }

    def _write_log(self, episode: int, reward: float, steps: int, collision: bool,
                   avg_speed: float, lane_deviation: float, elapsed: float) -> None:
        with open(self.log_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([episode, reward, steps, int(collision), avg_speed, lane_deviation, elapsed])
=======
        return {'steer': steer, 'throttle': throttle, 'brake': brake}

    def _write_log(self, episode, reward, steps, collision, avg_speed, lane_dev, elapsed):
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, steps, int(collision), avg_speed, lane_dev, elapsed])
>>>>>>> clean-branch

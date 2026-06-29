"""Benchmark: compare a trained RL agent (PPO or SAC) against BaselineAgent.

Usage
-----
python -m backend.src.scripts.benchmark \
    --rl-algorithm ppo \
    --rl-model checkpoints/ppo_episode_200.pt \
    --episodes 10 \
    --seed 42 \
    --output benchmark_results.json
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend.src.config import Config
from backend.src.training.trainer import Trainer
from backend.src.agents.baseline.baseline_agent import BaselineAgent
from backend.src.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _run_agent(trainer: Trainer, agent, episodes: int, seeds: List[int]) -> List[Dict[str, Any]]:
    """Run *agent* for *episodes* episodes and return per-episode result dicts."""
    results: List[Dict[str, Any]] = []

    if not trainer.env.initialize():
        raise RuntimeError("CARLA environment initialisation failed.")

    for ep_idx, seed in enumerate(seeds):
        _set_seed(seed)
        logger.info("Episode %d/%d  seed=%d", ep_idx + 1, episodes, seed)

        try:
            state = trainer.env.reset()
        except Exception as exc:
            logger.error("Reset error: %s", exc)
            continue

        ep_reward = 0.0
        step = 0
        collision = False
        speeds: List[float] = []
        lane_devs: List[float] = []
        done = False

        while not done:
            try:
                action_output = agent.select_action(state, deterministic=True)
                action_dict = trainer._format_action(action_output["action"])
                state, reward, done, info = trainer.env.step(action_dict)
            except Exception as exc:
                logger.error("Step error: %s", exc)
                break

            ep_reward += reward
            speeds.append(info.get("speed", 0.0))
            lane_devs.append(abs(info.get("lane_offset", 0.0)))
            collision = collision or bool(info.get("collision", False))
            step += 1

        results.append(
            {
                "episode": ep_idx + 1,
                "reward": ep_reward,
                "steps": step,
                "collision": collision,
                "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
                "lane_deviation": float(np.mean(lane_devs)) if lane_devs else 0.0,
            }
        )
        logger.info(
            "  reward=%.2f  steps=%d  collision=%s  speed=%.2f  lane=%.3f",
            ep_reward, step, collision, results[-1]["avg_speed"], results[-1]["lane_deviation"],
        )

    return results


def _print_comparison(rl_metrics: Dict, baseline_metrics: Dict, rl_label: str) -> None:
    print("\n" + "=" * 60)
    print(f"{'BENCHMARK RESULTS':^60}")
    print("=" * 60)
    print(f"{'Metric':<25} {rl_label:>15} {'Baseline':>15}")
    print("-" * 60)

    keys = [
        ("avg_reward",       "Avg Reward"),
        ("collision_rate",   "Collision Rate"),
        ("avg_speed",        "Avg Speed (m/s)"),
        ("avg_lane_dev",     "Avg Lane Dev (m)"),
        ("success_rate",     "Success Rate"),
    ]

    for key, label in keys:
        rl_val = rl_metrics.get(key, 0.0)
        bl_val = baseline_metrics.get(key, 0.0)
        winner = "◀" if _better(key, rl_val, bl_val) else "▶"
        print(f"  {label:<23} {rl_val:>15.3f} {bl_val:>15.3f}  {winner}")

    print("=" * 60)


def _better(metric: str, rl: float, baseline: float) -> bool:
    """Return True if RL result is better than baseline for this metric."""
    lower_is_better = {"collision_rate", "avg_lane_dev"}
    if metric in lower_is_better:
        return rl <= baseline
    return rl >= baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RL agent vs TransFuser baseline.")
    parser.add_argument("--rl-algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--rl-model", type=str, default=None, help="Path to RL model checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    config = Config(args.config) if args.config else Config()
    seeds = [args.seed + i for i in range(args.episodes)]

    # ------------------------------------------------------------------ RL agent
    logger.info("=== Evaluating RL agent (%s) ===", args.rl_algorithm.upper())
    rl_trainer = Trainer(config, algorithm=args.rl_algorithm)

    rl_model_path = Path(args.rl_model) if args.rl_model else Path(
        config["training"]["model_path"]
    ).with_name(f"{args.rl_algorithm}_agent.pth")

    if rl_model_path.exists():
        rl_trainer.agent.load(str(rl_model_path))
        logger.info("Loaded RL model from %s", rl_model_path)
    else:
        logger.warning("RL model not found at %s — using random weights.", rl_model_path)

    rl_results = _run_agent(rl_trainer, rl_trainer.agent, args.episodes, seeds)
    rl_metrics = compute_metrics(rl_results)

    # ------------------------------------------------------------------ Baseline agent
    logger.info("=== Evaluating Baseline (%s) ===", config["baseline"]["model_type"])
    baseline_agent = BaselineAgent(config._config)
    baseline_trainer = Trainer(config, algorithm="ppo")  # reuse env, swap agent
    baseline_results = _run_agent(baseline_trainer, baseline_agent, args.episodes, seeds)
    baseline_metrics = compute_metrics(baseline_results)

    # ------------------------------------------------------------------ Report
    _print_comparison(rl_metrics, baseline_metrics, rl_label=args.rl_algorithm.upper())

    report = {
        "config": {
            "rl_algorithm": args.rl_algorithm,
            "rl_model": str(rl_model_path),
            "episodes": args.episodes,
            "seed": args.seed,
        },
        "rl_metrics": rl_metrics,
        "baseline_metrics": baseline_metrics,
        "rl_episodes": rl_results,
        "baseline_episodes": baseline_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()

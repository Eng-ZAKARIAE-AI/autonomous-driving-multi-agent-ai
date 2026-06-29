"""Visualization helpers for training and evaluation metrics."""

import matplotlib.pyplot as plt
from typing import Dict, List


def plot_training_history(path: str, history: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(12, 8))
    plt.plot(history['episode'], history['reward'], label='Reward')
    plt.plot(history['episode'], history['avg_speed'], label='Avg Speed')
    plt.plot(history['episode'], history['lane_deviation'], label='Lane Deviation')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_comparison(metrics: Dict[str, float], path: str) -> None:
    keys = list(metrics.keys())
    values = [metrics[k] for k in keys]
    plt.figure(figsize=(10, 5))
    plt.bar(keys, values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Evaluation Metrics')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

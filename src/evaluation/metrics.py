"""Evaluation metrics for autonomous driving episodes."""

from typing import Dict, List, Any


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            'collision_rate': 0.0,
            'average_reward': 0.0,
            'average_steps': 0.0,
            'average_speed': 0.0,
            'average_lane_deviation': 0.0,
            'success_rate': 0.0
        }

    collision_rate = sum(1 for item in results if item['collision']) / len(results)
    average_reward = sum(item['reward'] for item in results) / len(results)
    average_steps = sum(item['steps'] for item in results) / len(results)
    average_speed = sum(item['avg_speed'] for item in results) / len(results)
    average_lane_deviation = sum(item['lane_deviation'] for item in results) / len(results)
    success_rate = sum(1 for item in results if not item['collision']) / len(results)

    return {
        'collision_rate': float(collision_rate),
        'average_reward': float(average_reward),
        'average_steps': float(average_steps),
        'average_speed': float(average_speed),
        'average_lane_deviation': float(average_lane_deviation),
        'success_rate': float(success_rate)
    }

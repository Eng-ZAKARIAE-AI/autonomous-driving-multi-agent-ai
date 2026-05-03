"""Training loop for the CARLA multi-agent RL system."""

import time
from pathlib import Path
from typing import Optional

from .config import config
from .system import MultiAgentSystem


def train_model(episodes: Optional[int] = None, save_every: Optional[int] = None) -> Path:
    model_path = Path(config['training']['model_path'])
    episodes = int(episodes or config['training']['episodes'])
    save_every = int(save_every or config['training']['save_every'])

    system = MultiAgentSystem(training=True)
    print('🚀 Starting CARLA training session...')
    if not system.initialize():
        raise RuntimeError('CARLA environment initialization failed. Make sure the server is running.')

    best_reward = float('-inf')
    for episode in range(1, episodes + 1):
        state = system.reset()
        episode_reward = 0.0
        step = 0
        start_time = time.time()

        while True:
            decision = system.decision.select_action(state)
            action = {
                'steer': float(decision['action'][0]),
                'speed_adjust': float(decision['action'][1])
            }

            next_state, reward, done, info = system.step(action)
            system.decision.store_transition(state, decision['action'], decision['log_prob'], decision['value'], reward, done)

            episode_reward += reward
            state = next_state
            step += 1

            if done:
                next_value = 0.0
                system.decision.update(next_value)
                break

        elapsed = time.time() - start_time
        best_reward = max(best_reward, episode_reward)

        print(f"Episode {episode}/{episodes} | reward={episode_reward:.2f} | steps={step} | best={best_reward:.2f} | time={elapsed:.1f}s")

        if episode % save_every == 0 or episode == episodes:
            system.decision.save(str(model_path))
            print(f"✅ Saved model checkpoint to {model_path}")

    system.close()
    return model_path

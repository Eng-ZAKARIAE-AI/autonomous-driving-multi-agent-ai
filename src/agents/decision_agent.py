"""Decision agent that implements PPO for continuous control."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'mean': self.actor(x),
            'value': self.critic(x).squeeze(-1)
        }


class PPOAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = int(self.config['agents']['decision']['state_dim'])
        self.action_dim = int(self.config['agents']['decision']['action_dim'])
        self.model = ActorCritic(self.state_dim, self.action_dim, int(self.config['agents']['decision']['hidden_dim'])).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.config['training']['learning_rate']))

        self.action_std = float(self.config['training']['action_std'])
        self.gamma = float(self.config['training']['gamma'])
        self.gae_lambda = float(self.config['training']['gae_lambda'])
        self.clip_epsilon = float(self.config['training']['clip_epsilon'])
        self.ppo_epochs = int(self.config['training']['ppo_epochs'])
        self.batch_size = int(self.config['training']['batch_size'])
        self.value_coef = 0.5
        self.entropy_coef = 0.01

        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []

        self.model_path = Path(self.config['training']['model_path'])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.exists():
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"✅ Loaded existing PPO model from {self.model_path}")
            except Exception as exc:
                print(f"⚠️ Failed to load existing PPO model: {exc}")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Dict[str, Any]:
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        output = self.model(state_tensor)
        mean = output['mean']
        dist = torch.distributions.Normal(mean, self.action_std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = output['value']

        return {
            'action': action.squeeze(0).detach().cpu().numpy().astype(np.float32),
            'log_prob': log_prob.item(),
            'value': value.item()
        }

    def store_transition(self, state: np.ndarray, action: np.ndarray, log_prob: float, value: float, reward: float, done: bool) -> None:
        self.states.append(torch.FloatTensor(state).to(self.device))
        self.actions.append(torch.FloatTensor(action).to(self.device))
        self.log_probs.append(torch.FloatTensor([log_prob]).to(self.device))
        self.values.append(torch.FloatTensor([value]).to(self.device))
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def _compute_returns_and_advantages(self, next_value: float) -> Dict[str, torch.Tensor]:
        returns: List[float] = []
        advantages: List[float] = []
        gae = 0.0
        values = [v.item() for v in self.values] + [next_value]

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1.0 - self.dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[step]) * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)

        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        return {'returns': returns_tensor, 'advantages': advantages_tensor}

    def update(self, next_value: float = 0.0) -> None:
        if len(self.states) == 0:
            return

        data = self._compute_returns_and_advantages(next_value)
        returns = data['returns']
        advantages = data['advantages']

        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.cat(self.log_probs).view(-1)

        dataset_size = states.size(0)
        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                indices = permutation[start:end]

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]
                batch_old_log_probs = old_log_probs[indices]

                output = self.model(batch_states)
                mean = output['mean']
                values = output['value']
                dist = torch.distributions.Normal(mean, self.action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()
                entropy_loss = -self.entropy_coef * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def save(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(target))

    def load(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.model_path
        if target.exists():
            self.model.load_state_dict(torch.load(str(target), map_location=self.device))


class DecisionAgent:
    def __init__(self, config: Dict[str, Any], training: bool = True):
        self.ppo = PPOAgent(config)
        self.training = training

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Dict[str, Any]:
        return self.ppo.select_action(state, deterministic=deterministic)

    def store_transition(self, state: np.ndarray, action: np.ndarray, log_prob: float, value: float, reward: float, done: bool) -> None:
        if self.training:
            self.ppo.store_transition(state, action, log_prob, value, reward, done)

    def update(self, next_value: float = 0.0) -> None:
        if self.training:
            self.ppo.update(next_value)

    def save(self, path: Optional[str] = None) -> None:
        self.ppo.save(path)

    def load(self, path: Optional[str] = None) -> None:
        self.ppo.load(path)

"""PPO implementation for continuous control in CARLA."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.rl.common import ImageEncoder, VectorEncoder


# =========================================================
# 🔧 INIT WEIGHTS
# =========================================================
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)


# =========================================================
# 🧠 ACTOR-CRITIC NETWORK
# =========================================================
class PPOActorCritic(nn.Module):
    def __init__(self, image_channels: int, vector_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.image_encoder = ImageEncoder(input_channels=image_channels, feature_dim=128)
        self.vector_encoder = VectorEncoder(input_dim=vector_dim, feature_dim=64)

        self.shared = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # 🔥 IMPORTANT
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.apply(weights_init_)

    def forward(self, image: torch.Tensor, vector: torch.Tensor):
        image_features = self.image_encoder(image)
        vector_features = self.vector_encoder(vector)

        features = torch.cat([image_features, vector_features], dim=-1)
        shared = self.shared(features)

        mean = self.actor_mean(shared)
        value = self.critic(shared).squeeze(-1)

        return mean, value


# =========================================================
# 🎯 PPO AGENT
# =========================================================
class PPOAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_channels = int(config['camera']['channels'])
        self.vector_dim = int(config['state']['vector_dim'])
        self.action_dim = int(config['agents']['decision']['action_dim'])

        self.model = PPOActorCritic(
            self.image_channels,
            self.vector_dim,
            self.action_dim,
            hidden_dim=int(config['agents']['decision']['hidden_dim'])
        ).to(self.device)

        self.action_std = float(config['training'].get('action_std', 0.5))
        self.model.log_std.data.fill_(np.log(self.action_std))

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(config['training']['learning_rate'])
        )

        self.clip_eps = float(config['training']['clip_epsilon'])
        self.ppo_epochs = int(config['training']['ppo_epochs'])
        self.batch_size = int(config['training']['batch_size'])
        self.gamma = float(config['training']['gamma'])
        self.gae_lambda = float(config['training']['gae_lambda'])

        # buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

        self.model_path = Path(config['training']['model_path'])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # ⚠️ IMPORTANT: ignore old incompatible models
        self._load_model()

    # =====================================================
    # 📥 LOAD MODEL
    # =====================================================
    def _load_model(self):
        if self.model_path.exists():
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                print(f"✅ Loaded PPO model")
            except Exception:
                print("⚠️ Ignoring incompatible checkpoint")

    # =====================================================
    # 🎯 ACTION
    # =====================================================
    def select_action(self, state: Dict[str, np.ndarray], deterministic=False):

        image = torch.FloatTensor(state['image']).to(self.device).unsqueeze(0)
        vector = torch.FloatTensor(state['vector']).to(self.device).unsqueeze(0)

        mean, value = self.model(image, vector)

        # ✅ FIX HERE
        std = self.model.log_std.exp().expand_as(mean)

        dist = Normal(mean, std)

        if deterministic:
            z = mean
            action = torch.tanh(z)
        else:
            z = dist.rsample()
            action = torch.tanh(z)

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return {
            "action": action.squeeze(0).cpu().detach().numpy(),
            "log_prob": log_prob.item(),
            "value": value.item()
        }

    # =====================================================
    # 💾 STORE
    # =====================================================
    def store_transition(self, state, action, log_prob, value, reward, done):

        self.states.append({
            "image": torch.FloatTensor(state["image"]).to(self.device),
            "vector": torch.FloatTensor(state["vector"]).to(self.device)
        })

        self.actions.append(torch.FloatTensor(action).to(self.device))
        self.log_probs.append(torch.FloatTensor([log_prob]).to(self.device))
        self.values.append(torch.FloatTensor([value]).to(self.device))
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    # =====================================================
    # 📊 GAE
    # =====================================================
    def _compute_gae(self, next_value):

        returns = []
        advantages = []

        gae = 0
        values = [v.item() for v in self.values] + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae

            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    # =====================================================
    # 🔁 UPDATE
    # =====================================================
    def update(self, next_value=0.0):

        if len(self.states) == 0:
            return

        returns, advantages = self._compute_gae(next_value)

        images = torch.stack([s["image"] for s in self.states])
        vectors = torch.stack([s["vector"] for s in self.states])
        actions = torch.stack(self.actions)

        old_log_probs = torch.cat(self.log_probs).view(-1)

        for _ in range(self.ppo_epochs):

            idx = torch.randperm(images.size(0))

            for start in range(0, len(idx), self.batch_size):

                batch = idx[start:start+self.batch_size]

                mean, values = self.model(images[batch], vectors[batch])

                std = self.model.log_std.exp().expand_as(mean)
                dist = Normal(mean, std)

                raw_actions = torch.atanh(actions[batch].clamp(-0.999, 0.999))
                new_log_probs = dist.log_prob(raw_actions).sum(dim=-1) - torch.log(1 - actions[batch].pow(2) + 1e-6).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                ratio = torch.exp(new_log_probs - old_log_probs[batch])

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[batch]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[batch])
                entropy_loss = -0.01 * entropy.mean()

                loss = policy_loss + 0.5 * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        # reset buffer
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    # =====================================================
    # 💾 LOAD
    # =====================================================
    def load(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.model_path
        if target.exists():
            self.model.load_state_dict(torch.load(str(target), map_location=self.device))
            print(f"✅ Loaded PPO model from {target}")

    # =====================================================
    # 💾 SAVE
    # =====================================================
    def save(self, path=None):
        path = Path(path) if path else self.model_path
        torch.save(self.model.state_dict(), str(path))
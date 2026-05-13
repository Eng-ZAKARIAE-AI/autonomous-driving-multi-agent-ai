"""Soft Actor-Critic implementation for CARLA."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.src.agents.common import ImageEncoder, VectorEncoder


class QNetwork(nn.Module):
    def __init__(self, image_channels: int, vector_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.image_encoder = ImageEncoder(input_channels=image_channels, feature_dim=128)
        self.vector_encoder = VectorEncoder(input_dim=vector_dim, feature_dim=64)
        self.net = nn.Sequential(
            nn.Linear(128 + 64 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image: torch.Tensor, vector: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        vector_features = self.vector_encoder(vector)
        x = torch.cat([image_features, vector_features, action], dim=-1)
        return self.net(x).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, image_channels: int, vector_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.image_encoder = ImageEncoder(input_channels=image_channels, feature_dim=128)
        self.vector_encoder = VectorEncoder(input_dim=vector_dim, feature_dim=64)
        self.net = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, image: torch.Tensor, vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_features = self.image_encoder(image)
        vector_features = self.vector_encoder(vector)
        x = torch.cat([image_features, vector_features], dim=-1)
        x = self.net(x)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return {'mean': mean, 'log_std': log_std}

    def sample(self, image: torch.Tensor, vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        params = self.forward(image, vector)
        mean = params['mean']
        std = torch.exp(params['log_std'])
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return {'action': action, 'log_prob': log_prob, 'mean': mean}


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.memory: List[Dict[str, Any]] = []
        self.position = 0

    def push(self, transition: Dict[str, Any]) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, Any]:
        batch = np.random.choice(self.memory, batch_size, replace=False)
        images = torch.stack([item['state']['image'] for item in batch])
        vectors = torch.stack([item['state']['vector'] for item in batch])
        actions = torch.stack([item['action'] for item in batch])
        rewards = torch.FloatTensor([item['reward'] for item in batch]).unsqueeze(-1)
        next_images = torch.stack([item['next_state']['image'] for item in batch])
        next_vectors = torch.stack([item['next_state']['vector'] for item in batch])
        dones = torch.FloatTensor([item['done'] for item in batch]).unsqueeze(-1)
        return {
            'images': images,
            'vectors': vectors,
            'actions': actions,
            'rewards': rewards,
            'next_images': next_images,
            'next_vectors': next_vectors,
            'dones': dones
        }

    def __len__(self) -> int:
        return len(self.memory)


class SACAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_channels = int(config['camera']['channels'])
        self.vector_dim = int(config['state']['vector_dim'])
        self.action_dim = int(config['agents']['decision']['action_dim'])

        self.policy = GaussianPolicy(self.image_channels, self.vector_dim, self.action_dim,
                                     hidden_dim=int(config['agents']['decision']['hidden_dim'])).to(self.device)
        self.q1 = QNetwork(self.image_channels, self.vector_dim, self.action_dim,
                           hidden_dim=int(config['agents']['decision']['hidden_dim'])).to(self.device)
        self.q2 = QNetwork(self.image_channels, self.vector_dim, self.action_dim,
                           hidden_dim=int(config['agents']['decision']['hidden_dim'])).to(self.device)
        self.q1_target = QNetwork(self.image_channels, self.vector_dim, self.action_dim,
                                  hidden_dim=int(config['agents']['decision']['hidden_dim'])).to(self.device)
        self.q2_target = QNetwork(self.image_channels, self.vector_dim, self.action_dim,
                                  hidden_dim=int(config['agents']['decision']['hidden_dim'])).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=float(config['training']['learning_rate']))
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=float(config['training']['learning_rate']))
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=float(config['training']['learning_rate']))
        self.alpha = float(config['training']['sac_alpha'])
        self.gamma = float(config['training']['gamma'])
        self.tau = float(config['training']['target_smoothing_tau'])
        self.batch_size = int(config['training']['batch_size'])
        self.replay_buffer = ReplayBuffer(capacity=int(config['training']['replay_size']))

        self.model_path = Path(config['training']['model_path'])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_model()

    def select_action(self, state: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        image = torch.FloatTensor(state['image']).to(self.device).unsqueeze(0)
        vector = torch.FloatTensor(state['vector']).to(self.device).unsqueeze(0)
        with torch.no_grad():
            sample = self.policy.sample(image, vector)
        action = sample['action']
        if deterministic:
            action = torch.tanh(sample['mean'])

        return {
            'action': action.squeeze(0).cpu().detach().numpy().astype(np.float32),
            'log_prob': sample['log_prob'].item() if not deterministic else 0.0
        }

    def store_transition(self, state: Dict[str, Any], action: np.ndarray, reward: float,
                         next_state: Dict[str, Any], done: bool) -> None:
        self.replay_buffer.push({
            'state': {
                'image': torch.FloatTensor(state['image']).to(self.device),
                'vector': torch.FloatTensor(state['vector']).to(self.device)
            },
            'action': torch.FloatTensor(action).to(self.device),
            'reward': float(reward),
            'next_state': {
                'image': torch.FloatTensor(next_state['image']).to(self.device),
                'vector': torch.FloatTensor(next_state['vector']).to(self.device)
            },
            'done': float(done)
        })

    def update(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        images = batch['images'].to(self.device)
        vectors = batch['vectors'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_images = batch['next_images'].to(self.device)
        next_vectors = batch['next_vectors'].to(self.device)
        dones = batch['dones'].to(self.device)

        with torch.no_grad():
            next_policy = self.policy.sample(next_images, next_vectors)
            next_actions = next_policy['action']
            next_log_prob = next_policy['log_prob'].unsqueeze(-1)
            target_q1 = self.q1_target(next_images, next_vectors, next_actions)
            target_q2 = self.q2_target(next_images, next_vectors, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            q_target = rewards + self.gamma * (1 - dones) * target_q

        q1_loss = F.mse_loss(self.q1(images, vectors, actions).unsqueeze(-1), q_target)
        q2_loss = F.mse_loss(self.q2(images, vectors, actions).unsqueeze(-1), q_target)

        self.q1_optimizer.zero_grad(); q1_loss.backward(); self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(); q2_loss.backward(); self.q2_optimizer.step()

        policy_output = self.policy.sample(images, vectors)
        policy_actions = policy_output['action']
        log_prob = policy_output['log_prob'].unsqueeze(-1)
        q1_pi = self.q1(images, vectors, policy_actions)
        q2_pi = self.q2(images, vectors, policy_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * log_prob - q_pi.unsqueeze(-1)).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _load_model(self) -> None:
        if self.model_path.exists():
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy'])
            self.q1.load_state_dict(checkpoint['q1'])
            self.q2.load_state_dict(checkpoint['q2'])
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            print(f"✅ Loaded SAC model from {self.model_path}")

    def save(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.model_path
        torch.save({'policy': self.policy.state_dict(),
                    'q1': self.q1.state_dict(),
                    'q2': self.q2.state_dict()}, str(target))

    def load(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.model_path
        if target.exists():
            checkpoint = torch.load(str(target), map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy'])
            self.q1.load_state_dict(checkpoint['q1'])
            self.q2.load_state_dict(checkpoint['q2'])
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())

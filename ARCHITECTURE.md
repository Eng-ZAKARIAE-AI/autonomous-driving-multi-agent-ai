# Production-Ready Autonomous Driving System Architecture

## Executive Summary

This document outlines the architecture of our autonomous driving platform. We have transitioned from a basic academic setup to a modular, multi-agent system featuring multi-modal perception, advanced reinforcement learning (SAC/PPO), and a robust telemetry pipeline.

---

## 1. SYSTEM ARCHITECTURE

### 1.1 Current Architecture (Modular & Multi-Modal)

The system is designed with clear separation between perception, decision-making, and simulation.

```
┌─────────────────────────────────────────────────────────┐
│           PERCEPTION LAYER (Implemented)                │
├─────────────────────────────────────────────────────────┤
│  • RGB Camera Processing (84x84 Grayscale/Color)       │
│  • Vector State Extraction (Speed, GPS, IMU)           │
│  • Multi-Modal Encoding (CNN + MLP Fusion)             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│        DECISION MAKING LAYER (Implemented)              │
├─────────────────────────────────────────────────────────┤
│  • RL Agents: SAC (Soft Actor-Critic) & PPO            │
│  • Continuous Action Space: [steer, throttle, brake]   │
│  • Reward Engineering (Safety, Efficiency, Comfort)    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│      SIMULATION & TELEMETRY (Implemented)               │
├─────────────────────────────────────────────────────────┤
│  • CARLA Simulator Integration                         │
│  • WebSocket Telemetry Server (FastAPI)                │
│  • Real-time Monitoring & Dashboard                    │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Project Structure

```
backend/src/
├── agents/
│   ├── common.py           # Shared encoders (ImageEncoder, VectorEncoder)
│   ├── ppo/                # PPO implementation
│   └── sac/                # SAC implementation (Implemented Improvement)
├── perception/
│   └── camera.py           # Image preprocessing & normalization
├── reward/
│   └── reward_function.py  # Multi-objective weighted reward
├── training/
│   └── trainer.py          # Unified training & evaluation pipeline
├── visualization/
│   └── plots.py            # Training metrics visualization
└── multi_agent_main.py     # Root entrypoint with Telemetry Server
```

---

## 2. REINFORCEMENT LEARNING ALGORITHMS

We support both **PPO** and **SAC** for continuous control. While PPO provides stability, **SAC** is our primary recommendation for production due to its superior sample efficiency and automatic entropy regularization.

### 2.1 Implemented: SAC (Soft Actor-Critic)

The implementation utilizes dual Q-networks to mitigate overestimation bias and an entropy-regularized policy for better exploration.

```python
# Simplified implementation overview (see sac_agent.py for details)
class SACAgent:
    def __init__(self, config):
        self.policy = GaussianPolicy(...) # Actor
        self.q1 = QNetwork(...)           # Critic 1
        self.q2 = QNetwork(...)           # Critic 2
        self.replay_buffer = ReplayBuffer(capacity=100_000)
        
    def update(self):
        # Off-policy learning from Replay Buffer
        batch = self.replay_buffer.sample(self.batch_size)
        # Update Critics via Bellman Error
        # Update Actor via Reparameterization Trick
        # Soft-update target networks
```

### 2.2 Implemented: PPO (Proximal Policy Optimization)

PPO is used as a robust baseline, featuring GAE (Generalized Advantage Estimation) and a clipped objective function.

---

## 3. MULTI-MODAL STATE REPRESENTATION

Instead of relying solely on vector data, our agents "see" the environment through a fused representation.

### 3.1 State Fusion (Implemented)

We combine raw pixel data with internal vehicle state to provide high-dimensional context.

```python
# From backend/src/agents/common.py
class MultiModalEncoder(nn.Module):
    def __init__(self):
        self.image_encoder = ImageEncoder()  # CNN for 84x84 images
        self.vector_encoder = VectorEncoder() # MLP for speed, offset, etc.
        
    def forward(self, image, vector):
        img_feats = self.image_encoder(image)     # 128D
        vec_feats = self.vector_encoder(vector)   # 64D
        return torch.cat([img_feats, vec_feats], dim=-1) # 192D Unified State
```

---

## 4. ROBUST REWARD FUNCTION

Our `RewardFunction` is designed to balance multiple competing objectives, ensuring the agent drives safely, efficiently, and comfortably.

### 4.1 Weighted Multi-Objective Design (Implemented)

```python
reward = (
    weights['speed'] * speed_reward +      # Match target speed
    weights['lane'] * lane_reward +        # Stay centered
    weights['progress'] * progress_reward + # Move towards goal
    weights['comfort'] * comfort_penalty +  # Minimize jerk/sudden actions
    collision_penalty +                    # Large penalty on crash
    success_reward                         # Bonus for reaching destination
)
```

---

## 5. TELEMETRY & MONITORING

A unique feature of this architecture is the **real-time telemetry server**.

- **FastAPI + WebSockets:** Broadcasts live vehicle stats (speed, lane offset, rewards) to a frontend dashboard.
- **Asynchronous Loop:** The AI training runs in a background thread while the telemetry server handles concurrent web client connections.

---

## 6. ROADMAP & FUTURE INNOVATIONS

While the core architecture is solid, the following modules are planned for the next phases:

### Phase 1: Advanced Perception (Planned)
- **Object Detection:** Integrate YOLOv8 for explicit obstacle detection.
- **Semantic Segmentation:** Use lane masks for more precise lateral control.

### Phase 2: Safety Verification (Planned)
- **Control Barrier Functions (CBF):** A safety layer that overrides RL actions if they would lead to a guaranteed collision.
- **Emergency Braking:** Rule-based fallback for immediate hazards.

### Phase 3: Hardware-in-Loop
- **ROS Bridge:** Adapting the `trainer.py` logic to interface with Robot Operating System (ROS) for real-world deployment.

---

## 7. KEY INNOVATIONS FOR PFA

1. **Multi-Modal Fusion:** Combining vision and kinematics for superior spatial awareness.
2. **Entropy-Regularized SAC:** Solving the exploration-exploitation trade-off automatically.
3. **Real-Time Telemetry:** Bridging the gap between "black-box" RL training and human-interpretable monitoring.
4. **Modular Reward Design:** Explicitly penalizing "uncomfortable" driving (jerk) to mimic human-like behavior.

---

*This architecture document is updated to reflect the current implementation as of May 2026.*

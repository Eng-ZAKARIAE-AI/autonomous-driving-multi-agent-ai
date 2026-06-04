# Slide 5: AI Architecture

## Brain of the System: Multi-Modal RL

### 1. State Fusion (MultiModalEncoder)
- **Visual Encoder:** 3-layer CNN for processing 84x84 RGB images.
- **Kinematic Encoder:** 2-layer MLP for speed, lane offset, and navigation data.
- **Fusion:** Concatenation into a 192D unified state vector.

### 2. Decision Logic (SAC/PPO)
- **SAC (Primary):** Off-policy algorithm with entropy regularization for robust exploration and sample efficiency.
- **Dual Q-Networks:** Mitigates overestimation bias in value estimation.

### 3. Reward Function
- Multi-objective weighted sum:
  - `Speed Reward`: Matching target velocity.
  - `Lane Penalty`: Keeping centered in the lane.
  - `Comfort Penalty`: Reducing sudden steering/acceleration (Jerk).
  - `Collision Penalty`: Large negative reward for safety.

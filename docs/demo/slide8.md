# Slide 8: Challenges

## Technical Hurdles & Solutions

1.  **Sim-to-Real Gap / Convergence:** RL agents require significant training time to stabilize. *Solution:* Using SAC with entropy tuning to improve exploration.
2.  **Environment Synchronization:** Maintaining consistent FPS between the CARLA simulator and the Python AI loop. *Solution:* Implementation of synchronous mode in `CarlaGymEnv`.
3.  **Hardware Constraints:** Deep Learning and high-fidelity simulation require significant GPU memory. *Solution:* Containerization with NVIDIA-Docker to optimize resource allocation.
4.  **Reward Engineering:** Balancing speed vs. safety. *Solution:* Iterative weight tuning and introducing "comfort" penalties for smoother driving.

# 4. AI Pipeline Diagram

```mermaid
graph LR
    subgraph "Input Modalities"
        IMG[RGB Camera 84x84]
        VEC[Kinematics: Speed, Offset, GPS]
    end

    subgraph "State Representation (MultiModalEncoder)"
        CNN[Image Encoder (CNN)]
        MLP[Vector Encoder (MLP)]
        Fusion[Concat / Fusion Layer]
    end

    subgraph "Decision Engine (Actor-Critic)"
        Policy[Gaussian Policy (Actor)]
        Value[Q-Networks (Critics)]
    end

    subgraph "Reward Engineering"
        RF[Reward Function]
        Weights[Speed, Lane, Progress, Comfort]
    end

    IMG --> CNN
    VEC --> MLP
    CNN --> Fusion
    MLP --> Fusion
    Fusion --> Policy
    Fusion --> Value
    Policy --> Action[Control: [s, t, b]]
    Action --> RF
    RF --> Weights
    Weights --> ScalarReward[Total Reward]
```

    # 1. Product Overview

## Product Concept

The **Autonomous Driving Multi-Agent AI Dashboard** is a high-performance, real-time monitoring and control interface for training, evaluating, and deploying autonomous driving agents within the CARLA simulator. 

The interface is designed to bridge the gap between complex reinforcement learning (RL) backends and human-interpretable insights. It provides researchers and engineers with a "mission control" experience to observe agent behavior, analyze telemetry, and manage the training lifecycle with precision.

## Target Users

*   **AI Researchers:** Monitoring RL training stability, reward convergence, and agent behavior.
*   **Autonomous Driving Engineers:** Validating perception and control logic in simulated environments.
*   **Data Scientists:** Analyzing telemetry logs and performance metrics (Speed, Lane Offset, Collisions).
*   **Students/PFA Candidates:** Managing the project workflow and documenting results.

## Core Product Experience

Users should be able to:

*   **Visualize Live Telemetry:** Monitor speed, throttle, steer, brake, and reward in real-time via WebSockets.
*   **Monitor Vision:** View processed camera feeds (84x84 grayscale/color) used by the agents.
*   **Track Training Progress:** Observe live graphs for cumulative rewards, episode lengths, and algorithm-specific losses.
*   **Manage Agents:** Switch between PPO and SAC configurations and trigger training/inference modes.
*   **Audit Safety:** Track collision events, lane departures, and "jerk" (comfort) metrics.
*   **Environment Control:** View and potentially adjust simulation parameters (weather, traffic density).

## Product Personality

The product should feel:

*   **Technical & Precise:** High-density data visualization without clutter.
*   **Real-Time & Responsive:** Low-latency updates reflecting the simulation state.
*   **Reliable:** Industrial-grade aesthetic that builds trust in the data.
*   **Modern & Dark-Mode Optimized:** Reducing eye strain during long training sessions.
*   **Focused:** Bringing the most critical "driving" metrics to the forefront.

---

# 2. Design Principles

## Live-First Data

Since the core value is monitoring a running simulation, the UI must prioritize real-time data flow. Use optimized charting libraries and WebSocket-ready components.

## Spatial Awareness

The dashboard should help the user understand where the agent is and what it "sees". Integrate mini-maps or coordinate displays alongside camera feeds.

## Meaningful Metrics

Don't just show numbers; show trends. Use color-coded indicators for safety (Red for collisions, Green for lane centering).

## High-Density Clarity

Autonomous driving generates a lot of data. Use a grid-based dashboard that allows scanning multiple agents or sensors simultaneously without feeling overwhelmed.

## Error & Status Transparency

Clearly distinguish between "Simulator Disconnected," "Agent Crashed," and "Training Paused" states.

---

# 3. User Experience Goals

## Primary Goals

*   **Real-Time Monitoring:** Zero-lag visualization of the agent's current state.
*   **Fast Debugging:** Quickly identify why an agent failed (e.g., reward drop or collision).
*   **Workflow Efficiency:** Seamlessly move from configuring a training run to observing it.
*   **Performance Insight:** Make it easy to compare PPO vs. SAC performance over time.

## Secondary Goals

*   **Historical Analysis:** Review past training sessions and checkpoint performance.
*   **Exportable Insights:** Generate reports or screenshots for academic documentation (PFA rapport).
*   **Mobile-Friendly Alerts:** Check training status from a phone with a simplified "status-at-a-glance" view.

---

# 4. Information Architecture

## Global Structure

The application uses a persistent side navigation to switch between different views of the AI lifecycle.

1.  **Dashboard (Live):** The "Mission Control" center for active simulations.
2.  **Training:** Real-time and historical training metrics (Tensorboard-style).
3.  **Agents:** Library of trained models and active configurations.
4.  **Simulation:** Environment settings, CARLA connection status, and map selection.
5.  **Telemetry Logs:** Raw data stream and export options.
6.  **Settings:** API endpoints, UI preferences, and authentication.

## Navigation Model

*   **Desktop:** Left sidebar for main navigation + Top header for global status (Simulator Heartbeat, Backend Link).
*   **Mobile:** Bottom tab bar for key monitoring views.

## Desktop Layout Structure

```md
App Shell
├── Sidebar Navigation (Dashboard, Training, Agents, Sim, Logs)
├── Top Header
│   ├── Project Title: "AD-MAI"
│   ├── Connection Status (CARLA: OK | Backend: OK)
│   ├── Global Search (Agents/Logs)
│   └── User Profile
└── Main Content Area
    ├── Dashboard Grid (Camera, Gauges, Mini-map)
    └── Contextual Action Sidebar (Start/Stop Train, Save Checkpoint)
```

---

# 5. Page-by-Page UI Specifications

## 5.1 Dashboard (Live View)

### Purpose
The primary interface for watching the agent drive in real-time.

### Components
*   **Live Camera Feed:** Displaying the 84x84 multi-modal input.
*   **Telemetry Gauges:** Speedometer (circular), Throttle/Brake/Steer (linear bars).
*   **Reward Tracker:** Moving line chart of the last 100 reward steps.
*   **Safety Panel:** Collision indicator (flashing red on hit), Lane offset meter.
*   **Agent Status:** Current Algorithm (SAC/PPO), Mode (Train/Infer), Episode count.

## 5.2 Training Metrics

### Purpose
Analyzing the "brain" development of the agent.

### Components
*   **Convergence Charts:** Cumulative Reward per Episode, Average Speed.
*   **Loss Charts:** Policy Loss, Value Loss, Entropy (for SAC).
*   **Performance Heatmaps:** Where on the map the agent usually fails.
*   **Episode History:** Table of recent episodes with duration and final state (Success/Crash).

## 5.3 Agent Management

### Purpose
Configuring and selecting models.

### Components
*   **Model List:** Grid of saved checkpoints with metadata (Algorithm, Date, Total Timesteps).
*   **Config Editor:** YAML/JSON editor for hyper-parameters (Learning Rate, Gamma, Batch Size).
*   **Architecture View:** Visual representation of the Multi-Modal Encoder (CNN + MLP).

## 5.4 Simulation Config

### Purpose
Controlling the CARLA environment.

### Components
*   **Map Selector:** Visual thumbnails of available CARLA towns.
*   **Weather Presets:** Clear, Rainy, Foggy, Night.
*   **Traffic Density:** Sliders for number of vehicles and pedestrians.
*   **Sync Mode:** Toggle for synchronous vs. asynchronous simulation.

---

# 6. Component System

## Telemetry Gauges
*   **Speedometer:** High-contrast radial gauge with target speed indicator.
*   **Action Bars:** Horizontal bars showing normalized agent output [-1, 1] for steer and [0, 1] for throttle/brake.

## Status Badges
*   **Algorithm:** `SAC` (Purple), `PPO` (Blue).
*   **State:** `TRAINING` (Pulsing Green), `IDLE` (Gray), `EVALUATING` (Amber).
*   **Health:** `HEALTHY` (Checkmark), `ERROR` (Cross).

## Metric Cards
*   Small, focused cards showing a single number + trend (e.g., "Avg Reward: +15%").

---

# 7. Design Tokens

## Color Palette (Tech-Focused)

### Light/Dark Mode
*   **Primary Accent:** `#6366F1` (Indigo - Represents the AI).
*   **Safety Red:** `#EF4444` (Collision/Emergency).
*   **Safety Green:** `#10B981` (Target reached/Healthy).
*   **Telemetry Orange:** `#F59E0B` (Control actions).

### Dark Mode (Default Recommendation)
*   **Background:** `#0F172A` (Deep Slate).
*   **Surface:** `#1E293B`.
*   **Gauges Background:** `#334155`.

---

# 8. Interaction Patterns

## Real-Time Toggle
A global "Live" switch to pause UI updates if the user wants to inspect a static state without stopping the backend.

## Interactive Charts
Hovering over a training point should show the specific checkpoint or video clip (if recorded) associated with that episode.

## Command Console
A small, collapsible terminal window at the bottom for raw backend logs and manual command input.

---

# 9. Technical Implementation (Frontend)

*   **Framework:** React 18+ with TypeScript.
*   **State:** Redux Toolkit or React Query for managing telemetry streams.
*   **Streaming:** `WebSockets` for telemetry, `MJPEG` or `WebRTC` for camera feeds.
*   **Charts:** `Recharts` or `Chart.js` (optimized for frequent updates).
*   **Icons:** `Lucide-React` for clean, geometric iconography.
*   **Styling:** `Tailwind CSS` for rapid, consistent layout.

---

# 10. PFA Specifics

*   **Rapport Assets:** Include a "Snapshot" button that captures the current dashboard state in a high-resolution, print-ready format for the final report.
*   **Architecture Mapping:** A dedicated section that maps the UI components to the Python classes in the backend (e.g., "Reward Function Gauge" -> `reward_function.py`).

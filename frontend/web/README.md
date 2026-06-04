# AD-MAI Frontend

High-performance real-time dashboard for Autonomous Driving Multi-Agent AI monitoring.

## 🚀 Features

- **Live Telemetry:** Real-time speed, lane offset, and collision monitoring.
- **Reward Tracking:** Live graph of agent rewards per step.
- **Control Interface:** Visualization of agent actions (steer/throttle).
- **Modern Dark UI:** Optimized for long monitoring sessions.

## 🛠️ Tech Stack

- **Framework:** React 19 + TypeScript
- **Styling:** Tailwind CSS v4
- **Icons:** Lucide React
- **Charts:** Recharts
- **Communication:** Native WebSockets

## 🏃 Running Locally

### 1. Install Dependencies
```bash
cd frontend/web
npm install
```

### 2. Start Development Server
```bash
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

### 3. Build for Production
```bash
npm run build
```

## 🔌 Backend Integration

The frontend connects to the telemetry server at `ws://localhost:8000/ws/telemetry`. Ensure the backend is running (see `backend/README.md`).

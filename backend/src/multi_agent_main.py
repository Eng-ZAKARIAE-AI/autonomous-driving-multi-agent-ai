"""Root entrypoint for the CARLA multi-agent RL project with WebSocket telemetry."""

import argparse
import asyncio
import json
import threading
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from backend.src.config import Config
from backend.src.training.trainer import Trainer

app = FastAPI(title="Autonomous Driving Telemetry")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()
latest_telemetry = {}

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/telemetry")
async def get_telemetry():
    return latest_telemetry

def run_ai_loop(args, config, model_path):
    global latest_telemetry
    trainer = Trainer(config, algorithm=args.algorithm)
    
    # Inject telemetry hook into trainer/env
    original_step = trainer.env.step
    def step_with_telemetry(action):
        obs, reward, done, info = original_step(action)
        # Update latest telemetry
        nonlocal latest_telemetry
        telemetry_data = {
            "speed": info.get("speed", 0),
            "lane_offset": info.get("lane_offset", 0),
            "collision": info.get("collision", False),
            "reward": reward,
            "action": action.tolist() if hasattr(action, 'tolist') else action
        }
        latest_telemetry = telemetry_data
        
        # Broadcast via websocket if possible
        try:
            # We need to run the broadcast in the event loop of the FastAPI app
            # For simplicity in this script, we just update the global latest_telemetry
            # and the websocket endpoint can pull it or we can use a callback.
            pass
        except Exception:
            pass
            
        return obs, reward, done, info
    
    trainer.env.step = step_with_telemetry

    if args.mode == 'train':
        trainer.train(episodes=args.episodes)
    elif args.mode == 'evaluate' or args.mode == 'infer':
        trainer.agent.load(str(model_path))
        deterministic = (args.mode == 'infer')
        trainer.evaluate(episodes=int(args.episodes or config['evaluation']['episodes']), deterministic=deterministic)
    else:
        # Auto mode
        if model_path.exists():
            trainer.agent.load(str(model_path))
            trainer.evaluate(episodes=int(args.episodes or config['evaluation']['episodes']), deterministic=True)
        else:
            trainer.train(episodes=args.episodes)

def parse_args():
    parser = argparse.ArgumentParser(description='Run or train the CARLA multi-agent RL system.')
    parser.add_argument('--mode', choices=['auto', 'train', 'infer', 'evaluate'], default='auto', help='Execution mode.')
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo', help='RL algorithm.')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes to train or infer.')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the model file.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    parser.add_argument('--port', type=int, default=8000, help='Port for the telemetry server.')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config(args.config) if args.config else Config()
    
    base_model_path = Path(config['training']['model_path'])
    if args.model_path is None and base_model_path.name in ('ppo_agent.pth', 'sac_agent.pth'):
        model_path = base_model_path.with_name(f'{args.algorithm}_agent.pth')
    else:
        model_path = Path(args.model_path or config['training']['model_path'])

    # Start AI loop in a separate thread
    ai_thread = threading.Thread(target=run_ai_loop, args=(args, config, model_path), daemon=True)
    ai_thread.start()

    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == '__main__':
    main()

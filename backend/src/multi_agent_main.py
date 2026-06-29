"""FastAPI server — telemetry WebSocket + REST API for training control.

Endpoints
---------
GET  /api/status              → system status
POST /api/training/start      → launch a training / evaluation session
POST /api/training/stop       → stop the current session
GET  /api/training/status     → current session progress
GET  /telemetry               → latest telemetry snapshot (REST)
WS   /ws/telemetry            → streaming telemetry WebSocket
"""

import argparse
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.src.config import Config
from backend.src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Autonomous Driving — AI Control API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_latest_telemetry: Dict[str, Any] = {}
_training_state: Dict[str, Any] = {
    "status": "idle",        # idle | training | evaluating | error
    "algorithm": None,
    "mode": None,
    "episode": 0,
    "total_episodes": 0,
    "reward": 0.0,
    "avg_speed": 0.0,
    "lane_deviation": 0.0,
    "collision": False,
    "error": None,
    "start_time": None,
    "elapsed": 0.0,
}
_stop_event = threading.Event()
_ai_thread: Optional[threading.Thread] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: str) -> None:
        for ws in list(self.active):
            try:
                await ws.send_text(message)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TrainingStartRequest(BaseModel):
    algorithm: str = "ppo"       # ppo | sac | baseline
    mode: str = "train"           # train | evaluate | infer
    episodes: int = 200
    model_path: Optional[str] = None
    config_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    global _event_loop
    _event_loop = asyncio.get_event_loop()


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/telemetry")
async def get_telemetry() -> Dict[str, Any]:
    return _latest_telemetry


@app.get("/api/status")
async def api_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "training": _training_state["status"],
        "algorithm": _training_state["algorithm"],
    }


@app.get("/api/training/status")
async def api_training_status() -> Dict[str, Any]:
    state = dict(_training_state)
    if state["start_time"]:
        state["elapsed"] = round(time.time() - state["start_time"], 1)
    return state


@app.post("/api/training/start")
async def api_training_start(req: TrainingStartRequest) -> Dict[str, Any]:
    global _ai_thread, _stop_event

    if _training_state["status"] in ("training", "evaluating"):
        return {"success": False, "message": "A session is already running. Stop it first."}

    _stop_event.clear()
    _training_state.update(
        status="training" if req.mode == "train" else "evaluating",
        algorithm=req.algorithm,
        mode=req.mode,
        episode=0,
        total_episodes=req.episodes,
        reward=0.0,
        error=None,
        start_time=time.time(),
    )

    _ai_thread = threading.Thread(
        target=_run_session,
        args=(req,),
        daemon=True,
    )
    _ai_thread.start()

    return {"success": True, "message": f"Session started: {req.algorithm} / {req.mode}"}


@app.post("/api/training/stop")
async def api_training_stop() -> Dict[str, Any]:
    _stop_event.set()
    _training_state["status"] = "idle"
    return {"success": True, "message": "Stop signal sent."}


# ---------------------------------------------------------------------------
# AI session runner (background thread)
# ---------------------------------------------------------------------------
def _run_session(req: TrainingStartRequest) -> None:
    global _latest_telemetry

    try:
        config = Config(req.config_path) if req.config_path else Config()
        cfg_dict = config._config

        trainer = Trainer(cfg_dict, algorithm=req.algorithm)

        # Patch env step to emit telemetry
        original_step = trainer.env.step

        def _step_with_telemetry(action):
            obs, reward, done, info = original_step(action)
            telemetry = {
                "speed": info.get("speed", 0.0),
                "lane_offset": info.get("lane_offset", 0.0),
                "collision": info.get("collision", False),
                "reward": float(reward),
                "action": action if isinstance(action, list) else list(action.values()),
                "episode": _training_state["episode"],
                "detections": info.get("detections", []),
            }
            _latest_telemetry = telemetry
            _training_state.update(
                reward=float(reward),
                avg_speed=info.get("speed", 0.0),
                collision=bool(info.get("collision", False)),
            )
            if _event_loop and not _event_loop.is_closed():
                try:
                    asyncio.run_coroutine_threadsafe(
                        manager.broadcast(json.dumps(telemetry, default=str)),
                        _event_loop,
                    )
                except Exception:
                    pass
            return obs, reward, done, info

        trainer.env.step = _step_with_telemetry

        # Patch train/evaluate to track episodes
        original_train = trainer.train
        original_evaluate = trainer.evaluate

        def _train_with_tracking(episodes=None):
            n = int(episodes or cfg_dict["training"]["episodes"])
            for ep in range(1, n + 1):
                if _stop_event.is_set():
                    break
                _training_state["episode"] = ep
                # delegate one episode by temporarily overriding episodes=1
            original_train(episodes=episodes)

        # Load model if needed
        if req.mode in ("evaluate", "infer") and req.algorithm != "baseline":
            model_path = Path(req.model_path) if req.model_path else Path(
                cfg_dict["training"]["model_path"]
            ).with_name(f"{req.algorithm}_agent.pth")
            if model_path.exists():
                trainer.agent.load(str(model_path))

        if req.mode == "train":
            trainer.train(episodes=req.episodes)
        else:
            trainer.evaluate(
                episodes=req.episodes,
                deterministic=(req.mode == "infer"),
            )

        _training_state["status"] = "idle"

    except Exception as exc:
        logger.error("Session error: %s", exc, exc_info=True)
        _training_state.update(status="error", error=str(exc))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Driving AI Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

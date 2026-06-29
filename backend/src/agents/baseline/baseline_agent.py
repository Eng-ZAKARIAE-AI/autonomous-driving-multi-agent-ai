"""BaselineAgent — wraps BaselineNet for inference in the CARLA loop.

Provides two interfaces:
  predict(obs)         → {'steer', 'throttle', 'brake'}   (CARLA control dict)
  select_action(state) → {'action': np.ndarray}            (trainer-compatible)

Weights are loaded from config baseline.model_path.
If the file is missing or the architecture does not match, the agent falls back
to random weights and logs a warning — training / evaluation still runs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .transfuser import BaselineNet

logger = logging.getLogger(__name__)


class BaselineAgent:
    """Inference-only agent wrapping BaselineNet."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        cfg = config.get("baseline", {})

        self.device = torch.device("cpu")  # TransFuser → CPU (32 GB RAM strategy)
        self.model_type: str = str(cfg.get("model_type", "transfuser"))
        self.model_path: Path = Path(cfg.get("model_path", "backend/weights/transfuser/model_seed1_39.pth"))

        lidar_cfg = config.get("lidar", {})
        self.lidar_channels: int = int(cfg.get("bev_channels", lidar_cfg.get("bev_channels", 2)))
        self.bev_size: int = int(cfg.get("bev_size", lidar_cfg.get("bev_size", 128)))
        self.embed_dim: int = int(cfg.get("embed_dim", 256))
        self.n_heads: int = int(cfg.get("n_heads", 4))
        self.n_layers: int = int(cfg.get("n_layers", 2))

        img_cfg = config.get("camera", {})
        self.img_h: int = int(img_cfg.get("height", 84))
        self.img_w: int = int(img_cfg.get("width", 84))

        self.net = BaselineNet(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            lidar_channels=self.lidar_channels,
            model_type=self.model_type,
        ).to(self.device)

        self.net.eval()
        self._load_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """Run inference and return a CARLA-compatible control dict.

        Parameters
        ----------
        obs : observation dict from CarlaGymEnv._get_observation()
              Expected keys: 'camera' (H×W×3 uint8), optionally 'lidar_bev'

        Returns
        -------
        {'steer': float, 'throttle': float, 'brake': float}  all in [0, 1] / [-1, 1]
        """
        rgb_t = self._preprocess_rgb(obs.get("camera"))
        lidar_t = self._preprocess_lidar(obs.get("lidar_bev"))

        with torch.no_grad():
            control_t = self.net(rgb_t, lidar_t)  # (1, 3)

        steer, throttle_raw, brake_raw = control_t.squeeze(0).tolist()

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip((throttle_raw + 1.0) / 2.0, 0.0, 1.0))
        brake = float(np.clip((brake_raw + 1.0) / 2.0, 0.0, 1.0))

        if throttle > brake:
            brake = 0.0
        else:
            throttle = 0.0

        return {"steer": steer, "throttle": throttle, "brake": brake}

    def select_action(
        self, state: Dict[str, np.ndarray], deterministic: bool = True
    ) -> Dict[str, Any]:
        """Trainer-compatible interface matching PPO/SAC select_action signature.

        state must contain at least 'image' (C×H×W float32).
        """
        rgb_np = state.get("image")
        if rgb_np is not None:
            rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).float().to(self.device)
        else:
            rgb_t = torch.zeros(1, 3, self.img_h, self.img_w, device=self.device)

        with torch.no_grad():
            control_t = self.net(rgb_t, None)

        action = control_t.squeeze(0).cpu().numpy().astype(np.float32)
        return {"action": action, "log_prob": 0.0}

    # ------------------------------------------------------------------
    # Training-mode guard
    # ------------------------------------------------------------------

    def store_transition(self, *args, **kwargs) -> None:  # noqa: ANN002
        raise NotImplementedError("BaselineAgent is inference-only — no training.")

    def update(self, *args, **kwargs) -> None:  # noqa: ANN002
        raise NotImplementedError("BaselineAgent is inference-only — no training.")

    def save(self, path: Optional[str] = None) -> None:
        logger.warning("BaselineAgent.save() called — weights are read-only pretrained.")

    def load(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.model_path
        self._load_weights(target)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_weights(self, path: Optional[Path] = None) -> None:
        target = path or self.model_path
        if not target.exists():
            logger.warning(
                "Pretrained weights not found at '%s'. "
                "BaselineAgent will use random initialisation. "
                "Download weights or set baseline.model_path in config.yaml.",
                target,
            )
            return

        try:
            # weights_only=False: checkpoint may contain non-tensor objects (optimizer
            # state, Python scalars) pickled by the original TransFuser training code.
            # Source is trusted (local file produced by our own training run), so
            # arbitrary-pickle risk is accepted; weights_only=True would fail to load.
            state_dict = torch.load(str(target), map_location=self.device, weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.net.load_state_dict(state_dict, strict=False)
            logger.info("Loaded baseline weights from '%s' (strict=False).", target)
        except Exception as exc:
            logger.warning(
                "Could not load weights from '%s': %s. "
                "Continuing with random initialisation.",
                target,
                exc,
            )

    def _preprocess_rgb(self, camera: Optional[np.ndarray]) -> torch.Tensor:
        if camera is None or not isinstance(camera, np.ndarray):
            return torch.zeros(1, 3, self.img_h, self.img_w, device=self.device)

        import cv2

        if camera.shape[:2] != (self.img_h, self.img_w):
            camera = cv2.resize(camera, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)

        t = torch.from_numpy(camera.transpose(2, 0, 1)).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def _preprocess_lidar(self, lidar_bev: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if lidar_bev is None or not isinstance(lidar_bev, np.ndarray):
            return None
        t = torch.from_numpy(lidar_bev).float()
        if t.ndim == 2:
            t = t.unsqueeze(0)  # add channel dim
        return t.unsqueeze(0).to(self.device)

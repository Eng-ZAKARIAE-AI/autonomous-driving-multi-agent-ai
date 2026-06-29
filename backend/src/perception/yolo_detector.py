"""YOLOv8 perception module for real-time object detection in CARLA.

PerceptionModule wraps ultralytics YOLOv8n and is toggled via
perception.use_yolo in config.yaml.  When disabled it is a no-op so the
rest of the pipeline never needs to branch on USE_YOLO explicitly.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO class ids relevant to autonomous driving
_CLASSES_OF_INTEREST: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    11: "stop_sign",
}


def _class_color(class_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR color for a given COCO class id."""
    digest = hashlib.md5(str(class_id).encode()).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))


class PerceptionModule:
    """YOLOv8-based object detector.

    Usage
    -----
    module = PerceptionModule(config)
    detections = module.detect(rgb_frame)        # list[Detection]
    annotated  = module.draw_overlay(frame, detections)

    Each detection dict has keys:
        class_name  : str
        class_id    : int
        confidence  : float  (0–1)
        bbox        : [x1, y1, x2, y2]  (pixel coords in original frame)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        cfg = config.get("perception", {})

        self.use_yolo: bool = bool(cfg.get("use_yolo", False))
        self.model_name: str = str(cfg.get("model_name", "yolov8n.pt"))
        self.confidence_threshold: float = float(cfg.get("confidence_threshold", 0.40))
        self.device: str = str(cfg.get("device", "cuda"))
        self.input_width: int = int(cfg.get("input_width", 640))
        self.input_height: int = int(cfg.get("input_height", 640))

        self._model = None

        if self.use_yolo:
            self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.use_yolo and self._model is not None

    def detect(self, frame: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """Run YOLOv8 detection on an RGB frame.

        Parameters
        ----------
        frame : np.ndarray | None
            H×W×3 uint8 RGB image (CARLA camera output).

        Returns
        -------
        list of detection dicts, empty when disabled or frame is None.
        """
        if not self.enabled or frame is None:
            return []

        try:
            results = self._model(
                frame,
                imgsz=(self.input_height, self.input_width),
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            return []

        detections: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
                class_name = _CLASSES_OF_INTEREST.get(
                    cls_id, result.names.get(cls_id, str(cls_id))
                )
                detections.append(
                    {
                        "class_name": class_name,
                        "class_id": cls_id,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        return detections

    def draw_overlay(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw bounding boxes + labels onto a copy of *frame*.

        Parameters
        ----------
        frame      : H×W×3 uint8 RGB image
        detections : output from detect()

        Returns
        -------
        Annotated copy of the frame (RGB).
        """
        if frame is None or not detections:
            return frame

        import cv2  # lazy import — only needed when overlay is requested

        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = _class_color(det["class_id"])

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(output, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(
                output, label, (x1 + 1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            )
        return output

    def filter_by_class(
        self,
        detections: List[Dict[str, Any]],
        class_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Keep only detections whose class_name is in *class_names*."""
        names_set = set(class_names)
        return [d for d in detections if d["class_name"] in names_set]

    def closest_obstacle(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> Optional[Dict[str, Any]]:
        """Return the detection with the largest bounding-box area (heuristic for closest).

        Returns None when detections is empty.
        """
        if not detections:
            return None
        return max(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_name)
            self._model.to(self.device)
            logger.info(
                "YOLOv8 model '%s' loaded on device '%s'", self.model_name, self.device
            )
        except ImportError:
            logger.error(
                "ultralytics is not installed — YOLO disabled. "
                "Install it with: pip install ultralytics"
            )
            self.use_yolo = False
        except Exception as exc:
            logger.error("Failed to load YOLO model '%s': %s", self.model_name, exc)
            self.use_yolo = False

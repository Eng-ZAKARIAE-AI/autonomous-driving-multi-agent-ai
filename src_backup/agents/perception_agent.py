"""
Perception Agent - Handles object detection and scene understanding.

This agent is responsible for:
- Detecting vehicles, pedestrians, and other objects in the environment
- Estimating distances and positions
- Providing semantic understanding of the scene
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional

# Try to import YOLO, fallback to simple detection if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available, using simple detection")

from agents.base_agent import BaseAgent, EnvironmentState, MessageType, AgentState


class DetectedObject:
    """Represents a detected object in the environment."""

    def __init__(self, class_id: int, class_name: str, confidence: float,
                 bbox: Tuple[float, float, float, float], position_3d: Optional[Tuple[float, float, float]] = None):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2) in image coordinates
        self.position_3d = position_3d  # (x, y, z) in world coordinates
        self.velocity = None  # Will be estimated by prediction agent


class PerceptionAgent(BaseAgent):
    """Agent responsible for environmental perception using computer vision."""

    def __init__(self, message_bus, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        super().__init__("perception_agent", message_bus)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.camera_intrinsics = None  # Will be set from CARLA camera
        self.last_detections = []

        # Class names for COCO dataset (YOLOv8 default)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def initialize(self) -> bool:
        """Initialize the perception model."""
        try:
            if YOLO_AVAILABLE:
                # Load YOLOv8 model (you can specify a custom model path)
                model_name = "yolov8n.pt"  # nano model for real-time performance
                self.model = YOLO(model_name)
                print(f"✅ Perception Agent initialized with {model_name}")
            else:
                print("⚠️ Perception Agent initialized with simple detection (YOLO not available)")
                self.model = None

            self.state = AgentState.READY
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Perception Agent: {e}")
            self.state = AgentState.ERROR
            return False

    def process(self, environment_state: EnvironmentState) -> None:
        """Process camera images and detect objects."""
        if self.state != AgentState.READY:
            return

        self.state = AgentState.PROCESSING

        try:
            # Get camera image from environment state
            camera_image = environment_state.ego_vehicle.camera if hasattr(environment_state.ego_vehicle, 'camera') else None

            if camera_image is None:
                # For now, we'll work with the existing CARLA simulation structure
                # This will be updated when we integrate with the full multi-agent system
                detections = self._simulate_detections(environment_state)
            else:
                detections = self._detect_objects(camera_image)

            # Store detections for other agents
            self.last_detections = detections

            # Send perception data to other agents
            perception_data = {
                'detections': [
                    {
                        'class_name': obj.class_name,
                        'confidence': obj.confidence,
                        'bbox': obj.bbox,
                        'position_3d': obj.position_3d
                    } for obj in detections
                ],
                'timestamp': environment_state.timestamp
            }

            self.broadcast_message(MessageType.PERCEPTION_DATA, perception_data)

            # Send status update
            self.broadcast_message(MessageType.STATUS_UPDATE, {
                'agent_id': self.agent_id,
                'status': 'processing',
                'detections_count': len(detections)
            })

        except Exception as e:
            print(f"❌ Perception Agent error: {e}")
            self.state = AgentState.ERROR
        finally:
            if self.state == AgentState.PROCESSING:
                self.state = AgentState.READY

    def _detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """Run object detection on an image."""
        if self.model is None or not YOLO_AVAILABLE:
            return self._detect_objects_simple(image)

        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                        # Create detection object
                        detection = DetectedObject(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(confidence),
                            bbox=(float(x1), float(y1), float(x2), float(y2))
                        )

                        # Estimate 3D position (simplified - would need proper depth estimation)
                        detection.position_3d = self._estimate_3d_position(detection.bbox, image.shape)

                        detections.append(detection)

            return detections

        except Exception as e:
            print(f"❌ Object detection failed: {e}")
            return []

    def _detect_objects_simple(self, image: np.ndarray) -> List[DetectedObject]:
        """Simple object detection using OpenCV (fallback when YOLO not available)."""
        if image is None:
            return []

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections = []
            min_area = 500  # Minimum contour area
            max_area = 50000  # Maximum contour area

            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate aspect ratio
                    aspect_ratio = w / h if h > 0 else 1.0

                    # Classify based on shape
                    if aspect_ratio > 1.2:  # Wider than tall - likely vehicle
                        class_name = "car"
                        class_id = 2
                        confidence = 0.6
                    elif aspect_ratio < 0.8:  # Taller than wide - likely pedestrian
                        class_name = "person"
                        class_id = 0
                        confidence = 0.7
                    else:  # Square-ish - could be traffic sign
                        class_name = "traffic light"
                        class_id = 9
                        confidence = 0.5

                    if confidence >= self.confidence_threshold:
                        detection = DetectedObject(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(float(x), float(y), float(x+w), float(y+h))
                        )

                        # Estimate 3D position
                        detection.position_3d = self._estimate_3d_position(detection.bbox, image.shape)
                        detections.append(detection)

            return detections

        except Exception as e:
            print(f"❌ Simple detection failed: {e}")
            return []

    def _estimate_3d_position(self, bbox: Tuple[float, float, float, float],
                            image_shape: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Estimate 3D position from 2D bounding box (simplified implementation)."""
        # This is a very simplified estimation
        # In a real system, you would use depth estimation, stereo vision, or LiDAR
        x1, y1, x2, y2 = bbox
        image_height, image_width = image_shape[:2]

        # Estimate distance based on bounding box size (larger = closer)
        bbox_area = (x2 - x1) * (y2 - y1)
        max_area = image_width * image_height * 0.5  # Assume max 50% of image
        distance = max(5.0, 50.0 * (1.0 - bbox_area / max_area))  # 5-50 meters

        # Estimate lateral position based on horizontal center
        center_x = (x1 + x2) / 2
        lateral_offset = (center_x - image_width / 2) / (image_width / 2) * 10.0  # -10 to +10 meters

        return (lateral_offset, distance, 0.0)  # x, y, z relative to ego vehicle

    def _simulate_detections(self, environment_state: EnvironmentState) -> List[DetectedObject]:
        """Simulate detections based on environment state (for development)."""
        detections = []

        # Simulate pedestrian detection
        if environment_state.pedestrians:
            for ped in environment_state.pedestrians:
                detection = DetectedObject(
                    class_id=0,  # person
                    class_name="person",
                    confidence=0.9,
                    bbox=(100, 100, 200, 300),  # dummy bbox
                    position_3d=ped.get('position', (0, 10, 0))
                )
                detections.append(detection)

        # Simulate vehicle detections
        if environment_state.other_vehicles:
            for veh in environment_state.other_vehicles:
                detection = DetectedObject(
                    class_id=2,  # car
                    class_name="car",
                    confidence=0.85,
                    bbox=(50, 80, 150, 180),  # dummy bbox
                    position_3d=veh.get('position', (5, 15, 0))
                )
                detections.append(detection)

        return detections

    def get_status(self) -> Dict[str, Any]:
        """Return perception agent status."""
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'model_loaded': self.model is not None,
            'last_detections_count': len(self.last_detections),
            'confidence_threshold': self.confidence_threshold
        }
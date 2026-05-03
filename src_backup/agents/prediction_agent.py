"""
Prediction Agent - Predicts behavior of other vehicles and pedestrians.

This agent is responsible for:
- Tracking objects over time
- Predicting future trajectories
- Estimating intentions (lane changes, stops, turns)
- Providing uncertainty estimates
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
from agents.base_agent import BaseAgent, EnvironmentState, MessageType, AgentMessage, AgentState


class TrackedObject:
    """Represents a tracked object with prediction capabilities."""

    def __init__(self, object_id: str, initial_detection: Dict[str, Any]):
        self.object_id = object_id
        self.class_name = initial_detection.get('class_name', 'unknown')
        self.positions = deque(maxlen=10)  # Store last 10 positions
        self.velocities = deque(maxlen=10)  # Store last 10 velocities
        self.timestamps = deque(maxlen=10)  # Store timestamps

        # Initialize with first detection
        initial_pos = initial_detection.get('position_3d', (0, 0, 0))
        self.positions.append(initial_pos)
        self.velocities.append((0, 0, 0))  # Initial velocity unknown
        self.timestamps.append(initial_detection.get('timestamp', 0))

        # Prediction state
        self.predicted_trajectory = []
        self.predicted_intention = "unknown"
        self.confidence = 0.5

    def update(self, detection: Dict[str, Any]) -> None:
        """Update tracking with new detection."""
        current_pos = detection.get('position_3d', (0, 0, 0))
        current_time = detection.get('timestamp', 0)

        if len(self.positions) > 0:
            # Calculate velocity
            prev_pos = self.positions[-1]
            prev_time = self.timestamps[-1]
            dt = current_time - prev_time

            if dt > 0:
                velocity = (
                    (current_pos[0] - prev_pos[0]) / dt,
                    (current_pos[1] - prev_pos[1]) / dt,
                    (current_pos[2] - prev_pos[2]) / dt
                )
                self.velocities.append(velocity)

        self.positions.append(current_pos)
        self.timestamps.append(current_time)

        # Update predictions
        self._update_predictions()

    def _update_predictions(self) -> None:
        """Update trajectory predictions and intentions."""
        if len(self.positions) < 3:
            return

        # Simple linear prediction based on recent velocity
        current_pos = self.positions[-1]
        current_vel = self.velocities[-1] if self.velocities else (0, 0, 0)

        # Predict next 2 seconds (assuming 10Hz updates = 20 steps)
        self.predicted_trajectory = []
        for i in range(20):
            dt = (i + 1) * 0.1  # 0.1 second steps
            predicted_pos = (
                current_pos[0] + current_vel[0] * dt,
                current_pos[1] + current_vel[1] * dt,
                current_pos[2] + current_vel[2] * dt
            )
            self.predicted_trajectory.append(predicted_pos)

        # Simple intention prediction based on velocity and position
        speed = np.linalg.norm(current_vel)
        if speed < 0.5:
            self.predicted_intention = "stopped"
            self.confidence = 0.8
        elif abs(current_vel[0]) > 2.0:  # Significant lateral movement
            self.predicted_intention = "lane_change"
            self.confidence = 0.7
        elif current_vel[1] < -5.0:  # Moving towards ego vehicle
            self.predicted_intention = "approaching"
            self.confidence = 0.9
        else:
            self.predicted_intention = "cruising"
            self.confidence = 0.6

    def get_prediction(self) -> Dict[str, Any]:
        """Get current prediction for this object."""
        return {
            'object_id': self.object_id,
            'class_name': self.class_name,
            'current_position': self.positions[-1] if self.positions else (0, 0, 0),
            'current_velocity': self.velocities[-1] if self.velocities else (0, 0, 0),
            'predicted_trajectory': self.predicted_trajectory,
            'predicted_intention': self.predicted_intention,
            'confidence': self.confidence,
            'track_length': len(self.positions)
        }


class PredictionAgent(BaseAgent):
    """Agent responsible for predicting behavior of other road users."""

    def __init__(self, message_bus, max_track_age: float = 2.0):
        super().__init__("prediction_agent", message_bus)
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.next_object_id = 0
        self.max_track_age = max_track_age  # seconds

    def initialize(self) -> bool:
        """Initialize the prediction agent."""
        try:
            print("✅ Prediction Agent initialized")
            self.state = AgentState.READY
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Prediction Agent: {e}")
            self.state = AgentState.ERROR
            return False

    def process(self, environment_state: EnvironmentState) -> None:
        """Process perception data and update predictions."""
        if self.state != AgentState.READY:
            return

        self.state = AgentState.PROCESSING

        try:
            # Get perception data from messages
            messages = self.receive_messages()
            perception_data = None

            for message in messages:
                if message.message_type == MessageType.PERCEPTION_DATA:
                    perception_data = message.data
                    break

            if perception_data:
                self._update_tracks(perception_data)
                self._cleanup_old_tracks(perception_data.get('timestamp', 0))

                # Generate predictions
                predictions = self._generate_predictions()

                # Send predictions to other agents
                self.broadcast_message(MessageType.PREDICTION_UPDATE, {
                    'predictions': predictions,
                    'timestamp': perception_data.get('timestamp', 0)
                })

            # Send status update
            self.broadcast_message(MessageType.STATUS_UPDATE, {
                'agent_id': self.agent_id,
                'status': 'processing',
                'tracked_objects': len(self.tracked_objects)
            })

        except Exception as e:
            print(f"❌ Prediction Agent error: {e}")
            self.state = AgentState.ERROR
        finally:
            if self.state == AgentState.PROCESSING:
                self.state = AgentState.READY

    def _update_tracks(self, perception_data: Dict[str, Any]) -> None:
        """Update object tracks with new detections."""
        detections = perception_data.get('detections', [])
        timestamp = perception_data.get('timestamp', 0)

        # Associate detections with existing tracks (simple nearest neighbor)
        used_detections = set()

        for detection in detections:
            detection['timestamp'] = timestamp
            best_match = self._find_best_track_match(detection)

            if best_match:
                self.tracked_objects[best_match].update(detection)
                used_detections.add(id(detection))
            else:
                # Create new track
                object_id = f"obj_{self.next_object_id}"
                self.next_object_id += 1
                self.tracked_objects[object_id] = TrackedObject(object_id, detection)
                used_detections.add(id(detection))

    def _find_best_track_match(self, detection: Dict[str, Any]) -> Optional[str]:
        """Find the best existing track for a detection (simple distance-based matching)."""
        detection_pos = detection.get('position_3d', (0, 0, 0))
        min_distance = float('inf')
        best_match = None

        for track_id, track in self.tracked_objects.items():
            if not track.positions:
                continue

            track_pos = track.positions[-1]
            distance = np.linalg.norm(np.array(detection_pos) - np.array(track_pos))

            # Only match if close enough and same class
            if distance < 5.0 and track.class_name == detection.get('class_name'):
                if distance < min_distance:
                    min_distance = distance
                    best_match = track_id

        return best_match

    def _cleanup_old_tracks(self, current_time: float) -> None:
        """Remove tracks that haven't been updated recently."""
        to_remove = []
        for track_id, track in self.tracked_objects.items():
            if track.timestamps and current_time - track.timestamps[-1] > self.max_track_age:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracked_objects[track_id]

    def _generate_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions for all tracked objects."""
        predictions = []
        for track in self.tracked_objects.values():
            predictions.append(track.get_prediction())
        return predictions

    def get_status(self) -> Dict[str, Any]:
        """Return prediction agent status."""
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'tracked_objects': len(self.tracked_objects),
            'max_track_age': self.max_track_age
        }
"""
Decision-Making Agent - Makes high-level driving decisions.

This agent is responsible for:
- Analyzing the current situation
- Determining appropriate driving behavior
- Coordinating with other agents
- Making safety-critical decisions
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
from agents.base_agent import BaseAgent, EnvironmentState, MessageType, ControlCommand, DrivingAction, AgentState


class DecisionContext:
    """Context information for decision making."""

    def __init__(self):
        self.ego_speed = 0.0
        self.ego_lane = "unknown"
        self.speed_limit = 50.0  # km/h
        self.distance_to_next_vehicle = float('inf')
        self.distance_to_pedestrian = float('inf')
        self.traffic_light_state = "unknown"
        self.weather_conditions = "clear"
        self.time_to_collision = float('inf')
        self.safety_margin = 2.5  # meters


class DecisionMakingAgent(BaseAgent):
    """Agent responsible for high-level driving decisions."""

    def __init__(self, message_bus, safety_margin: float = 2.5, reaction_time: float = 1.5):
        super().__init__("decision_agent", message_bus)
        self.safety_margin = safety_margin
        self.reaction_time = reaction_time  # seconds
        self.context = DecisionContext()
        self.last_action = DrivingAction.CRUISE

    def initialize(self) -> bool:
        """Initialize the decision-making agent."""
        try:
            print("✅ Decision-Making Agent initialized")
            self.state = AgentState.READY
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Decision-Making Agent: {e}")
            self.state = AgentState.ERROR
            return False

    def process(self, environment_state: EnvironmentState) -> None:
        """Analyze situation and make driving decisions."""
        if self.state != AgentState.READY:
            return

        self.state = AgentState.PROCESSING

        try:
            # Get data from other agents
            messages = self.receive_messages()
            perception_data = None
            prediction_data = None

            for message in messages:
                if message.message_type == MessageType.PERCEPTION_DATA:
                    perception_data = message.data
                elif message.message_type == MessageType.PREDICTION_UPDATE:
                    prediction_data = message.data

            # Update context with available data
            self._update_context(environment_state, perception_data, prediction_data)

            # Make decision
            action = self._make_decision()

            # Send decision to planning agent
            decision_data = {
                'action': action.value,
                'context': self._context_to_dict(),
                'timestamp': environment_state.timestamp,
                'reasoning': self._get_decision_reasoning(action)
            }

            self.send_message("planning_agent", MessageType.DECISION_REQUEST, decision_data)

            # Send status update
            self.broadcast_message(MessageType.STATUS_UPDATE, {
                'agent_id': self.agent_id,
                'status': 'processing',
                'current_action': action.value
            })

        except Exception as e:
            print(f"❌ Decision-Making Agent error: {e}")
            self.state = AgentState.ERROR
        finally:
            if self.state == AgentState.PROCESSING:
                self.state = AgentState.READY

    def _update_context(self, env_state: EnvironmentState,
                       perception_data: Optional[Dict[str, Any]],
                       prediction_data: Optional[Dict[str, Any]]) -> None:
        """Update decision context with latest information."""

        # Update ego vehicle information
        if hasattr(env_state.ego_vehicle, 'velocity'):
            ego_vel = env_state.ego_vehicle.velocity
            self.context.ego_speed = np.linalg.norm(ego_vel) * 3.6  # Convert to km/h

        # Update distances from perception data
        if perception_data:
            detections = perception_data.get('detections', [])
            self._update_distances_from_detections(detections)

        # Update predictions
        if prediction_data:
            predictions = prediction_data.get('predictions', [])
            self._update_predictions(predictions)

    def _update_distances_from_detections(self, detections: List[Dict[str, Any]]) -> None:
        """Update distance measurements from detections."""
        min_vehicle_distance = float('inf')
        min_pedestrian_distance = float('inf')

        for detection in detections:
            position_3d = detection.get('position_3d')
            if position_3d:
                distance = np.linalg.norm(position_3d)
                class_name = detection.get('class_name', '')

                if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                    min_vehicle_distance = min(min_vehicle_distance, distance)
                elif class_name in ['person', 'pedestrian']:
                    min_pedestrian_distance = min(min_pedestrian_distance, distance)

        self.context.distance_to_next_vehicle = min_vehicle_distance
        self.context.distance_to_pedestrian = min_pedestrian_distance

    def _update_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """Update context with prediction information."""
        # Calculate time to collision based on predictions
        min_ttc = float('inf')

        for prediction in predictions:
            trajectory = prediction.get('predicted_trajectory', [])
            current_pos = prediction.get('current_position', (0, 0, 0))
            current_vel = prediction.get('current_velocity', (0, 0, 0))

            # Simple TTC calculation (relative speed towards ego vehicle)
            relative_speed = -current_vel[1]  # Assuming ego is moving in +Y direction
            if relative_speed > 0.1:  # Object is approaching
                distance = abs(current_pos[1])
                ttc = distance / relative_speed
                min_ttc = min(min_ttc, ttc)

        self.context.time_to_collision = min_ttc

    def _make_decision(self) -> DrivingAction:
        """Make the primary driving decision based on current context."""

        # Emergency situations first
        if self.context.time_to_collision < 2.0:
            return DrivingAction.EMERGENCY_BRAKE

        if self.context.distance_to_pedestrian < 5.0:
            return DrivingAction.EMERGENCY_BRAKE

        # Safety margins
        if self.context.distance_to_next_vehicle < self.safety_margin * 2:
            return DrivingAction.DECELERATE

        # Speed control
        if self.context.ego_speed > self.context.speed_limit:
            return DrivingAction.DECELERATE

        if self.context.ego_speed < self.context.speed_limit * 0.8:
            return DrivingAction.ACCELERATE

        # Default cruising behavior
        return DrivingAction.CRUISE

    def _get_decision_reasoning(self, action: DrivingAction) -> str:
        """Provide reasoning for the decision."""
        if action == DrivingAction.EMERGENCY_BRAKE:
            if self.context.time_to_collision < 2.0:
                return f"Emergency brake: TTC = {self.context.time_to_collision:.1f}s"
            elif self.context.distance_to_pedestrian < 5.0:
                return f"Emergency brake: pedestrian at {self.context.distance_to_pedestrian:.1f}m"
        elif action == DrivingAction.DECELERATE:
            if self.context.distance_to_next_vehicle < self.safety_margin * 2:
                return f"Decelerate: vehicle at {self.context.distance_to_next_vehicle:.1f}m"
            else:
                return f"Decelerate: speed {self.context.ego_speed:.1f} > limit {self.context.speed_limit}"
        elif action == DrivingAction.ACCELERATE:
            return f"Accelerate: speed {self.context.ego_speed:.1f} < target"
        elif action == DrivingAction.CRUISE:
            return f"Cruise: maintaining speed {self.context.ego_speed:.1f}"

        return "Default cruising behavior"

    def _context_to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for messaging."""
        return {
            'ego_speed': self.context.ego_speed,
            'distance_to_next_vehicle': self.context.distance_to_next_vehicle,
            'distance_to_pedestrian': self.context.distance_to_pedestrian,
            'time_to_collision': self.context.time_to_collision,
            'speed_limit': self.context.speed_limit,
            'safety_margin': self.context.safety_margin
        }

    def get_status(self) -> Dict[str, Any]:
        """Return decision-making agent status."""
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'current_action': self.last_action.value,
            'context': self._context_to_dict()
        }
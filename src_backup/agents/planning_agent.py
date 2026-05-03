"""
Planning Agent - Handles trajectory planning and path generation.

This agent is responsible for:
- Generating safe trajectories
- Path planning around obstacles
- Optimizing for comfort and efficiency
- Providing waypoints for the control agent
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from agents.base_agent import BaseAgent, EnvironmentState, MessageType, ControlCommand, DrivingAction, AgentState


class TrajectoryPoint:
    """A point in a planned trajectory."""

    def __init__(self, x: float, y: float, yaw: float, speed: float, time: float):
        self.x = x
        self.y = y
        self.yaw = yaw  # radians
        self.speed = speed  # m/s
        self.time = time  # seconds from start


class Trajectory:
    """A planned vehicle trajectory."""

    def __init__(self, points: List[TrajectoryPoint], total_time: float):
        self.points = points
        self.total_time = total_time
        self.current_index = 0

    def get_current_target(self, current_time: float) -> Optional[TrajectoryPoint]:
        """Get the target point for the current time."""
        if not self.points:
            return None

        # Find the appropriate point based on time
        for i, point in enumerate(self.points):
            if point.time >= current_time:
                return point

        # If we've passed all points, return the last one
        return self.points[-1] if self.points else None

    def is_complete(self, current_time: float) -> bool:
        """Check if trajectory is complete."""
        return current_time >= self.total_time


class PlanningAgent(BaseAgent):
    """Agent responsible for trajectory planning and path generation."""

    def __init__(self, message_bus, planning_horizon: float = 5.0, time_step: float = 0.1):
        super().__init__("planning_agent", message_bus)
        self.planning_horizon = planning_horizon  # seconds
        self.time_step = time_step  # seconds
        self.current_trajectory: Optional[Trajectory] = None
        self.max_acceleration = 3.0  # m/s²
        self.max_deceleration = -6.0  # m/s²
        self.max_speed = 13.89  # m/s (50 km/h)
        self.min_speed = 0.0  # m/s

    def initialize(self) -> bool:
        """Initialize the planning agent."""
        try:
            print("✅ Planning Agent initialized")
            self.state = AgentState.READY
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Planning Agent: {e}")
            self.state = AgentState.ERROR
            return False

    def process(self, environment_state: EnvironmentState) -> None:
        """Generate or update trajectory based on decisions."""
        if self.state != AgentState.READY:
            return

        self.state = AgentState.PROCESSING

        try:
            # Get decision from decision-making agent
            messages = self.receive_messages()
            decision_data = None

            for message in messages:
                if message.message_type == MessageType.DECISION_REQUEST:
                    decision_data = message.data
                    break

            if decision_data:
                action = DrivingAction(decision_data.get('action', 'cruise'))
                context = decision_data.get('context', {})

                # Generate trajectory
                trajectory = self._generate_trajectory(action, context, environment_state)

                if trajectory:
                    self.current_trajectory = trajectory

                    # Send trajectory to control agent
                    trajectory_data = self._trajectory_to_dict(trajectory)
                    self.send_message("control_agent", MessageType.PLANNING_RESULT, {
                        'trajectory': trajectory_data,
                        'timestamp': environment_state.timestamp
                    })

            # Send status update
            self.broadcast_message(MessageType.STATUS_UPDATE, {
                'agent_id': self.agent_id,
                'status': 'processing',
                'has_trajectory': self.current_trajectory is not None
            })

        except Exception as e:
            print(f"❌ Planning Agent error: {e}")
            self.state = AgentState.ERROR
        finally:
            if self.state == AgentState.PROCESSING:
                self.state = AgentState.READY

    def _generate_trajectory(self, action: DrivingAction,
                           context: Dict[str, Any],
                           env_state: EnvironmentState) -> Optional[Trajectory]:
        """Generate a trajectory based on the desired action."""

        # Get current state
        current_pos = (0, 0, 0)  # Relative to ego vehicle
        current_speed = context.get('ego_speed', 0) / 3.6  # Convert km/h to m/s
        current_yaw = 0.0  # radians

        # Determine target speed and behavior based on action
        target_speed = self._get_target_speed(action, context)

        # Generate trajectory points
        points = []
        current_time = 0.0

        while current_time < self.planning_horizon:
            # Calculate speed for this time step
            speed = self._calculate_speed_profile(current_speed, target_speed, current_time)

            # Calculate position (simple kinematic model)
            # For now, assume straight line motion
            dt = self.time_step
            dx = speed * np.cos(current_yaw) * dt
            dy = speed * np.sin(current_yaw) * dt

            current_pos = (
                current_pos[0] + dx,
                current_pos[1] + dy,
                current_pos[2]
            )

            # Create trajectory point
            point = TrajectoryPoint(
                x=current_pos[0],
                y=current_pos[1],
                yaw=current_yaw,
                speed=speed,
                time=current_time
            )
            points.append(point)

            current_time += dt

        return Trajectory(points, current_time) if points else None

    def _get_target_speed(self, action: DrivingAction, context: Dict[str, Any]) -> float:
        """Determine target speed based on action and context."""
        base_speed = context.get('ego_speed', 30) / 3.6  # Convert km/h to m/s
        speed_limit = context.get('speed_limit', 50) / 3.6

        if action == DrivingAction.EMERGENCY_BRAKE:
            return 0.0
        elif action == DrivingAction.STOP:
            return 0.0
        elif action == DrivingAction.DECELERATE:
            return max(base_speed * 0.8, 0.0)
        elif action == DrivingAction.ACCELERATE:
            return min(base_speed * 1.2, speed_limit)
        elif action in [DrivingAction.LANE_CHANGE_LEFT, DrivingAction.LANE_CHANGE_RIGHT,
                       DrivingAction.TURN_LEFT, DrivingAction.TURN_RIGHT]:
            return base_speed * 0.9  # Slightly slower for maneuvers
        else:  # CRUISE or default
            return min(base_speed, speed_limit)

    def _calculate_speed_profile(self, current_speed: float, target_speed: float, time: float) -> float:
        """Calculate speed at a given time using smooth acceleration."""
        speed_diff = target_speed - current_speed

        if abs(speed_diff) < 0.1:
            return target_speed

        # Calculate required acceleration
        if speed_diff > 0:
            # Acceleration
            accel = min(self.max_acceleration, speed_diff / (self.planning_horizon - time))
        else:
            # Deceleration
            accel = max(self.max_deceleration, speed_diff / (self.planning_horizon - time))

        # Calculate speed at this time step
        speed = current_speed + accel * self.time_step
        speed = np.clip(speed, self.min_speed, self.max_speed)

        return speed

    def _trajectory_to_dict(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Convert trajectory to dictionary for messaging."""
        return {
            'points': [
                {
                    'x': point.x,
                    'y': point.y,
                    'yaw': point.yaw,
                    'speed': point.speed,
                    'time': point.time
                } for point in trajectory.points
            ],
            'total_time': trajectory.total_time
        }

    def get_current_target(self, current_time: float) -> Optional[TrajectoryPoint]:
        """Get current trajectory target point."""
        if self.current_trajectory:
            return self.current_trajectory.get_current_target(current_time)
        return None

    def get_status(self) -> Dict[str, Any]:
        """Return planning agent status."""
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'has_trajectory': self.current_trajectory is not None,
            'planning_horizon': self.planning_horizon,
            'time_step': self.time_step
        }
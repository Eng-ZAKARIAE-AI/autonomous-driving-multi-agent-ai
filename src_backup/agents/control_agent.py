"""
Control Agent - Handles low-level vehicle control.

This agent is responsible for:
- Converting trajectory points to control commands
- PID control for speed and steering
- Executing control commands
- Safety monitoring and overrides
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from agents.base_agent import BaseAgent, EnvironmentState, MessageType, ControlCommand, AgentState


class PIDController:
    """Simple PID controller for speed and steering control."""

    def __init__(self, kp: float, ki: float, kd: float, output_limits: Tuple[float, float] = (-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = None

    def update(self, error: float, current_time: float) -> float:
        """Update PID controller and return control output."""
        if self.previous_time is None:
            self.previous_time = current_time
            dt = 0.1  # Default time step
        else:
            dt = current_time - self.previous_time
            if dt <= 0:
                dt = 0.1

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt

        # Calculate output
        output = proportional + integral + derivative

        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Update state
        self.previous_error = error
        self.previous_time = current_time

        return output


class ControlAgent(BaseAgent):
    """Agent responsible for low-level vehicle control."""

    def __init__(self, message_bus):
        super().__init__("control_agent", message_bus)

        # PID controllers
        self.speed_controller = PIDController(kp=0.5, ki=0.1, kd=0.05, output_limits=(-1.0, 1.0))
        self.steering_controller = PIDController(kp=1.0, ki=0.0, kd=0.1, output_limits=(-1.0, 1.0))

        # Control state
        self.current_command = ControlCommand()
        self.target_trajectory = None
        self.last_update_time = 0.0

    def initialize(self) -> bool:
        """Initialize the control agent."""
        try:
            print("✅ Control Agent initialized")
            self.state = AgentState.READY
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Control Agent: {e}")
            self.state = AgentState.ERROR
            return False

    def process(self, environment_state: EnvironmentState) -> None:
        """Generate control commands based on planning results."""
        if self.state != AgentState.READY:
            return

        self.state = AgentState.PROCESSING

        try:
            current_time = environment_state.timestamp

            # Get planning results
            messages = self.receive_messages()
            planning_data = None

            for message in messages:
                if message.message_type == MessageType.PLANNING_RESULT:
                    planning_data = message.data
                    break

            if planning_data:
                self.target_trajectory = planning_data.get('trajectory')

            # Generate control command
            control_command = self._generate_control_command(current_time)

            # Safety checks
            control_command = self._apply_safety_checks(control_command, environment_state)

            # Store current command
            self.current_command = control_command

            # In a real system, this would send commands to the vehicle
            # For now, we'll just log the commands
            self._log_control_command(control_command, current_time)

            # Send status update
            self.broadcast_message(MessageType.STATUS_UPDATE, {
                'agent_id': self.agent_id,
                'status': 'processing',
                'control_command': self._command_to_dict(control_command)
            })

        except Exception as e:
            print(f"❌ Control Agent error: {e}")
            self.state = AgentState.ERROR
        finally:
            if self.state == AgentState.PROCESSING:
                self.state = AgentState.READY

    def _generate_control_command(self, current_time: float) -> ControlCommand:
        """Generate control command based on current trajectory."""

        command = ControlCommand()

        if not self.target_trajectory:
            # No trajectory available, maintain current state
            return command

        # Get target point from trajectory
        target_point = self._get_target_point(current_time)
        if not target_point:
            return command

        # Get current vehicle state (simplified)
        current_speed = 0.0  # Would come from environment state
        current_yaw = 0.0

        # Extract target speed and heading from trajectory point
        if isinstance(target_point, dict):
            target_speed = target_point.get('speed', 0.0)
            target_yaw = target_point.get('yaw', 0.0)
        else:
            target_speed = getattr(target_point, 'speed', 0.0)
            target_yaw = getattr(target_point, 'yaw', 0.0)

        # Speed control
        speed_error = target_speed - current_speed
        throttle_brake = self.speed_controller.update(speed_error, current_time)

        if throttle_brake > 0:
            command.throttle = throttle_brake
            command.brake = 0.0
        else:
            command.throttle = 0.0
            command.brake = -throttle_brake

        # Steering control
        # Calculate heading error (simplified - assumes straight road)
        yaw_error = target_yaw - current_yaw

        # Normalize angle to [-pi, pi]
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

        command.steer = self.steering_controller.update(yaw_error, current_time)
        command.timestamp = current_time

        return command

    def _get_target_point(self, current_time: float) -> Optional[Dict[str, Any]]:
        """Get target point from trajectory for current time."""
        if not self.target_trajectory:
            return None

        points = self.target_trajectory.get('points', [])
        if not points:
            return None

        # Find appropriate point based on time
        for point in points:
            if point['time'] >= current_time:
                return point

        # Return last point if we've passed all
        return points[-1] if points else None

    def _apply_safety_checks(self, command: ControlCommand, env_state: EnvironmentState) -> ControlCommand:
        """Apply safety checks and overrides to control command."""

        # Emergency brake override
        # This would check for imminent collisions, etc.
        # For now, just ensure reasonable limits

        command.throttle = np.clip(command.throttle, 0.0, 1.0)
        command.brake = np.clip(command.brake, 0.0, 1.0)
        command.steer = np.clip(command.steer, -1.0, 1.0)

        # If brake is too high, reduce throttle to zero
        if command.brake > 0.8:
            command.throttle = 0.0

        return command

    def _log_control_command(self, command: ControlCommand, timestamp: float) -> None:
        """Log control command for debugging."""
        print(f"🎮 Control Command [t={timestamp:.2f}]: "
              f"throttle={command.throttle:.2f}, "
              f"brake={command.brake:.2f}, "
              f"steer={command.steer:.2f}")

    def _command_to_dict(self, command: ControlCommand) -> Dict[str, Any]:
        """Convert control command to dictionary."""
        return {
            'throttle': command.throttle,
            'brake': command.brake,
            'steer': command.steer,
            'hand_brake': command.hand_brake,
            'reverse': command.reverse
        }

    def get_current_command(self) -> ControlCommand:
        """Get the current control command."""
        return self.current_command

    def get_status(self) -> Dict[str, Any]:
        """Return control agent status."""
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'current_command': self._command_to_dict(self.current_command),
            'has_trajectory': self.target_trajectory is not None
        }
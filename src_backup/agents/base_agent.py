"""
Multi-Agent Autonomous Driving System Architecture

This module defines the core multi-agent architecture for autonomous driving,
including base agent classes and communication mechanisms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import threading
import queue


class AgentState(Enum):
    """States that agents can be in during operation."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""
    PERCEPTION_DATA = "perception_data"
    PREDICTION_UPDATE = "prediction_update"
    DECISION_REQUEST = "decision_request"
    PLANNING_RESULT = "planning_result"
    CONTROL_COMMAND = "control_command"
    SAFETY_ALERT = "safety_alert"
    STATUS_UPDATE = "status_update"


class DrivingAction(Enum):
    """High-level driving actions."""
    CRUISE = "cruise"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    STOP = "stop"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    EMERGENCY_BRAKE = "emergency_brake"
    YIELD = "yield"
    OVERTAKE = "overtake"


@dataclass
class ControlCommand:
    """Low-level control command for the vehicle."""
    throttle: float = 0.0  # 0.0 to 1.0
    brake: float = 0.0     # 0.0 to 1.0
    steer: float = 0.0     # -1.0 (left) to 1.0 (right)
    timestamp: float = 0.0
    hand_brake: bool = False
    reverse: bool = False

    def __post_init__(self):
        # Validate ranges
        self.throttle = max(0.0, min(1.0, self.throttle))
        self.brake = max(0.0, min(1.0, self.brake))
        self.steer = max(-1.0, min(1.0, self.steer))


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    timestamp: float
    data: Dict[str, Any]
    priority: int = 1  # 1=low, 5=high


@dataclass
class VehicleState:
    """Current state of the ego vehicle."""
    position: Tuple[float, float, float]  # x, y, z
    velocity: Tuple[float, float, float]  # vx, vy, vz
    acceleration: Tuple[float, float, float]  # ax, ay, az
    orientation: Tuple[float, float, float]  # roll, pitch, yaw
    angular_velocity: Tuple[float, float, float]  # wx, wy, wz


@dataclass
class EnvironmentState:
    """Current state of the driving environment."""
    timestamp: float
    ego_vehicle: VehicleState
    other_vehicles: List[Dict[str, Any]]
    pedestrians: List[Dict[str, Any]]
    traffic_lights: List[Dict[str, Any]]
    lane_info: Dict[str, Any]
    weather_conditions: Dict[str, Any]


class MessageBus:
    """Central communication hub for inter-agent messaging."""

    def __init__(self):
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.message_queues: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()

    def register_agent(self, agent: 'BaseAgent') -> None:
        """Register an agent with the message bus."""
        with self.lock:
            self.agents[agent.agent_id] = agent
            self.message_queues[agent.agent_id] = queue.Queue()

    def send_message(self, message: AgentMessage) -> None:
        """Send a message to a specific agent."""
        if message.receiver_id in self.message_queues:
            self.message_queues[message.receiver_id].put(message)
        else:
            print(f"Warning: No agent registered with ID {message.receiver_id}")

    def broadcast_message(self, message: AgentMessage, exclude_sender: bool = True) -> None:
        """Broadcast a message to all registered agents."""
        with self.lock:
            for agent_id in self.agents:
                if exclude_sender and agent_id == message.sender_id:
                    continue
                self.send_message(AgentMessage(
                    sender_id=message.sender_id,
                    receiver_id=agent_id,
                    message_type=message.message_type,
                    timestamp=message.timestamp,
                    data=message.data,
                    priority=message.priority
                ))

    def get_messages(self, agent_id: str) -> List[AgentMessage]:
        """Get all pending messages for an agent."""
        messages = []
        if agent_id in self.message_queues:
            while not self.message_queues[agent_id].empty():
                messages.append(self.message_queues[agent_id].get())
        return messages


class BaseAgent(ABC):
    """Base class for all autonomous driving agents."""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.state = AgentState.INITIALIZING
        self.last_update_time = time.time()
        self.message_bus.register_agent(self)

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the agent. Return True if successful."""
        pass

    @abstractmethod
    def process(self, environment_state: EnvironmentState) -> None:
        """Process the current environment state and perform agent-specific tasks."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return the current status of the agent."""
        pass

    def send_message(self, receiver_id: str, message_type: MessageType,
                    data: Dict[str, Any], priority: int = 1) -> None:
        """Send a message to another agent."""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            timestamp=time.time(),
            data=data,
            priority=priority
        )
        self.message_bus.send_message(message)

    def broadcast_message(self, message_type: MessageType,
                         data: Dict[str, Any], priority: int = 1) -> None:
        """Broadcast a message to all other agents."""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id="",  # Will be set for each recipient
            message_type=message_type,
            timestamp=time.time(),
            data=data,
            priority=priority
        )
        self.message_bus.broadcast_message(message)

    def receive_messages(self) -> List[AgentMessage]:
        """Receive pending messages."""
        return self.message_bus.get_messages(self.agent_id)

    def shutdown(self) -> None:
        """Shutdown the agent."""
        self.state = AgentState.SHUTDOWN
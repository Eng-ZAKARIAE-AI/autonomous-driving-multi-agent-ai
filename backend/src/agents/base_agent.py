"""Abstract base class for all multi-agents.

Each concrete agent receives the shared Blackboard and the config dict,
implements ``run()``, and publishes its output back to the blackboard.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from backend.src.agents.blackboard import Blackboard


class BaseAgent(ABC):
    """Common interface for Perception, Decision, Planning, Control, Safety agents."""

    def __init__(self, blackboard: Blackboard, config: Dict[str, Any]) -> None:
        self.blackboard = blackboard
        self.config = config

    @abstractmethod
    def run(self, **kwargs: Any) -> None:
        """Execute one step and publish output to the blackboard."""

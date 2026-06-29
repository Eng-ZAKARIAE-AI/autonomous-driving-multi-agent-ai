"""Thread-safe Blackboard for inter-agent communication.

All five agents read/write through this single shared object.
The Orchestrator calls agents sequentially each CARLA tick; only the
sensor watchdog thread writes concurrently (to safety.sensor_fault only).

Slots
-----
perception  : YOLO detections + CARLA-API lane geometry
decision    : FSM state + target speed
planning    : local waypoint trajectory
control     : PID-computed vehicle commands
safety      : TTC, veto flag, sensor-fault flag, intervention log
rl_policy   : reserved — future PPO lane-change module (always None for now)
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Typed slot definitions
# ---------------------------------------------------------------------------

@dataclass
class PerceptionState:
    detections: List[Dict[str, Any]] = field(default_factory=list)
    # Detections dropped by camera_geometry (non-ground class, ray above horizon,
    # intrinsics mismatch). Hook for Safety Agent (step 6) to inspect anomalies.
    rejected_detections: List[Dict[str, Any]] = field(default_factory=list)
    # CARLA ground-truth waypoints — NOT visual lane detection (assumed simplification)
    lane_geometry: Dict[str, Any] = field(default_factory=dict)
    # Traffic light ground-truth (CarlaTrafficLightPerception)
    # state  : "Red" | "Yellow" | "Green" | "Off" | "None"
    # dist_m : distance to stop line in metres (inf = no TL in range)
    traffic_light_state: str = "None"
    traffic_light_dist_m: float = float("inf")
    # Stop sign ground-truth (CarlaStopSignPerception)
    # dist_m : distance to stop line in metres (inf = no stop sign in range)
    stop_sign_dist_m: float = float("inf")
    timestamp: float = 0.0


@dataclass
class DecisionState:
    fsm_state: str = "FOLLOW_LANE"   # FOLLOW_LANE | SLOW_DOWN | STOP | YIELD | EMERGENCY
    target_speed: float = 16.0       # km/h — override pour PlanningAgent
    branch_heading: Optional[float] = None  # deg — cap souhaite au carrefour
                                             # None = pas de carrefour / pas de FSM active
    is_at_junction: bool = False     # True pendant traversee carrefour
                                     # -> SafetyAgent adapte son seuil TTC en carrefour
    reason: str = ""
    timestamp: float = 0.0


@dataclass
class PlanningState:
    # Each waypoint: (x_world, y_world, target_speed_kmh)
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)
    junction_in_chain: bool = False            # True si un wp.is_junction dans la trajectoire
                                              # DIAGNOSTIC SEULEMENT — non utilise par FSM
    junction_distance_m: float = float("inf") # distance premier wp is_junction — diagnostic
    branch_in_chain: bool = False             # True si len(nexts)>1 dans la trajectoire
                                              # SIGNAL FSM (vrai fork, pas just is_junction)
    branch_distance_m: float = float("inf")  # distance au premier vrai fork
    route_complete: bool = False              # True quand ego arrive a destination (GlobalPlanner)
    timestamp: float = 0.0


@dataclass
class RouteState:
    """Itineraire A->B calcule par GlobalPlannerAgent. Inactif par defaut."""
    # Liste de (carla.Location, RoadOption) issue de trace_route().
    # Stocke les Locations (pas les Waypoints complets) pour allegement memoire.
    route: list = field(default_factory=list)
    destination: Any = None   # carla.Location
    active: bool = False      # True = route calculee, PlanningAgent doit la suivre


@dataclass
class ControlState:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    timestamp: float = 0.0


@dataclass
class SafetyState:
    override: bool = False
    ttc: float = float("inf")       # Time-To-Collision in seconds
    reason: str = ""
    sensor_fault: bool = False
    interventions_count: int = 0
    interventions_log: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class RLPolicyState:
    """Reserved slot for a future PPO lane-change module.
    No agent writes here yet; the interface is kept open by design."""
    action: Optional[str] = None    # e.g. "change_lane_left"
    confidence: float = 0.0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------

class Blackboard:
    """Central shared state for the multi-agent pipeline.

    Thread-safety model
    -------------------
    The main pipeline (Perception → Decision → Planning → Control → Safety)
    runs sequentially inside the CARLA tick loop — no lock contention there.

    The only concurrent writer is SensorWatchdog, which writes exclusively to
    ``safety.sensor_fault``. This is covered by the single ``_lock``.

    Usage
    -----
    bb = Blackboard()
    bb.perception.detections = [...]
    bb.publish_perception(state)   # preferred: atomic replace + timestamp
    state = bb.perception           # direct read (safe in sequential context)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.perception = PerceptionState()
        self.decision = DecisionState()
        self.planning = PlanningState()
        self.control = ControlState()
        self.safety = SafetyState()
        self.rl_policy = RLPolicyState()
        self.route = RouteState()          # itineraire A->B (inactif par defaut)

    # ------------------------------------------------------------------
    # Atomic publish helpers (set timestamp + replace atomically)
    # ------------------------------------------------------------------

    def publish_perception(self, state: PerceptionState) -> None:
        state.timestamp = time.monotonic()
        with self._lock:
            self.perception = state

    def publish_decision(self, state: DecisionState) -> None:
        state.timestamp = time.monotonic()
        with self._lock:
            self.decision = state

    def publish_planning(self, state: PlanningState) -> None:
        state.timestamp = time.monotonic()
        with self._lock:
            self.planning = state

    def publish_control(self, state: ControlState) -> None:
        state.timestamp = time.monotonic()
        with self._lock:
            self.control = state

    def publish_safety(self, state: SafetyState) -> None:
        state.timestamp = time.monotonic()
        with self._lock:
            self.safety = state

    def publish_route(self, state: "RouteState") -> None:
        """Publie l'itineraire global calcule par GlobalPlannerAgent (une seule fois)."""
        with self._lock:
            self.route = state

    # ------------------------------------------------------------------
    # Safety-specific helpers (called from watchdog thread)
    # ------------------------------------------------------------------

    def set_sensor_fault(self, fault: bool) -> None:
        """Called exclusively by SensorWatchdog thread."""
        with self._lock:
            self.safety.sensor_fault = fault

    def log_safety_intervention(self, reason: str, ttc: float) -> None:
        entry = {
            "timestamp": time.monotonic(),
            "reason": reason,
            "ttc": ttc,
        }
        with self._lock:
            self.safety.interventions_count += 1
            self.safety.interventions_log.append(entry)

    # ------------------------------------------------------------------
    # Snapshot for telemetry (serialisable dict for WebSocket broadcast)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "perception": {
                    "detections_count": len(self.perception.detections),
                    "detections": self.perception.detections,
                    "rejected_detections_count": len(self.perception.rejected_detections),
                    "lane_geometry": self.perception.lane_geometry,
                    "ts": self.perception.timestamp,
                },
                "decision": {
                    "fsm_state": self.decision.fsm_state,
                    "target_speed": self.decision.target_speed,
                    "branch_heading": self.decision.branch_heading,
                    "is_at_junction": self.decision.is_at_junction,
                    "reason": self.decision.reason,
                    "ts": self.decision.timestamp,
                },
                "planning": {
                    "waypoints_count": len(self.planning.waypoints),
                    "junction_in_chain": self.planning.junction_in_chain,
                    "junction_distance_m": round(self.planning.junction_distance_m, 1),
                    "branch_in_chain": self.planning.branch_in_chain,
                    "branch_distance_m": round(self.planning.branch_distance_m, 1),
                    "ts": self.planning.timestamp,
                },
                "control": {
                    "steer": round(self.control.steer, 4),
                    "throttle": round(self.control.throttle, 4),
                    "brake": round(self.control.brake, 4),
                    "ts": self.control.timestamp,
                },
                "safety": {
                    "override": self.safety.override,
                    "ttc": round(self.safety.ttc, 2),
                    "reason": self.safety.reason,
                    "sensor_fault": self.safety.sensor_fault,
                    "interventions_count": self.safety.interventions_count,
                    "ts": self.safety.timestamp,
                },
            }

    def reset_episode(self) -> None:
        """Reset per-episode state; keep cumulative intervention count."""
        with self._lock:
            prev_count = self.safety.interventions_count
            prev_log = self.safety.interventions_log

            self.perception = PerceptionState()
            self.decision = DecisionState()
            self.planning = PlanningState()
            self.control = ControlState()
            self.safety = SafetyState()
            self.rl_policy = RLPolicyState()

            self.safety.interventions_count = prev_count
            self.safety.interventions_log = prev_log


# ---------------------------------------------------------------------------
# Sensor watchdog (the only component allowed to run on a separate thread)
# ---------------------------------------------------------------------------

class SensorWatchdog:
    """Monitors camera frame freshness in a background daemon thread.

    Writes only to ``blackboard.safety.sensor_fault`` (via the lock-protected
    ``set_sensor_fault`` method).  No other blackboard field is touched.

    Parameters
    ----------
    blackboard        : shared Blackboard instance
    max_gap_seconds   : declare a fault if the last perception timestamp is
                        older than this value (default: 3 × fixed_delta = 0.15s)
    poll_interval     : how often the watchdog checks (seconds)
    """

    def __init__(
        self,
        blackboard: Blackboard,
        max_gap_seconds: float = 0.15,
        poll_interval: float = 0.05,
    ) -> None:
        self._bb = blackboard
        self._max_gap = max_gap_seconds
        self._poll = poll_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sensor-watchdog")

    def start(self) -> None:
        self._stop_event.clear()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            last_ts = self._bb.perception.timestamp
            gap = time.monotonic() - last_ts if last_ts > 0 else 0.0
            fault = gap > self._max_gap and last_ts > 0
            self._bb.set_sensor_fault(fault)
            self._stop_event.wait(self._poll)

"""Perception Agent — YOLO detection + distance estimation + lane geometry.

Reused without modification
---------------------------
  backend.src.perception.yolo_detector.PerceptionModule
      detect(frame) -> List[Detection dicts]
  backend.src.perception.camera_geometry
      CameraIntrinsics, intrinsics_from_blueprint_attrs,
      estimate_ground_distance, assert_frame_matches_intrinsics
  backend.src.agents.blackboard
      Blackboard, PerceptionState, publish_perception()
  backend.src.agents.base_agent.BaseAgent

New in this module
------------------
  LaneDetector (ABC)
      Interface for lane geometry extraction.
      Extension point: implement a subclass (e.g. UFLDv2LaneDetector) and
      pass it as lane_detector= to PerceptionAgent to swap in visual detection
      without modifying this file.
  CarlaWaypointLaneDetector(LaneDetector)
      Current implementation. Uses CARLA Waypoint API — zero VRAM, zero latency.
      NOT visual lane detection (assumed simplification, documented in blackboard.py).
  PerceptionAgent(BaseAgent)
      Orchestrates YOLO + distance estimation + lane detection + blackboard publish.
      Logs every rejected detection so the Safety Agent can inspect anomalies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.src.agents.base_agent import BaseAgent
from backend.src.agents.blackboard import Blackboard, PerceptionState
from backend.src.perception.camera_geometry import (
    CameraIntrinsics,
    assert_frame_matches_intrinsics,
    estimate_ground_distance,
)
from backend.src.perception.yolo_detector import PerceptionModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lane detector interface + CARLA implementation
# ---------------------------------------------------------------------------

class LaneDetector(ABC):
    """Interface for lane geometry providers.

    Extension point
    ---------------
    The current implementation (CarlaWaypointLaneDetector) reads ground-truth
    lane data from the CARLA Waypoint API.  To add visual lane detection
    (UFLDv2, LaneATT, etc.) without touching PerceptionAgent:

        class UFLDv2LaneDetector(LaneDetector):
            def detect_lane(self, frame, ego_state):
                ...  # run visual model
                return {same keys as CarlaWaypointLaneDetector}

        agent = PerceptionAgent(bb, cfg, lane_detector=UFLDv2LaneDetector(...))

    The dict contract (all keys optional, missing keys default to None downstream):
        ego_position       : (x_world, y_world) metres
        ego_heading_deg    : float — vehicle yaw in degrees
        lane_width         : float — metres
        left_marking       : str   — "broken" | "solid" | "unknown"
        right_marking      : str   — "broken" | "solid" | "unknown"
        waypoints_ahead    : List[(x, y, yaw_deg)] — next N waypoints
        speed_limit_kmh    : float | None
        is_junction        : bool
    """

    @abstractmethod
    def detect_lane(
        self,
        frame: Optional[np.ndarray],
        ego_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a lane_geometry dict (see class docstring for keys)."""


class CarlaWaypointLaneDetector(LaneDetector):
    """Lane geometry from the CARLA Waypoint API.

    NOT visual lane detection — uses CARLA simulator ground truth.
    Assumed simplification: zero VRAM cost, zero inference latency.
    See LaneDetector (above) for the visual detection extension point.

    Parameters
    ----------
    carla_map : carla.Map | None
        Pass world.get_map() after connecting to CARLA.
        If None (CARLA not connected) all calls return {}.
    lookahead : int
        Number of waypoints to include in waypoints_ahead (default 10).
    spacing   : float
        Metres between consecutive waypoints (default 3.0 m).
    """

    # Map CARLA LaneMarkingType enum values to readable strings
    _MARKING_MAP = {
        "Broken": "broken",
        "Solid": "solid",
        "SolidSolid": "solid",
        "BrokenSolid": "broken",
        "SolidBroken": "solid",
        "BrokenBroken": "broken",
        "Curb": "solid",
        "None": "unknown",
        "Other": "unknown",
        "NONE": "unknown",
    }

    def __init__(
        self,
        carla_map=None,
        lookahead: int = 10,
        spacing: float = 3.0,
    ) -> None:
        self._map = carla_map
        self._lookahead = lookahead
        self._spacing = spacing

    def detect_lane(
        self,
        frame: Optional[np.ndarray],
        ego_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._map is None or not ego_state:
            return {}

        location = ego_state.get("location")
        rotation = ego_state.get("rotation")
        if location is None:
            return {}

        try:
            waypoint = self._map.get_waypoint(location, project_to_road=True)
            if waypoint is None:
                return {}

            # Next waypoints ahead
            next_wps: List[Tuple[float, float, float]] = []
            wp = waypoint
            for _ in range(self._lookahead):
                nexts = wp.next(self._spacing)
                if not nexts:
                    break
                wp = nexts[0]
                t = wp.transform
                next_wps.append((
                    t.location.x,
                    t.location.y,
                    t.rotation.yaw,
                ))

            left_type = str(waypoint.left_lane_marking.type) if waypoint.left_lane_marking else "NONE"
            right_type = str(waypoint.right_lane_marking.type) if waypoint.right_lane_marking else "NONE"

            ego_x = location.x if hasattr(location, 'x') else location[0]
            ego_y = location.y if hasattr(location, 'y') else location[1]
            ego_yaw = rotation.yaw if rotation is not None and hasattr(rotation, 'yaw') else 0.0

            speed_limit = None
            try:
                speed_limit = float(waypoint.get_landmarks_of_type(
                    self._spacing * self._lookahead, "274"
                )[0].value) if hasattr(waypoint, 'get_landmarks_of_type') else None
            except (IndexError, AttributeError, ValueError):
                pass

            return {
                "ego_position": (ego_x, ego_y),
                "ego_heading_deg": ego_yaw,
                "lane_width": float(waypoint.lane_width),
                "left_marking": self._MARKING_MAP.get(left_type, "unknown"),
                "right_marking": self._MARKING_MAP.get(right_type, "unknown"),
                "waypoints_ahead": next_wps,
                "speed_limit_kmh": speed_limit,
                "is_junction": bool(waypoint.is_junction),
            }

        except Exception as exc:
            logger.warning("CarlaWaypointLaneDetector error: %s", exc)
            return {}


# ---------------------------------------------------------------------------
# Perception Agent
# ---------------------------------------------------------------------------

class PerceptionAgent(BaseAgent):
    """Orchestrates YOLO detection, distance estimation, and lane geometry.

    Pipeline per tick
    -----------------
    1. Validate frame (None or wrong shape -> log + publish empty state)
    2. Run YOLO (PerceptionModule.detect)
    3. Enrich each detection with forward_dist / lateral_offset
       (camera_geometry.estimate_ground_distance)
       Rejections logged to rejected_detections with explicit reason
    4. Get lane geometry (LaneDetector.detect_lane)
    5. Publish PerceptionState to blackboard

    Parameters
    ----------
    blackboard    : shared Blackboard
    config        : full config dict
    lane_detector : LaneDetector instance (default: CarlaWaypointLaneDetector(map=None))
    intrinsics    : CameraIntrinsics (default: built from perception_camera config)
    camera_height : metres above road surface (default: 2.4 m from carla_env.py mount)
    """

    def __init__(
        self,
        blackboard: Blackboard,
        config: Dict[str, Any],
        lane_detector: Optional[LaneDetector] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        camera_height: float = 2.4,
    ) -> None:
        super().__init__(blackboard, config)

        # YOLO module — reused unchanged from yolo_detector.py
        self._yolo = PerceptionModule(config)

        # Camera intrinsics — built from config at init; override with
        # intrinsics_from_blueprint_attrs(sensor.attributes) once CARLA is connected
        if intrinsics is None:
            cam = config.get("perception_camera", config.get("camera", {}))
            intrinsics = CameraIntrinsics(
                frame_width=int(cam.get("width", 640)),
                frame_height=int(cam.get("height", 640)),
                fov_deg=float(cam.get("fov", 90.0)),
            )
        self.intrinsics = intrinsics
        self.camera_height = camera_height

        # Lane detector — defaults to CARLA API (no map yet; set later)
        self._lane_detector: LaneDetector = (
            lane_detector if lane_detector is not None
            else CarlaWaypointLaneDetector(carla_map=None)
        )

        # LiDAR extension point — not implemented in the multi-agent pipeline.
        # TransFuser baseline uses LiDAR independently (see baseline_agent.py).
        # To add LiDAR to the perception pipeline: pass lidar_sensor= here and
        # build a BEV occupancy map; publish it as an extra key in lane_geometry.
        self._lidar_sensor = None  # type: ignore[assignment]

        # First-frame flag for intrinsics coherence check
        self._first_frame = True

    def set_carla_map(self, carla_map: Any) -> None:
        """Called after CARLA world is ready; wires the lane detector to the map."""
        if isinstance(self._lane_detector, CarlaWaypointLaneDetector):
            self._lane_detector._map = carla_map

    def update_intrinsics_from_sensor(self, sensor_attributes: Any) -> None:
        """Update intrinsics from live CARLA sensor.attributes (call after spawn)."""
        from backend.src.perception.camera_geometry import intrinsics_from_blueprint_attrs
        self.intrinsics = intrinsics_from_blueprint_attrs(sensor_attributes)
        self._first_frame = True  # re-check shape on next frame
        logger.info("Intrinsics updated from sensor: %s", self.intrinsics)

    # ------------------------------------------------------------------
    # Main pipeline step — called every tick by the Orchestrator
    # ------------------------------------------------------------------

    def run(
        self,
        frame: Optional[np.ndarray] = None,
        ego_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run one perception tick and publish to blackboard.

        Parameters
        ----------
        frame     : H x W x 3 uint8 RGB numpy array from CARLA camera callback.
                    None if the camera has not sent a frame yet (sensor fault).
        ego_state : dict with 'location' and 'rotation' (carla objects or plain dicts).
                    Used by the lane detector. None if CARLA not yet connected.
        """
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        # --- 1. Frame validation ---
        if frame is None:
            logger.warning("[PerceptionAgent] frame=None — camera not ready or sensor fault")
            self._publish(accepted, rejected, ego_state)
            return

        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            logger.error(
                "[PerceptionAgent] Invalid frame shape %s — expected (H,W,3) uint8",
                getattr(frame, 'shape', type(frame)),
            )
            self._publish(accepted, rejected, ego_state)
            return

        # Intrinsics coherence check on first real frame
        if self._first_frame:
            try:
                assert_frame_matches_intrinsics(frame.shape, self.intrinsics, "PerceptionAgent")
                self._first_frame = False
            except AssertionError as exc:
                logger.error("[PerceptionAgent] Intrinsics mismatch: %s", exc)
                rejected.append({
                    "reason": "intrinsics_mismatch",
                    "detail": str(exc),
                })
                self._publish(accepted, rejected, ego_state)
                return

        # --- 2. YOLO detection ---
        raw_detections = self._yolo.detect(frame)

        # --- 3. Distance enrichment ---
        for det in raw_detections:
            enriched, reject_reason = self._enrich_detection(det)
            if reject_reason is None:
                accepted.append(enriched)
            else:
                rejected_entry = dict(det)
                rejected_entry["reason"] = reject_reason
                rejected.append(rejected_entry)
                logger.debug(
                    "[PerceptionAgent] Detection rejected — class=%s reason=%s",
                    det.get("class_name"), reject_reason,
                )

        # --- 4 + 5. Lane + publish ---
        self._publish(accepted, rejected, ego_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enrich_detection(
        self,
        det: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Add forward_dist and lateral_offset; return (enriched, reject_reason).

        reject_reason is None when distance was computed successfully, or when
        the class is non-ground (traffic_light, stop_sign) — those are kept in
        accepted with forward_dist=None (non-ground is informative, not a fault).
        reject_reason is a string only for geometry errors (ray above horizon).
        """
        enriched = dict(det)
        result = estimate_ground_distance(
            bbox=tuple(det["bbox"]),
            class_name=det.get("class_name", ""),
            intrinsics=self.intrinsics,
            camera_height=self.camera_height,
        )

        if result is not None:
            enriched["forward_dist"] = round(result[0], 2)
            enriched["lateral_offset"] = round(result[1], 2)
            return enriched, None

        # estimate_ground_distance returned None — two causes:
        # (a) non-ground class (traffic_light, stop_sign) — keep, dist = None
        # (b) ray above horizon (bbox top half of frame) — reject + log
        from backend.src.perception.camera_geometry import _GROUND_LEVEL_CLASSES
        if det.get("class_name", "") not in _GROUND_LEVEL_CLASSES:
            enriched["forward_dist"] = None
            enriched["lateral_offset"] = None
            return enriched, None  # accepted, no distance expected

        # Ground-level class but ray above horizon — genuine geometry failure
        enriched["forward_dist"] = None
        enriched["lateral_offset"] = None
        return enriched, "ray_above_horizon"

    def _publish(
        self,
        accepted: List[Dict[str, Any]],
        rejected: List[Dict[str, Any]],
        ego_state: Optional[Dict[str, Any]],
    ) -> None:
        lane_geo = self._lane_detector.detect_lane(None, ego_state or {})
        state = PerceptionState(
            detections=accepted,
            rejected_detections=rejected,
            lane_geometry=lane_geo,
        )
        self.blackboard.publish_perception(state)

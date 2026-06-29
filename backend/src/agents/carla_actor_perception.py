"""CarlaActorPerception -- ground-truth obstacle distances for Safety Agent testing.

Replaces YOLO when testing Safety logic without a camera.
Uses world.get_actors() + 2-D projection to compute forward_dist / lateral_offset
and publishes to blackboard.perception.detections in the PerceptionAgent-compatible format.

Filters applied every tick
--------------------------
1. Skip ego itself (actor.id == ego.id).
2. forward_dist < 0       -> actor is BEHIND ego -> excluded (no collision risk).
3. forward_dist > max_range -> too far -> excluded.
4. |lateral_offset| > lane_half_width_m -> actor not on ego's lane -> excluded.
   Prevents false-positive braking for vehicles on adjacent lanes.

Projection math (CARLA left-handed XY, yaw=0 -> +X East, yaw=90 -> +Y South)
-------------------------------------------------------------------------------
  fwd_x = cos(yaw_rad)       fwd_y = sin(yaw_rad)
  forward_dist   = dx * fwd_x + dy * fwd_y      (dot product, positive = ahead)
  lateral_offset = dx * fwd_y - dy * fwd_x      (2-D cross, positive = actor to LEFT)

These are consistent with the CTE formula (ex*dy - ey*dx) / seg_len used in ControlAgent.

Compatibility
-------------
Each detection dict uses the same keys as PerceptionAgent._enrich_detection():
  class_name, confidence, forward_dist, lateral_offset, bbox
SafetyAgent will read these keys identically regardless of the detection source.
"""

import math
import time
from typing import Any, List

from backend.src.agents.blackboard import Blackboard, PerceptionState


class CarlaActorPerception:
    """Ground-truth CARLA actor distances published to blackboard.perception.

    Designed for Safety Agent testing without YOLO / camera infrastructure.
    Works in synchronous CARLA mode: call run() once per tick before other agents.

    Parameters
    ----------
    world             : carla.World
    ego               : carla.Vehicle (ego actor, excluded from detections)
    blackboard        : shared Blackboard
    lane_half_width_m : half-width of lane for lateral filter (default 1.5 m).
                        CARLA lane width is typically 3.0-3.5 m -> half ~1.5 m.
                        Increase to 2.0 if adjacent-lane vehicles are falsely excluded.
    max_range_m       : maximum forward distance to consider (default 50 m).
    """

    def __init__(
        self,
        world: Any,
        ego: Any,
        blackboard: Blackboard,
        lane_half_width_m: float = 1.5,
        max_range_m: float = 50.0,
    ) -> None:
        self._world = world
        self._ego = ego
        self._bb = blackboard
        self._lane_half_width = lane_half_width_m
        self._max_range_m = max_range_m

    # ------------------------------------------------------------------
    # Main tick method
    # ------------------------------------------------------------------

    def run(
        self,
        ego_x: float,
        ego_y: float,
        ego_heading_deg: float,
        tick: int,
    ) -> None:
        """Query CARLA actors, apply filters, publish to blackboard, print [PERCEPTION]."""
        yaw_rad = math.radians(ego_heading_deg)
        fwd_x = math.cos(yaw_rad)
        fwd_y = math.sin(yaw_rad)

        actors = self._world.get_actors()
        all_actors: List[Any] = (
            list(actors.filter("vehicle.*")) + list(actors.filter("walker.*"))
        )

        total_seen = 0
        on_lane: List[dict] = []

        for actor in all_actors:
            if actor.id == self._ego.id:
                continue

            loc = actor.get_location()
            dx = loc.x - ego_x
            dy = loc.y - ego_y
            total_seen += 1

            forward_dist   = dx * fwd_x + dy * fwd_y      # positive = ahead
            lateral_offset = dx * fwd_y - dy * fwd_x      # positive = left of ego

            # --- Filters ---
            if forward_dist < 0.0:
                continue                                   # behind ego
            if forward_dist > self._max_range_m:
                continue                                   # too far
            if abs(lateral_offset) > self._lane_half_width:
                continue                                   # adjacent lane

            class_name = "vehicle" if "vehicle" in actor.type_id else "walker"

            # Vitesse de l'acteur (pour car-following, Scénario 3)
            try:
                vel = actor.get_velocity()
                v_lead_kmh = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
            except Exception:
                v_lead_kmh = 0.0

            on_lane.append({
                "class_name":     class_name,
                "confidence":     1.0,
                "forward_dist":   round(forward_dist, 2),
                "lateral_offset": round(lateral_offset, 2),
                "v_lead_kmh":     round(v_lead_kmh, 1),
                "bbox":           None,
                "actor_id":       actor.id,
            })

        # Sort by proximity
        on_lane.sort(key=lambda d: d["forward_dist"])

        # Publish to blackboard
        state = PerceptionState(
            detections=on_lane,
            rejected_detections=[],
            lane_geometry={},
        )
        self._bb.publish_perception(state)

        # --- [PERCEPTION] log — unconditional, every tick ---
        if on_lane:
            c = on_lane[0]
            print(
                f"[PERCEPTION] t={tick:>4}  total={total_seen}  on_lane={len(on_lane)}"
                f"  closest: {c['class_name']}"
                f"  fwd={c['forward_dist']:>6.2f}m"
                f"  lat={c['lateral_offset']:>+6.2f}m"
            )
        else:
            print(
                f"[PERCEPTION] t={tick:>4}  total={total_seen}  on_lane=0"
                f"  (aucun acteur devant sur voie)"
            )

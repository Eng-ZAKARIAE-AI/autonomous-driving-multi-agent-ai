"""CarlaStopSignPerception -- stop sign ground-truth pour la FSM.

Publie dans blackboard.perception :
  stop_sign_dist_m : distance a la ligne d'arret (m) — inf si aucun panneau en portee

Source unique : landmarks OpenDRIVE type '206' (Zeichen 206 = panneau STOP).
  wp.get_landmarks(horizon_m) suit la courbure de la route.
  Pas d'acteur scan : les stop signs n'ont pas d'etat dynamique.

Architecture
------------
Appeler run() APRES CarlaTrafficLightPerception dans la boucle tick (troisieme
dans la chaine perception). Preserve tous les champs existants du PerceptionState
et ajoute stop_sign_dist_m.

Log [STOP]
----------
  Si stop sign en portee : tous les ticks.
  Sinon                  : tous les LOG_EVERY ticks.
  Format : [STOP] t=  120  dist=38.2m  src=landmark
"""

import math
from typing import Any

from backend.src.agents.blackboard import Blackboard, PerceptionState

_STOP_LM_TYPE = "206"
_LOG_EVERY = 40


class CarlaStopSignPerception:
    """Detection ground-truth des panneaux STOP via landmarks OpenDRIVE.

    Parameters
    ----------
    world      : carla.World
    ego        : carla.Vehicle (ego)
    blackboard : Blackboard partage
    horizon_m  : portee de recherche (defaut 60 m)
    """

    def __init__(
        self,
        world: Any,
        ego: Any,
        blackboard: Blackboard,
        horizon_m: float = 60.0,
    ) -> None:
        self._world     = world
        self._ego       = ego
        self._bb        = blackboard
        self._horizon_m = horizon_m
        self._carla_map = world.get_map()

    # ------------------------------------------------------------------

    def run(self, ego_x: float, ego_y: float, ego_heading_deg: float, tick: int) -> None:
        """Calcule la distance au prochain stop sign et met a jour le blackboard."""
        dist_m = float("inf")
        source = ""

        try:
            import carla
            wp = self._carla_map.get_waypoint(
                self._ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            landmarks = wp.get_landmarks(self._horizon_m, stop_at_junction=False)
            stop_lms = [lm for lm in landmarks if str(lm.type) == _STOP_LM_TYPE]
            if stop_lms:
                nearest = min(stop_lms, key=lambda lm: lm.distance)
                dist_m  = nearest.distance
                source  = f"landmark({nearest.name})"
        except Exception:
            pass

        # Preserve tous les champs existants, mise a jour stop_sign_dist_m uniquement
        cur = self._bb.perception
        updated = PerceptionState(
            detections           = cur.detections,
            rejected_detections  = cur.rejected_detections,
            lane_geometry        = cur.lane_geometry,
            traffic_light_state  = cur.traffic_light_state,
            traffic_light_dist_m = cur.traffic_light_dist_m,
            stop_sign_dist_m     = round(dist_m, 1) if dist_m < float("inf") else float("inf"),
        )
        self._bb.publish_perception(updated)

        # Log
        visible = dist_m < float("inf")
        if visible or (tick % _LOG_EVERY == 0):
            dist_str = f"{dist_m:.1f}m" if visible else "  inf"
            src_str  = f"  src={source}" if source else ""
            print(f"[STOP] t={tick:>4}  dist={dist_str}{src_str}")

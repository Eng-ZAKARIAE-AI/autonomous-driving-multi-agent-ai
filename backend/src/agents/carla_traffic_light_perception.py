"""CarlaTrafficLightPerception — feux tricolores ground-truth pour la FSM.

Publie dans blackboard.perception :
  traffic_light_state  : "Red" | "Yellow" | "Green" | "Off" | "None"
  traffic_light_dist_m : distance à la ligne d'arrêt (m) — inf si aucun feu en portée

Deux sources complémentaires
-----------------------------
1. Landmarks OpenDRIVE (type=1000001)
   waypoint.get_landmarks(horizon_m) — suit la courbure de la route.
   → Donne la DISTANCE de la ligne d'arrêt tôt et précisément (jusqu'à horizon_m).
   → Ne donne PAS l'état du feu (val=-1.0 toujours).

2. Scan des acteurs TL
   world.get_actors().filter("traffic.traffic_light*") + projection forward/lateral.
   → Donne l'ÉTAT du feu (Red/Green/Yellow) à n'importe quelle portée.
   → Effective quand l'ego est aligné avec l'axe du feu (lateral_max_m).

3. Hook is_at_traffic_light() (confirmation proche)
   ego.is_at_traffic_light() + ego.get_traffic_light().get_state()
   → Toujours correct dans la zone de déclenchement CARLA (~10m avant la ligne).

Fusion des sources
------------------
  dist_m  = landmark si disponible, sinon projection acteur
  state   = acteur si visible, sinon is_at_traffic_light(), sinon "None"
  Si dist_m connu ET state != "None" → information complète pour la FSM.

Architecture
------------
Appeler run() APRÈS CarlaActorPerception dans la boucle tick (ajoute les champs TL
au PerceptionState existant sans écraser les détections d'obstacles).

Log [TL]
--------
  Toutes les ticks si feu visible.
  Tous les LOG_EVERY ticks si aucun feu.
  Format : [TL] t= 270  state=Red      dist=11.3m  src=landmark+actor
"""

import math
from typing import Any, Optional

from backend.src.agents.blackboard import Blackboard, PerceptionState

_TL_LANDMARK_TYPE = "1000001"
_LOG_EVERY = 20          # ticks sans feu visible → log éclairci
_NO_TL_HORIZON_M = 80.0  # horizon de recherche landmarks


def _state_name(tl_state_obj: Any) -> str:
    """Extrait le nom lisible depuis carla.TrafficLightState."""
    if hasattr(tl_state_obj, "name"):
        return tl_state_obj.name          # carla.TrafficLightState.Red → "Red"
    raw = str(tl_state_obj)
    return raw.split(".")[-1] if "." in raw else raw


class CarlaTrafficLightPerception:
    """Détection ground-truth des feux tricolores CARLA.

    Paramètres
    ----------
    world         : carla.World
    ego           : carla.Vehicle (ego)
    blackboard    : Blackboard partagé
    horizon_m     : portée de détection par landmarks (défaut 80m)
    lateral_max_m : filtre latéral pour scan acteurs (défaut 20m)
                    Large par défaut car les mâts de feux sont souvent hors voie.
    """

    def __init__(
        self,
        world: Any,
        ego: Any,
        blackboard: Blackboard,
        horizon_m: float = _NO_TL_HORIZON_M,
        lateral_max_m: float = 20.0,
    ) -> None:
        self._world        = world
        self._ego          = ego
        self._bb           = blackboard
        self._horizon_m    = horizon_m
        self._lateral_max  = lateral_max_m
        self._carla_map    = world.get_map()

    # ------------------------------------------------------------------

    def run(self, ego_x: float, ego_y: float, ego_heading_deg: float, tick: int) -> None:
        """Calcule l'état + distance du prochain feu et met à jour le blackboard."""
        yaw_rad = math.radians(ego_heading_deg)
        fwd_x   = math.cos(yaw_rad)
        fwd_y   = math.sin(yaw_rad)

        dist_m = float("inf")
        state  = "None"
        source = ""

        # ------------------------------------------------------------------
        # Source 1 : landmarks → distance stop-line
        # ------------------------------------------------------------------
        try:
            import carla
            wp = self._carla_map.get_waypoint(
                self._ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            landmarks = wp.get_landmarks(self._horizon_m, stop_at_junction=False)
            tl_lms = [lm for lm in landmarks if str(lm.type) == _TL_LANDMARK_TYPE]
            if tl_lms:
                nearest_lm = min(tl_lms, key=lambda lm: lm.distance)
                dist_m = nearest_lm.distance
                source += "landmark"
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Source 2 : scan acteurs → état (fonctionne quand ego aligné)
        # ------------------------------------------------------------------
        try:
            best_actor_dist = float("inf")
            best_actor_state = "None"
            for tl in self._world.get_actors().filter("traffic.traffic_light*"):
                loc = tl.get_location()
                dx  = loc.x - ego_x
                dy  = loc.y - ego_y
                fwd = dx * fwd_x + dy * fwd_y
                lat = abs(dx * fwd_y - dy * fwd_x)
                if 0 < fwd < self._horizon_m and lat < self._lateral_max:
                    if fwd < best_actor_dist:
                        best_actor_dist  = fwd
                        best_actor_state = _state_name(tl.get_state())

            if best_actor_state != "None":
                state = best_actor_state
                if source:
                    source += "+actor"
                else:
                    dist_m = best_actor_dist
                    source  = "actor"
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Source 3 : hook is_at_traffic_light (confirmation dans la zone)
        # ------------------------------------------------------------------
        try:
            if self._ego.is_at_traffic_light():
                close_tl = self._ego.get_traffic_light()
                if close_tl is not None:
                    state = _state_name(close_tl.get_state())
                    # Distance acteur pour affinage si landmark indisponible
                    if dist_m == float("inf"):
                        loc = close_tl.get_location()
                        dx, dy = loc.x - ego_x, loc.y - ego_y
                        dist_m = max(0.0, dx * fwd_x + dy * fwd_y)
                    source += "+hook"
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Mise à jour blackboard (ajoute TL aux champs perception existants)
        # ------------------------------------------------------------------
        cur = self._bb.perception
        updated = PerceptionState(
            detections           = cur.detections,
            rejected_detections  = cur.rejected_detections,
            lane_geometry        = cur.lane_geometry,
            traffic_light_state  = state,
            traffic_light_dist_m = round(dist_m, 1) if dist_m < float("inf") else float("inf"),
        )
        self._bb.publish_perception(updated)

        # ------------------------------------------------------------------
        # Log [TL]
        # ------------------------------------------------------------------
        tl_visible = (state != "None" or dist_m < float("inf"))
        if tl_visible or (tick % _LOG_EVERY == 0):
            dist_str = f"{dist_m:.1f}m" if dist_m < float("inf") else "  inf"
            state_pad = f"{state:<8}"
            src_str   = f"  src={source}" if source else ""
            print(
                f"[TL] t={tick:>4}  state={state_pad}  dist={dist_str}{src_str}"
            )

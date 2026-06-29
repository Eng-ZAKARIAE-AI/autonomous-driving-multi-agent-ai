"""GlobalPlannerAgent — navigation A→B via CARLA GlobalRoutePlanner.

Calcule l'itineraire UNE SEULE FOIS a l'init (avant la boucle tick),
le valide (non vide), le dumpe dans les logs, et le publie dans le
blackboard via publish_route(RouteState).

Si le calcul echoue, l'agent logue l'erreur et ne publie rien (route.active=False)
→ PlanningAgent retombe sur GoStraightPolicy.

Logs cles
---------
  [GRP] route: N waypoints, dist=Xm
  [GRP] decisions (non-LANEFOLLOW):
  [GRP]   [idx]  x=...  y=...  ->  OPTION_NAME
  [GRP] TL landmark a ?.?m (idx=N)
  [GRP] STOP landmark a ?.?m (idx=N)
"""

import logging
import math
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Chemin CARLA par defaut (machine locale).
# Override via cfg.carla.agents_path (config.yaml : carla.agents_path: "C:/...")
_DEFAULT_AGENTS_PATH = r"C:\pfa\WindowsNoEditor\PythonAPI\carla"

_TL_LM_TYPE   = "1000001"
_STOP_LM_TYPE = "206"
_LM_HORIZON_M = 15.0


def _ensure_agents_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


class GlobalPlannerAgent:
    """Calcule et publie un itineraire A→B dans le blackboard.

    Usage
    -----
        gp = GlobalPlannerAgent(world, blackboard, cfg)
        ok = gp.compute_route(ego_location, destination_location)
        if ok:
            planning_agent.set_branch_policy("route_following")
    """

    def __init__(self, world: Any, blackboard: Any, cfg: Any = None) -> None:
        self._world = world
        self._bb    = blackboard
        self._cfg   = cfg
        self._grp   = None

        agents_path = _DEFAULT_AGENTS_PATH
        if cfg is not None:
            try:
                p = cfg.carla.get("agents_path") or cfg.carla.agents_path
                if p:
                    agents_path = p
            except Exception:
                pass

        _ensure_agents_path(agents_path)

        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            self._grp = GlobalRoutePlanner(world.get_map(), sampling_resolution=2.0)
            logger.info("GlobalPlannerAgent: GlobalRoutePlanner OK (sampling=2.0m)")
            print("[GRP] GlobalRoutePlanner instancie (sampling=2.0m)")
        except Exception as exc:
            logger.error("GlobalPlannerAgent: impossible d'instancier GRP: %s", exc)
            print(f"[GRP] ERREUR: GRP non disponible: {exc}")

    # ------------------------------------------------------------------

    def compute_route(self, origin_loc: Any, destination_loc: Any) -> bool:
        """Calcule l'itineraire, valide qu'il est non vide, publie dans le blackboard.

        Returns
        -------
        True si la route est calculee et publiee, False sinon.
        """
        if self._grp is None:
            print("[GRP] ERREUR: GRP non instancie -> GoStraightPolicy")
            return False

        try:
            from agents.navigation.local_planner import RoadOption
            from backend.src.agents.blackboard import RouteState
        except ImportError as exc:
            print(f"[GRP] ERREUR import: {exc}")
            return False

        try:
            raw_route = self._grp.trace_route(origin_loc, destination_loc)
        except Exception as exc:
            print(f"[GRP] ERREUR trace_route: {exc}")
            return False

        if not raw_route:
            print("[GRP] ERREUR: trace_route a retourne une route vide -> GoStraightPolicy")
            return False

        # Convertit (Waypoint, RoadOption) → (carla.Location, RoadOption)
        route: list = [(wp.transform.location, opt) for wp, opt in raw_route]

        # Longueur totale
        dist_total = sum(
            math.hypot(
                route[i + 1][0].x - route[i][0].x,
                route[i + 1][0].y - route[i][0].y,
            )
            for i in range(len(route) - 1)
        )

        # Publie dans le blackboard
        self._bb.publish_route(RouteState(
            route=route,
            destination=destination_loc,
            active=True,
        ))

        print(f"[GRP] route: {len(route)} waypoints, dist={dist_total:.0f}m")
        logger.info("GlobalPlannerAgent: route=%d wps dist=%.0fm", len(route), dist_total)

        # Dump decisions (non-LANEFOLLOW) + landmarks TL/STOP sur la route
        self._dump_route(route, raw_route, RoadOption)

        return True

    # ------------------------------------------------------------------

    def _dump_route(self, route: list, raw_route: list, RoadOption: Any) -> None:
        """Logue les points de decision et les landmarks remarquables."""
        decisions = [
            (i, loc, opt)
            for i, (loc, opt) in enumerate(route)
            if opt != RoadOption.LANEFOLLOW
        ]
        print(f"[GRP] {len(decisions)} decisions (non-LANEFOLLOW):")
        for i, loc, opt in decisions[:20]:
            print(f"[GRP]   [{i:>4}]  x={loc.x:>8.2f}  y={loc.y:>8.2f}  ->  {opt.name}")
        if len(decisions) > 20:
            print(f"[GRP]   ... (+{len(decisions)-20} supplémentaires)")

        # Landmarks TL et STOP le long de la route (via raw_route avec Waypoints)
        print("[GRP] Landmarks sur la route:")
        found_lm: dict = {}
        cum = 0.0
        prev_loc = None
        for wp, _ in raw_route:
            loc = wp.transform.location
            if prev_loc is not None:
                cum += math.hypot(loc.x - prev_loc.x, loc.y - prev_loc.y)
            prev_loc = loc
            try:
                for lm in wp.get_landmarks(_LM_HORIZON_M, stop_at_junction=False):
                    lm_type = str(lm.type)
                    if lm_type not in (_TL_LM_TYPE, _STOP_LM_TYPE):
                        continue
                    dist_here = cum + lm.distance
                    name = "TL  " if lm_type == _TL_LM_TYPE else "STOP"
                    # Dedoublonnage grossier
                    key = (lm_type, round(dist_here, 0))
                    if key in found_lm:
                        continue
                    found_lm[key] = True
                    print(f"[GRP]   {name} landmark a {dist_here:.1f}m (wp_idx~{int(cum/2)})")
            except Exception:
                pass

        if not found_lm:
            print("[GRP]   (aucun TL/STOP landmark trouve sur le trajet)")

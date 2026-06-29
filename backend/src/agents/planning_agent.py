"""Planning Agent -- Etape 4.

Genere une trajectoire de suivi de voie (liste de waypoints + vitesse cible)
publiee dans blackboard.planning pour etre consommee par ControlAgent.

Role dans l'architecture
------------------------
Consomme  : position ego passee directement a run()
            OU blackboard.perception.lane_geometry.waypoints_ahead (fallback)
Produit   : blackboard.planning.waypoints  [(x_world, y_world, speed_kmh), ...]

L'approche choisie est le lane-following resampling sur les waypoints CARLA :
  - pas de Frenet (d=0 partout en suivi de voie pur, transformation inutile)
  - pas d'A* (la carte CARLA fournit deja les waypoints ordonnes de la route)
  - rééchantillonnage a pas regulier (2m) via wp.next(spacing)
  - vitesse constante depuis config, avec point d'extension courbure

Bug fix (etape 4) : coherence de cap sur get_waypoint()
---------------------------------------------------------
get_waypoint(ego_loc) est purement geometrique : renvoie le waypoint le plus
proche en distance euclidienne, SANS tenir compte du cap de l'ego. A la frontiere
entre deux routes (ex: road900 -> road939 a Town10HD_Opt), il peut snapper sur une
route adjacente allant en sens inverse (delta_yaw ~ 163 deg mesure).
=> Tous les waypoints generes pointent "derriere" -> ControlAgent retourne None
   -> brake 0.3 indefiniment.

Fix : apres get_waypoint(), verifier |delta_yaw| = |ego_heading - road_yaw|.
  - Si < _WRONG_WAY_THRESHOLD_DEG (45 deg) : snap coherent, mise a jour de l'anchor
  - Si >= 45 deg : snap incohérent, avancer l'anchor de la trajectoire precedente
    via .next(spacing) jusqu'a la position ego courante (reste sur la bonne road/lane
    par construction, contrairement a get_waypoint qui peut basculer)

Seuil 45 deg calibre sur Town10HD_Opt : virages legitimes mesures < 17 deg
(ticks 120-236 diagnostique), snap incohérent observe a 163 deg.
Marge : 28 deg au-dessus du max legitime, 118 deg sous le seuil bug.

Points d'extension documentes
------------------------------
_select_next_waypoint()  : branche ARBITRAIRE aux carrefours
                           -> sera resolue par DecisionAgent FSM (etape 5)
_speed_for_curvature()   : retourne target_speed constant
                           -> curvature-adaptive speed a l'etape 5
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from backend.src.agents.blackboard import Blackboard, PlanningState

logger = logging.getLogger(__name__)

# Seuil de coherence cap : au-dela, get_waypoint a snappe sur une route opposee.
# Calibre sur Town10HD_Opt : virages legitimes < 17 deg (mesure experimentale ticks 120-236).
# 45 deg laisse 28 deg de marge sans faux positifs sur les vrais virages.
_WRONG_WAY_THRESHOLD_DEG: float = 45.0


class PlanningAgent:
    """Generateur de trajectoire lane-following.

    Parameters
    ----------
    blackboard : Blackboard  -- etat partage inter-agents (thread-safe)
    cfg        : dict        -- configuration complete (section agents.planning)
    carla_map  : carla.Map   -- fourni apres connexion CARLA via set_carla_map()
                               Si None, fallback sur perception.lane_geometry
    """

    def __init__(
        self,
        blackboard: Blackboard,
        cfg: Dict[str, Any],
        carla_map: Any = None,
        branch_policy: str = "straight",
    ) -> None:
        self._blackboard = blackboard
        self._map = carla_map

        plan_cfg = cfg.get("agents", {}).get("planning", {})
        self._n_waypoints: int = int(plan_cfg.get("lookahead_waypoints", 20))
        self._spacing: float = float(plan_cfg.get("waypoint_spacing", 2.0))
        self._target_speed: float = float(plan_cfg.get("target_speed", 16.0))
        self._slow_speed: float = float(plan_cfg.get("slow_speed", 6.0))

        # Branch policy : "straight" (GoStraightPolicy) ou "tl_seeking" (cherche un vrai feu).
        self._branch_policy: str = branch_policy
        self._tl_seek_lookahead_m: float = float(plan_cfg.get("tl_seek_lookahead_m", 80.0))
        # Cache (road_id, lane_id) -> bool : evite de recalculer le lookahead TL pour la
        # meme branche a chaque tick (le resultat est stable pour un segment donne).
        self._tl_branch_cache: dict = {}

        # Index courant dans blackboard.route.route : avance monotoniquement a chaque tick.
        # Utilise par RouteFollowingBranchPolicy pour trouver le RoadOption au prochain fork.
        self._route_idx: int = 0
        # Resultat de la derniere recherche fork->route wp (pour log [BRANCH-ROUTE]).
        self._last_fork_dist: float = float("inf")
        self._last_fork_idx: int = -1
        # Cache de resolution de branche : (road_id, lane_id) → (chosen_road_id, chosen_lane_id).
        # Evite le lookahead N-step (CARLA API calls) a chaque tick pour le meme fork.
        self._branch_resolution_cache: dict = {}

        # Anchor : dernier waypoint CARLA valide utilise comme point de depart.
        # Mis a jour apres chaque snap coherent.
        # En cas de snap incohérent, on avance l'anchor via .next() (reste sur la
        # meme road/lane, contrairement a get_waypoint qui peut sauter sur une route inverse).
        self._anchor_wp: Optional[Any] = None

        # Diagnostic : rempli apres chaque appel a _generate_from_carla_map.
        # Permet d'inspecter road_id/lane_id/yaw/snap_coherent depuis la demo.
        self._last_diag: Dict[str, Any] = {
            "road_id":              None,
            "lane_id":              None,
            "start_wp_yaw":         None,            # yaw de la route au point ego (degrees)
            "snap_coherent":        True,             # False si fallback anchor utilise
            # --- Signaux FSM (utilises par DecisionAgent) ---
            "branch_in_chain":      False,           # True si len(nexts)>1 dans la trajectoire
            "branch_distance_m":    float("inf"),    # distance au premier vrai fork
            # --- Signaux de diagnostic purs (NE declenchent PAS la FSM) ---
            "junction_in_chain":    False,           # True si is_junction dans la trajectoire
            "junction_distance_m":  float("inf"),    # distance au premier wp is_junction
            "start_wp_is_junction": False,           # le wp ego est-il DANS une junction ?
            "junction_wp_count":    0,               # nb wp is_junction dans la trajectoire
            "branch_count":         0,               # nb vrais forks (len>1) dans la trajectoire
        }

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_carla_map(self, carla_map: Any) -> None:
        """Connecte la carte CARLA. Appeler apres world.get_map(), avant la boucle."""
        self._map = carla_map

    def set_branch_policy(self, policy: str) -> None:
        """Change la politique de selection de branche aux carrefours.

        Valeurs acceptees :
          "straight"        : GoStraightPolicy (comportement par defaut)
          "tl_seeking"      : cherche la branche menant a un feu (same_road)
          "route_following" : suit l'itineraire GlobalRoutePlanner (blackboard.route)
        """
        self._branch_policy = policy

    def run(
        self,
        ego_x: float,
        ego_y: float,
        ego_heading_deg: float = 0.0,
    ) -> None:
        """Genere la trajectoire depuis la position ego et publie dans blackboard.

        Parameters
        ----------
        ego_x, ego_y      : position monde de l'ego (metres CARLA)
        ego_heading_deg   : cap du vehicule en degrees (yaw CARLA, +X = 0 deg)
                            Utilise pour le filtre de coherence de cap.
                            Defaut 0.0 si non fourni (compatibilite tests unitaires).

        Appele a chaque tick avant ControlAgent.run().
        """
        # Avance l'index de la route globale AVANT la generation de trajectoire.
        # Necessaire pour que _route_option_near_fork() cherche dans la bonne fenetre.
        self._advance_route_idx(ego_x, ego_y, ego_heading_deg)

        route_complete = self._check_destination_reached(ego_x, ego_y)

        waypoints = self._generate_trajectory(ego_x, ego_y, ego_heading_deg)
        self._blackboard.publish_planning(PlanningState(
            waypoints=waypoints,
            branch_in_chain=bool(self._last_diag.get("branch_in_chain", False)),
            branch_distance_m=float(self._last_diag.get("branch_distance_m", float("inf"))),
            junction_in_chain=bool(self._last_diag.get("junction_in_chain", False)),
            junction_distance_m=float(self._last_diag.get("junction_distance_m", float("inf"))),
            route_complete=route_complete,
        ))

    # ------------------------------------------------------------------
    # Generation de trajectoire
    # ------------------------------------------------------------------

    def _generate_trajectory(
        self,
        ego_x: float,
        ego_y: float,
        ego_heading_deg: float = 0.0,
    ) -> List[Tuple[float, float, float]]:
        """Selectionne la source de waypoints et genere la trajectoire."""
        if self._map is not None:
            return self._generate_from_carla_map(ego_x, ego_y, ego_heading_deg)

        # Fallback : lire depuis blackboard.perception (PerceptionAgent publie en etape 2)
        lane_geo = self._blackboard.perception.lane_geometry
        if lane_geo and "waypoints_ahead" in lane_geo:
            logger.debug("PlanningAgent: fallback sur perception.lane_geometry.")
            return self._convert_from_perception(lane_geo["waypoints_ahead"])

        logger.warning(
            "PlanningAgent: pas de carte CARLA et pas de perception.lane_geometry. "
            "Trajectoire vide publiee -> ControlAgent freinera."
        )
        return []

    def _generate_from_carla_map(
        self,
        ego_x: float,
        ego_y: float,
        ego_heading_deg: float = 0.0,
    ) -> List[Tuple[float, float, float]]:
        """Construit la trajectoire via l'API CARLA Waypoint.

        Filtre de coherence de cap
        --------------------------
        get_waypoint(ego_loc) renvoie le waypoint geometriquement le plus proche
        SANS tenir compte du cap de l'ego. Si |delta_yaw| >= _WRONG_WAY_THRESHOLD_DEG,
        le snap est sur une route incohérente (sens inverse probable).
        Dans ce cas, on avance l'anchor de la trajectoire precedente via .next(spacing),
        qui reste sur la meme route/voie par construction.
        """
        try:
            import carla  # type: ignore
            ego_loc = carla.Location(x=float(ego_x), y=float(ego_y), z=0.0)
        except ModuleNotFoundError:
            logger.error("PlanningAgent: module carla non disponible.")
            return []

        snapped_wp = self._map.get_waypoint(ego_loc, project_to_road=True)
        coherent = self._heading_coherent(snapped_wp, ego_heading_deg)

        if coherent:
            wp = snapped_wp
            self._anchor_wp = wp
        else:
            wp_yaw = self._safe_yaw(snapped_wp)
            delta = self._delta_yaw(ego_heading_deg, wp_yaw)
            logger.warning(
                "WRONG-WAY SNAP detecte (road_id=%s lane=%s) : "
                "road_yaw=%.1f deg, ego=%.1f deg, delta_yaw=%.1f deg > %.0f deg -- "
                "fallback anchor.next() (meme road/lane que tick precedent).",
                getattr(snapped_wp, "road_id", "?"),
                getattr(snapped_wp, "lane_id", "?"),
                wp_yaw, ego_heading_deg, delta,
                _WRONG_WAY_THRESHOLD_DEG,
            )
            if self._anchor_wp is None:
                # Premier tick, pas d'anchor disponible — rare si spawn correct
                logger.error(
                    "PlanningAgent: snap incohérent au premier tick et pas d'anchor. "
                    "Trajectoire vide (un tick de frein). Verifie le spawn point."
                )
                return []
            # Avancer l'anchor jusqu'a la position ego courante via .next()
            self._anchor_wp = self._advance_anchor_to_ego(ego_x, ego_y)
            wp = self._anchor_wp

        # Vitesse effective : DecisionAgent publie decision.target_speed (SLOW_DOWN = 6 km/h,
        # FOLLOW_LANE = 16 km/h). PlanningAgent l'embed dans les waypoints pour ControlAgent.
        # Si DecisionAgent n'a pas encore tourne (premier tick), decision.target_speed = 16.0.
        # LAG NOTE : decision.target_speed vient du tick precedent (voir module docstring).
        effective_speed = float(self._blackboard.decision.target_speed)
        if effective_speed < 0:
            # Seules les valeurs négatives sont invalides — 0 km/h = arrêt intentionnel (RED_LIGHT).
            effective_speed = self._target_speed

        # Remplir _last_diag avec les infos du point de depart effectivement utilise.
        # Reset complet : chaque tick repart de zero pour branch ET junction.
        start_is_junc = bool(getattr(wp, "is_junction", False))
        self._last_diag.update({
            "road_id":              getattr(wp, "road_id", None),
            "lane_id":              getattr(wp, "lane_id", None),
            "start_wp_yaw":         self._safe_yaw(wp),
            "snap_coherent":        coherent,
            # --- Signaux FSM : BRANCH (len(nexts)>1 uniquement) ---
            # Mis a jour dans la boucle de generation ci-dessous.
            "branch_in_chain":      False,
            "branch_distance_m":    float("inf"),
            # --- Diagnostics JUNCTION (is_junction, pas forcement un fork) ---
            # is_junction=True marque toute la zone geometrique d'une intersection
            # sur CARLA (Town10 : presque partout). NE PAS utiliser pour la FSM.
            "junction_in_chain":    start_is_junc,
            "junction_distance_m":  0.0 if start_is_junc else float("inf"),
            "start_wp_is_junction": start_is_junc,
            "junction_wp_count":    0,
            "branch_count":         0,
        })

        waypoints: List[Tuple[float, float, float]] = []
        for i in range(self._n_waypoints):
            next_wp, is_branch, is_junc_only = self._select_next_waypoint(wp, self._spacing)
            if next_wp is None:
                break

            # --- Signal FSM : vrai fork (len(nexts)>1) ---
            # is_branch=True seulement si _select_next_waypoint a vu len(nexts)>1.
            if is_branch and not self._last_diag["branch_in_chain"]:
                self._last_diag["branch_in_chain"] = True
                self._last_diag["branch_distance_m"] = (i + 1) * self._spacing

            # --- Diagnostic JUNCTION (is_junction, single ou multi-branche) ---
            next_is_junc = getattr(next_wp, "is_junction", False)
            if not self._last_diag["junction_in_chain"] and (is_branch or is_junc_only or next_is_junc):
                self._last_diag["junction_in_chain"] = True
                self._last_diag["junction_distance_m"] = (i + 1) * self._spacing
            if next_is_junc:
                self._last_diag["junction_wp_count"] += 1

            loc = next_wp.transform.location
            speed = self._speed_for_curvature(0.0, effective_speed)
            waypoints.append((float(loc.x), float(loc.y), speed))
            wp = next_wp

        return waypoints

    # ------------------------------------------------------------------
    # Coherence de cap
    # ------------------------------------------------------------------

    def _heading_coherent(self, wp: Any, ego_heading_deg: float) -> bool:
        """True si le cap de la route au waypoint est coherent avec le cap ego.

        Incoherent si |delta_yaw| >= _WRONG_WAY_THRESHOLD_DEG (45 deg).
        En cas d'AttributeError (mock sans rotation), retourne True par defaut.

        Utilise _safe_yaw (normalise) pour etre coherent avec _last_diag.
        """
        yaw = self._safe_yaw(wp)
        if yaw is None:
            return True
        return abs(self._delta_yaw(ego_heading_deg, yaw)) < _WRONG_WAY_THRESHOLD_DEG

    @staticmethod
    def _delta_yaw(ego_heading_deg: float, road_yaw_deg: float) -> float:
        """Plus petite difference angulaire signee (ego - route) dans [-180, 180].

        La formule `((a - b + 180) % 360) - 180` est mathematiquement correcte
        pour TOUTE valeur d'entree (pas de contrainte [-180, 180] sur les inputs).
        Demonstration : soit d = a - b, (d + 180) % 360 normalise d dans [0, 360),
        soustraire 180 donne [-180, 180). Python % garantit un resultat positif
        pour diviseur positif, donc -539.6 est gere correctement.
        """
        return ((ego_heading_deg - road_yaw_deg + 180.0) % 360.0) - 180.0

    @staticmethod
    def _safe_yaw(wp: Any) -> Optional[float]:
        """Lit transform.rotation.yaw et normalise dans [-180, 180).

        CARLA peut retourner des valeurs hors plage (ex: -539.6, +360.0, -270.2)
        car le moteur stocke parfois la rotation accumulee ou utilise des
        conventions de wrapping differentes selon le type de route.

        On normalise ICI pour que :
          - _last_diag["start_wp_yaw"] soit toujours lisible en [-180, 180)
          - _heading_coherent() opere sur des valeurs canoniques
        Note : _delta_yaw() gere correctement les inputs non-normalises (via % 360),
        donc normaliser les inputs est redondant pour la correction mathematique
        mais obligatoire pour la lisibilite du diagnostic et la testabilite.
        """
        try:
            raw = float(wp.transform.rotation.yaw)
        except (AttributeError, TypeError):
            return None
        return ((raw + 180.0) % 360.0) - 180.0

    # ------------------------------------------------------------------
    # Anchor fallback
    # ------------------------------------------------------------------

    def _advance_anchor_to_ego(self, ego_x: float, ego_y: float) -> Any:
        """Avance self._anchor_wp via .next(spacing) jusqu'au point le plus proche de l'ego.

        Reste sur la meme road/lane que l'anchor : .next() ne re-snappe jamais
        sur une route adjacente, contrairement a get_waypoint().

        La limite de boucle (n_waypoints * 4) evite une boucle infinie si la route
        se termine avant d'avoir atteint l'ego (ex: fin de carte).
        """
        wp = self._anchor_wp
        best_dist = math.hypot(
            wp.transform.location.x - ego_x,
            wp.transform.location.y - ego_y,
        )

        for _ in range(self._n_waypoints * 4):
            nexts = wp.next(self._spacing)
            if not nexts:
                break
            nxt = nexts[0]
            dist = math.hypot(
                nxt.transform.location.x - ego_x,
                nxt.transform.location.y - ego_y,
            )
            if dist > best_dist:
                # On a passe le point le plus proche — rester sur wp courant
                break
            best_dist = dist
            wp = nxt

        return wp

    # ------------------------------------------------------------------
    # Navigation et vitesse
    # ------------------------------------------------------------------

    def _select_next_waypoint(
        self, wp: Any, spacing: float
    ) -> "Tuple[Optional[Any], bool, bool]":
        """Suit la voie de `spacing` metres en avant.

        Retourne (next_wp, is_branch, is_junc_only) :

          is_branch    : len(nexts) > 1 — vrai point de bifurcation (SIGNAL FSM).
                         Declenche SLOW_DOWN dans DecisionAgent et active branch_heading.

          is_junc_only : len(nexts) == 1 ET nexts[0].is_junction=True.
                         Zone junction CARLA sans fork reel (DIAGNOSTIC UNIQUEMENT).
                         NE declenche PAS la FSM.

        DISTINCTION CRITIQUE (observee Town10HD_Opt, 400 ticks) :
          is_junction=True sur CARLA waypoints marque toute la zone geometrique
          d'une intersection (souvent 30-50m avant/apres). Sur Town10, bc=0 pendant
          400 ticks meme avec jwc=9-16/20 : ce sont des junctions "traversantes"
          (une seule lane predeterminee, pas de choix). is_junc_only seul -> inutile
          pour la FSM.
          Seul len(nexts)>1 indique une bifurcation reelle necessitant une decision.

        RESOLUTION DU FALLBK EN RAFALE (fix etape 5 preserve) :
          Quand is_branch=True, DecisionAgent a publie branch_heading -> on filtre
          nexts[] de facon deterministe -> meme branche a chaque tick -> pas d'oscillation.
        """
        nexts = wp.next(spacing)
        if not nexts:
            return None, False, False

        if len(nexts) > 1:
            # is_branch = True : vrai fork, signal FSM
            if "branch_count" in self._last_diag:
                self._last_diag["branch_count"] += 1

            # --- Route-Following branch policy ---
            # Prioritaire sur tl_seeking et GoStraightPolicy.
            # Lit le RoadOption de la route globale (blackboard.route) le plus proche du fork.
            # LEFT/RIGHT/STRAIGHT -> branche selectionnee par delta_yaw.
            # LANEFOLLOW/ambigü -> N-step lookahead (calcule une fois, cache par road_id/lane_id).
            # CHANGELANELEFT/CHANGELANERIGHT -> log + GoStraightPolicy (pas de changement voie).
            # Pas de wp route proche (>20m) -> GoStraightPolicy + log.
            # Log [BRANCH-ROUTE] INCONDITIONNEL a chaque fork.
            if self._branch_policy == "route_following":
                fork_loc = wp.transform.location
                road_yaw = self._safe_yaw(wp) or 0.0
                opt = self._route_option_near_fork(fork_loc.x, fork_loc.y)

                # Cache lookup : resoudre chaque fork (road_id, lane_id) une seule fois.
                fork_key = (getattr(wp, "road_id", None), getattr(wp, "lane_id", None))
                if fork_key in self._branch_resolution_cache:
                    tgt_road, tgt_lane = self._branch_resolution_cache[fork_key]
                    chosen = next(
                        (n for n in nexts
                         if getattr(n, "road_id", None) == tgt_road
                         and getattr(n, "lane_id", None) == tgt_lane),
                        nexts[0],  # fallback si la branche cachee disparait
                    )
                    chosen_yaw = self._safe_yaw(chosen) or 0.0
                    delta = self._delta_yaw(road_yaw, chosen_yaw)
                else:
                    chosen, chosen_yaw, delta = self._pick_branch_by_option(nexts, opt, road_yaw)
                    self._branch_resolution_cache[fork_key] = (
                        getattr(chosen, "road_id", None),
                        getattr(chosen, "lane_id", None),
                    )

                opt_name = getattr(opt, "name", None) or "GoStraight(no_route_wp)"
                branches_str = "[" + ", ".join(
                    f"{self._safe_yaw(n) or 0:.1f}" for n in nexts
                ) + "]"
                print(
                    f"[BRANCH-ROUTE] FORK ({fork_loc.x:.1f}, {fork_loc.y:.1f}):"
                    f"  opt={opt_name}"
                    f"  nearest_rt={self._last_fork_dist:.1f}m(idx={self._last_fork_idx})"
                    f"  route_idx={self._route_idx}"
                    f"  branches={branches_str}"
                    f"  -> yaw={chosen_yaw:.1f}  delta={delta:.1f}"
                )
                return chosen, True, False

            # --- TL-Seeking branch policy ---
            # Avant GoStraightPolicy : cherche la branche qui mene a un feu sur la meme route.
            # Si trouvee : la prendre directement (garantit feu visible a l'ego).
            # Sinon : fallback GoStraightPolicy (branch_heading depuis DecisionAgent).
            if self._branch_policy == "tl_seeking":
                tl_branches = [n for n in nexts if self._has_tl_on_branch(n)]
                if tl_branches:
                    chosen = tl_branches[0]
                    loc = wp.transform.location
                    chosen_yaw = self._safe_yaw(chosen) or 0.0
                    logger.info(
                        "TL-SEEKING FORK (%.1f, %.1f): %d branches -> TL same_road sur "
                        "branche yaw=%.1f deg (%d candidat(s))",
                        loc.x, loc.y, len(nexts), chosen_yaw, len(tl_branches),
                    )
                    return chosen, True, False
                logger.info(
                    "TL-SEEKING FORK (%.1f, %.1f): aucune branche TL same_road -> "
                    "fallback GoStraightPolicy",
                    wp.transform.location.x, wp.transform.location.y,
                )
            # --- end TL-Seeking ---

            branch_heading = getattr(
                getattr(self._blackboard, "decision", None), "branch_heading", None
            )
            if branch_heading is not None:
                best = min(
                    nexts,
                    key=lambda n: abs(self._delta_yaw(branch_heading, self._safe_yaw(n) or 0.0)),
                )
                best_yaw = self._safe_yaw(best) or 0.0
                delta_chosen = self._delta_yaw(branch_heading, best_yaw)
                loc = wp.transform.location
                alignment = (
                    "[ALIGNED]" if abs(delta_chosen) < 30.0
                    else "[NON-ALIGNED: branche la moins mauvaise -- GoStraightPolicy limite]"
                )
                logger.info(
                    "FORK (%.1f, %.1f): FSM branch_heading=%.1f deg -> "
                    "branche yaw=%.1f deg, delta=%.1f deg %s (%d candidats)",
                    loc.x, loc.y, branch_heading,
                    best_yaw, delta_chosen, alignment, len(nexts),
                )
                return best, True, False  # is_branch=True, is_junc_only=False

            loc = wp.transform.location
            logger.warning(
                "FORK at (%.1f, %.1f): %d branches -- nexts[0] arbitraire "
                "(DecisionAgent pas encore actif ou pas de branch_heading).",
                loc.x, loc.y, len(nexts),
            )
            return nexts[0], True, False  # is_branch=True, is_junc_only=False

        # Un seul suivant : zone junction CARLA sans vrai fork
        is_junc_only = getattr(nexts[0], "is_junction", False)
        return nexts[0], False, is_junc_only  # is_branch=False

    # ------------------------------------------------------------------
    # TL-Seeking lookahead
    # ------------------------------------------------------------------

    def _has_tl_on_branch(self, start_wp: Any) -> bool:
        """Marche jusqu a _tl_seek_lookahead_m depuis start_wp et cherche un TL same_road.

        "same_road" : lm.road_id == wp.road_id au moment de la detection.
        Indique que le feu controle la route sur laquelle l ego circulera (pas transversal).
        Pas infaillible sur TOUS les spawn (voir BFS), mais fiable pour Town10HD_Opt.

        Cache par (road_id, lane_id) : le resultat est stable pour un segment donne.
        Evite de recalculer le lookahead pour la meme branche a chaque tick.
        """
        key = (getattr(start_wp, "road_id", None), getattr(start_wp, "lane_id", None))
        if key in self._tl_branch_cache:
            return self._tl_branch_cache[key]

        step = 5.0
        wp = start_wp
        d = 0.0
        result = False
        while d < self._tl_seek_lookahead_m:
            try:
                for lm in wp.get_landmarks(60.0, stop_at_junction=False):
                    if str(lm.type) == "1000001":
                        lm_road = getattr(lm, "road_id", None)
                        if lm_road is not None and lm_road == wp.road_id:
                            result = True
                            break
                if result:
                    break
            except Exception:
                pass
            nexts = wp.next(step)
            if not nexts:
                break
            wp = nexts[0]
            d += step

        self._tl_branch_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Route-Following helpers
    # ------------------------------------------------------------------

    def _advance_route_idx(
        self, ego_x: float, ego_y: float, ego_heading_deg: float
    ) -> None:
        """Avance self._route_idx vers le waypoint de la route le plus proche de l'ego.

        Methode robuste par DISTANCE EUCLIDIENNE (pas de dependance au cap ego).
        Avance tant que le waypoint SUIVANT est plus proche que le waypoint COURANT.
        Garantit la monotonie : on ne recule jamais dans la route.

        Remplace l'ancien dot-product qui causait un saut massif de route_idx quand
        la route demarre en direction opposee au cap du spawn (tous les wps "derriere"
        selon le dot-product → idx saute a 55 en un seul tick).

        ego_heading_deg conserve dans la signature pour compat API mais inutilise.
        """
        route = self._blackboard.route
        if not route.active or not route.route:
            return
        while self._route_idx < len(route.route) - 1:
            loc_cur  = route.route[self._route_idx][0]
            loc_next = route.route[self._route_idx + 1][0]
            d_cur  = math.hypot(loc_cur.x  - ego_x, loc_cur.y  - ego_y)
            d_next = math.hypot(loc_next.x - ego_x, loc_next.y - ego_y)
            if d_next > d_cur + 0.05:
                break  # wp courant est le plus proche -> s'arreter
            # d_next <= d_cur + 0.05 : avancer.
            # Tolerance 5cm pour les pseudo-doublons de GlobalRoutePlanner (waypoints aux
            # points de decision places a quelques micrometres l'un de l'autre avec
            # des RoadOption differents : LANEFOLLOW+STRAIGHT, LANEFOLLOW+RIGHT, etc.)
            self._route_idx += 1

    def _route_option_near_fork(self, fork_x: float, fork_y: float) -> Optional[Any]:
        """Renvoie le RoadOption du wp de la route le plus proche du fork.

        Recherche depuis max(0, route_idx-5) jusqu'a la FIN de la route.
        Pas de fenetre superieure : le fork peut etre en avance sur route_idx si
        la trajectoire PlanningAgent depasse l'index courant (lookahead 40m).

        Tolerance 20m (etait 8m) : trace_route() et wp.next() generent des grilles
        legerement decalees. 8m causait des no_route_wp meme quand le wp etait a 9-12m.

        Stocke _last_fork_dist / _last_fork_idx pour le log [BRANCH-ROUTE].
        """
        route = self._blackboard.route
        if not route.active or not route.route:
            self._last_fork_dist = float("inf")
            self._last_fork_idx  = -1
            return None
        best_dist = 20.0  # tolerance en metres
        best_opt  = None
        best_idx  = -1
        search_from = max(0, self._route_idx - 5)
        for i in range(search_from, len(route.route)):
            loc = route.route[i][0]
            d = math.hypot(loc.x - fork_x, loc.y - fork_y)
            if d < best_dist:
                best_dist = d
                best_opt  = route.route[i][1]
                best_idx  = i
        self._last_fork_dist = best_dist
        self._last_fork_idx  = best_idx
        return best_opt

    def _pick_branch_by_option(
        self, nexts: list, option: Optional[Any], heading_deg: float
    ) -> "Tuple[Any, float, float]":
        """Choisit la branche de nexts[] selon le RoadOption.

        Mapping CARLA (coordonnees LEFT-HAND, yaw positif = virage droite) :
          LEFT  -> branche a yaw le PLUS ELEVE (delta_yaw positif max)
          RIGHT -> branche a yaw le PLUS BAS   (delta_yaw negatif min)
          STRAIGHT / LANEFOLLOW / None -> GoStraightPolicy (|delta_yaw| min)
          CHANGELANELEFT/RIGHT -> GoStraightPolicy + log warning

        Retourne (chosen_wp, chosen_yaw_deg, delta_deg).
        """
        try:
            from agents.navigation.local_planner import RoadOption
            _have_ro = True
        except ImportError:
            _have_ro = False
            RoadOption = None  # type: ignore

        deltas = [
            (n, self._delta_yaw(heading_deg, self._safe_yaw(n) or 0.0))
            for n in nexts
        ]

        if _have_ro and option == RoadOption.LEFT:
            chosen, d = max(deltas, key=lambda x: x[1])
        elif _have_ro and option == RoadOption.RIGHT:
            chosen, d = min(deltas, key=lambda x: x[1])
        elif _have_ro and option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            logger.warning(
                "[BRANCH-ROUTE] RoadOption=%s non supporte (pas de changement de voie)"
                " -> GoStraightPolicy",
                option.name,
            )
            chosen, d = min(deltas, key=lambda x: abs(x[1]))
        else:
            # STRAIGHT, LANEFOLLOW, VOID, None
            # N-step lookahead : suit chaque branche K pas en avant et compare au
            # waypoint de route a la meme profondeur (route[_last_fork_idx + K]).
            # Resout les lane-splits ambigus (branches yaw identiques) : les branches
            # divergent geometriquement apres K*spacing metres meme si elles partagent
            # le meme yaw initial (ex: Town10 fork (-41.6,83.7), branches=[-90.2,-90.2]).
            # Ce calcul est coûteux (K CARLA API calls par branche) mais _pick_branch_by_option
            # n'est appele QUE sur cache-miss (voir _branch_resolution_cache dans route_following).
            # Sinon : GoStraightPolicy (|delta_yaw| min).
            route = self._blackboard.route
            K = 8  # 8 pas × spacing=2m = 16m de lookahead
            ref_idx = (self._last_fork_idx + K if self._last_fork_idx >= 0
                       else self._route_idx + K)
            if route.active and route.route and 0 < ref_idx < len(route.route):
                target_loc = route.route[ref_idx][0]

                def _follow_k(start_wp: Any) -> Any:
                    cur = start_wp
                    for _ in range(K):
                        nxts = cur.next(self._spacing)
                        if not nxts:
                            break
                        cur = nxts[0]
                    return cur.transform.location

                chosen = min(
                    nexts,
                    key=lambda n: math.hypot(
                        _follow_k(n).x - target_loc.x,
                        _follow_k(n).y - target_loc.y,
                    ),
                )
                d = self._delta_yaw(heading_deg, self._safe_yaw(chosen) or 0.0)
            else:
                chosen, d = min(deltas, key=lambda x: abs(x[1]))

        return chosen, self._safe_yaw(chosen) or 0.0, d

    def _check_destination_reached(self, ego_x: float, ego_y: float) -> bool:
        """Retourne True si l'ego est a moins de 5m de la destination (route active)."""
        route = self._blackboard.route
        if not route.active or route.destination is None:
            return False
        dest = route.destination
        return math.hypot(dest.x - ego_x, dest.y - ego_y) < 5.0

    # ------------------------------------------------------------------
    # Points d'extension
    # ------------------------------------------------------------------

    def _speed_for_curvature(
        self, curvature: float, base_speed: Optional[float] = None
    ) -> float:
        """Vitesse cible selon la courbure locale.

        base_speed : vitesse fournie par DecisionAgent (decision.target_speed).
                     Si None, utilise self._target_speed (config).
        Extension point : curvature > threshold -> ralentir (etape ulterieure).
        Future: return max(self._slow_speed, base * (1 - k*|curvature|))
        """
        _ = curvature
        return base_speed if base_speed is not None else self._target_speed

    def _convert_from_perception(
        self,
        waypoints_ahead: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float, float]]:
        """Convertit (x, y, yaw_deg) [Perception] -> (x, y, speed_kmh) [Planning]."""
        return [(float(x), float(y), self._target_speed) for x, y, _ in waypoints_ahead]

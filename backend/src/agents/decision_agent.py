"""Decision Agent -- Etape 5.

FSM comportementale qui lit le blackboard (planning + perception) et publie
une decision haut niveau : etat FSM, vitesse cible, cap de branche au carrefour.

Role dans l'architecture
------------------------
Lit   : blackboard.planning  (tick t-1) -- junction_in_chain, junction_distance_m
         blackboard.perception (tick t-1) -- detections d'obstacles
         ego_speed_ms, ego_heading_deg, ego_position (passes a run() par le loop)
Publie: blackboard.decision -- fsm_state, target_speed, branch_heading, is_at_junction

ORDRE D'EXECUTION dans la boucle de tick :
  1. DecisionAgent.run()   -- publie decision avec branch_heading
  2. PlanningAgent.run()   -- lit decision.branch_heading -> filtre nexts[]
  3. ControlAgent.run()    -- lit planning.waypoints (avec vitesse FSM)

LIMITE 1 -- GoStraightPolicy : stable != correct
  branch_heading = ego_heading stabilise le choix (plus de FALLBK en rafale) mais
  ne garantit PAS la bonne branche a un carrefour non-aligne. Voir GoStraightPolicy.
  Critere de validation etape 5 : FALLBK disparu ET delta_yaw de la branche logue.

LIMITE 2 -- SLOW_DOWN ne garantit pas l'arret
  La FSM ralentit a slow_speed sur detection d'obstacle ou approche de carrefour.
  Elle NE s'arrete PAS sur obstacle : pas de STOP sur obstacle (seulement sur stop_sign
  via Perception, non implemente etape 5). Le freinage d'urgence est la responsabilite
  du Safety Agent (etape 6, droit de veto). LA FSM SEULE NE PREVIENT PAS LA COLLISION.

LIMITE 3 -- lag d'un tick (inoffensif pour carrefour, a surveiller pour obstacles proches)
  DecisionAgent lit planning.junction_in_chain du tick t-1 (planif 40m devant -> 7s
  d'anticipation a 20 km/h, largement suffisant). Pour les obstacles proches (< 15m),
  la latence CUMULEE perception->decision->planning->control devra etre evaluee en CARLA
  reel a l'etape 6. Un obstacle a 5m n'a pas 7s de marge.

Extension RL (non implementee etape 5) :
  BranchPolicy est injectee a __init__. Basculer sur PPO lane-change :
    agent = DecisionAgent(bb, cfg, branch_policy=RLLaneChangePolicy(model))
  Aucune modification de DecisionAgent ou PlanningAgent requise.
  RLLaneChangePolicy lira blackboard.rl_policy.action (slot reserve dans Blackboard).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from backend.src.agents.blackboard import Blackboard, DecisionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface BranchPolicy (extension point pour RL)
# ---------------------------------------------------------------------------

class BranchPolicy(ABC):
    """Interface : calcule le cap souhaite (degrees) a un carrefour.

    DecisionAgent appelle select_branch_heading() quand junction_in_chain=True.
    PlanningAgent filtre nexts[] pour retenir la branche dont le yaw est le
    plus proche de ce cap.

    Ce cap est une INTENTION, pas une garantie : si aucune branche n'est alignee,
    la moins mauvaise est choisie (voir GoStraightPolicy pour les limites et le
    log [NON-ALIGNED] dans PlanningAgent).

    Implementations :
    - GoStraightPolicy (defaut etape 5)  : retourne ego_heading_deg
    - RLLaneChangePolicy (etape future)  : PPO sur (ego_state, context) -> cap
        Lira blackboard.rl_policy.action (slot reserve Blackboard) pour l'intention
        ('change_lane_left' -> ego_heading - 90 deg, etc.).
    """

    @abstractmethod
    def select_branch_heading(
        self,
        ego_heading_deg: float,
        ego_speed_kmh: float,
        decision_context: DecisionState,
    ) -> float:
        """Retourne le cap souhaite (degrees) a un carrefour.

        PlanningAgent choisira le nexts[] dont le yaw est le plus proche de
        cette valeur via _select_next_waypoint (voir planning_agent.py).
        """
        raise NotImplementedError


class GoStraightPolicy(BranchPolicy):
    """Politique par defaut etape 5 : maintient le cap courant de l'ego.

    STABILITE vs CORRECTION -- distinction critique :
    -------------------------------------------------
    STABILITE (garantie) : branch_heading = ego_heading_deg est CONSTANT d'un tick
    a l'autre -> PlanningAgent choisit toujours la meme branche -> fin des FALLBK
    en rafale (19 FALLBK observes ticks 128/143, roads 467/375/382 Town10HD_Opt).

    CORRECTION (non garantie) : "aller vers mon cap" ne signifie PAS toujours la
    branche semantiquement correcte (ex: carrefour perpendiculaire ou toutes les
    branches sont > 45 deg de l'ego). Dans ce cas, la branche choisie est stable
    mais arbitraire -- c'est la "moins mauvaise" parmi les candidats.

    Exemple observe Town10HD_Opt :
      ego_heading = +135 deg, branches = {-90 deg, +45 deg, +55 deg}
      GoStraight choisit la branche a +55 deg (delta = +80 deg)
      -> log [NON-ALIGNED] visible dans PlanningAgent
      -> stable (meme branche chaque tick) mais pas semantiquement "tout droit"

    Pour un choix directionnel INTENTIONNEL (tourner a droite, changer de voie) :
      -> Remplacer par RLLaneChangePolicy (PPO, etape ulterieure) via injection a
         DecisionAgent.__init__. Aucune modification de PlanningAgent requise.

    Critere de validation etape 5 :
      1. FALLBK en rafale disparu (branche stable d'un tick a l'autre)
      2. Log JUNCTION visible avec delta_yaw de la branche choisie
      3. Si delta > 30 deg : [NON-ALIGNED] logue -- limite documentee, pas un bug
    """

    def select_branch_heading(
        self,
        ego_heading_deg: float,
        ego_speed_kmh: float,
        decision_context: DecisionState,
    ) -> float:
        return ego_heading_deg


# ---------------------------------------------------------------------------
# DecisionAgent FSM
# ---------------------------------------------------------------------------

_VALID_FSM_STATES = frozenset({"FOLLOW_LANE", "SLOW_DOWN", "RED_LIGHT", "STOP_SIGN", "STOP", "YIELD", "EMERGENCY"})


class DecisionAgent:
    """Agent de decision comportementale -- Machine a etats finis (FSM).

    Voir module docstring pour les limites et le contrat blackboard.

    Parameters
    ----------
    blackboard    : Blackboard partagee (thread-safe)
    cfg           : config complete (section agents.planning utilisee pour les seuils)
    branch_policy : BranchPolicy -- GoStraightPolicy par defaut.
                    Remplacer par RLLaneChangePolicy pour la lane-change RL.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        cfg: Dict[str, Any],
        branch_policy: Optional[BranchPolicy] = None,
    ) -> None:
        self._blackboard = blackboard
        plan_cfg = cfg.get("agents", {}).get("planning", {})
        self._normal_speed: float  = float(plan_cfg.get("target_speed", 16.0))
        self._slow_speed: float    = float(plan_cfg.get("slow_speed", 6.0))
        self._obstacle_dist: float = float(plan_cfg.get("obstacle_distance", 15.0))
        # obs_stop_m < obstacle_dist : arrêt complet (tgt=0) quand l'obstacle est très proche.
        # Empêche l'ego d'entrer dans la zone Safety MIN_SAFE_M (7.5m) par ralentissement trop lent.
        # À cette distance, l'ego a assez d'espace pour s'arrêter via le PID avant la zone Safety.
        self._obs_stop_m: float    = float(plan_cfg.get("obs_stop_m", 10.0))
        # Seuil de distance pour declencher SLOW_DOWN sur un fork imminent.
        # Seul len(nexts)>1 (branch_in_chain=True) active ce seuil —
        # is_junction seul (diagnostique) NE declenche PAS la FSM.
        self._branch_near_m: float = float(plan_cfg.get("branch_near_m", 15.0))

        # Car-following (Scénario 3) — IDM-simplifié à 2 termes
        # Loi : v_target = v_lead + k_dist*(d_obs-d_secu) - k_vel*(v_ego-v_lead)
        #       clampé [0, v_cruise]
        # d_secu = max(obs_stop_m, headway_s * v_ego_ms)  — règle des 2 secondes
        # k_dist [km/h/m] : gain sur l'erreur de distance.
        #   d_obs > d_secu → v_target > v_lead (rattrape), d_obs < d_secu → v_target < v_lead.
        # k_vel [sans dim] : amortissement sur la vitesse relative.
        #   si v_ego > v_lead (on se rapproche) → réduit v_target (freine avant de coller).
        self._cf_headway_s: float = float(plan_cfg.get("cf_headway_s", 2.0))
        self._cf_k_dist: float    = float(plan_cfg.get("cf_k_dist",    0.5))
        self._cf_k_vel: float     = float(plan_cfg.get("cf_k_vel",     0.5))

        # Hysterèse SLOW_DOWN : évite l'oscillation entrée/sortie autour du seuil obstacle_dist.
        # Une fois SLOW_DOWN/STOP actif pour obstacle, maintient obstacle_close=True
        # jusqu'à d > obstacle_dist + obs_slow_hysteresis_m (default 5m).
        # Cas typique : lead sort brièvement de 15m lors de son accélération initiale.
        # Sans hysterèse : SLOW_DOWN→FOLLOW_LANE→v_cruise→collision→Safety veto évitable.
        self._obs_slow_hysteresis_m: float = float(plan_cfg.get("obs_slow_hysteresis_m", 5.0))

        # Traffic light FSM params (ETAPE C — logique séparée détection / freinage)
        # tl_detect_m      : horizon d'armement TL (m). Large (50m) pour survivre aux
        #                    junctions masquantes (~28m). Le freinage N'EST PAS déclenché
        #                    à cette distance — seulement quand dist_freinage le justifie.
        # tl_brake_margin_m: marge de sécurité ajoutée à dist_freinage pour calculer
        #                    le seuil d'entrée en RED_LIGHT.
        #                    Freine quand dist_feu <= v²/(2*a) + tl_brake_margin_m.
        #                    @ 16 km/h : v²/2a=6.6m → seuil ≈ 9.6m (stop ~3m avant ligne).
        # tl_decel_mps2    : décélération de référence (m/s²). Utilisée pour dist_freinage
        #                    et la formule dilemme YELLOW.
        self._tl_detect_m: float      = float(plan_cfg.get("tl_detect_m",      50.0))
        self._tl_brake_margin_m: float = float(plan_cfg.get("tl_brake_margin_m", 3.0))
        self._tl_decel_mps2: float    = float(plan_cfg.get("tl_decel_mps2",     1.5))

        # Stop sign FSM params (logique séparée détection / freinage — symétrique au TL)
        # stop_detect_m      : horizon d'armement panneau STOP (m). Large pour voir tôt.
        # stop_brake_margin_m: marge de sécurité sur dist_freinage.
        #                      Freine quand stop_dist <= v²/(2*a) + stop_brake_margin_m.
        #                      @ 16 km/h : v²/2a=6.6m → seuil ≈ 9.6m (arrêt ~3-4m avant ligne).
        # stop_hold_ticks    : ticks à v≈0 avant reprise.
        # stop_clear_dist_m  : distance au-delà de laquelle le panneau est effacé (passé ou loin).
        self._stop_detect_m: float      = float(plan_cfg.get("stop_detect_m",      50.0))
        self._stop_brake_margin_m: float = float(plan_cfg.get("stop_brake_margin_m", 3.0))
        self._stop_hold_ticks: int      = int(plan_cfg.get("stop_hold_ticks",        20))
        self._stop_clear_dist_m: float  = float(plan_cfg.get("stop_clear_dist_m",   50.0))
        self._stop_held_ticks: int    = 0
        self._stop_sign_cleared: bool = False
        # Latch STOP_SIGN : meme principe que _tl_latch.
        # Maintient stop_sign_stop=True pendant le freinage (v decroit → brake@ decroit
        # → condition dist≤brake@ pourrait se rouvrir sans latch).
        # Effacé quand cleared=True (hold complet) OU dist > stop_detect_m (panneau passé).
        self._stop_latch: bool = False

        self._branch_policy: BranchPolicy = branch_policy or GoStraightPolicy()
        self._fsm_state: str = "FOLLOW_LANE"
        self._reason: str = "init"
        self._tick_count: int = 0
        # Latch TL : une fois en RED_LIGHT pour un feu rouge, maintient tl_stop=True
        # meme si dist_freinage(v_courant) repasse en-dessous de tl_dist (v decroit
        # pendant le freinage → brake@ decreit → condition dist≤brake@ peut se rouvrir).
        # Effacé quand TL passe Green/Off OU quand dist > tl_detect_m (feu passé).
        self._tl_latch: bool = False

    @property
    def fsm_state(self) -> str:
        return self._fsm_state

    def run(
        self,
        ego_speed_ms: float,
        ego_heading_deg: float,
        ego_position: tuple,
    ) -> None:
        """Evalue la FSM et publie DecisionState dans le blackboard.

        Appeler AVANT planning_agent.run() dans la boucle de tick.

        Parameters
        ----------
        ego_speed_ms    : vitesse ego en m/s
        ego_heading_deg : cap ego en degrees (yaw CARLA, convention +X = 0 deg)
        ego_position    : (x, y) metres -- reserve pour etapes futures (non utilise)

        Sorties (via blackboard.decision) :
          fsm_state      : etat courant de la FSM
          target_speed   : vitesse cible en km/h (16 ou 6 selon etat FSM)
          branch_heading : cap souhaite au carrefour (None si pas de junction)
          is_at_junction : True si junction_in_chain (pour SafetyAgent etape 6)
          reason         : raison de la derniere transition (log / metriques)
        """
        self._tick_count += 1
        ego_speed_kmh = ego_speed_ms * 3.6

        # Lecture du blackboard (tick t-1 -- lag intentionnel, voir LIMITE 3)
        planning   = self._blackboard.planning
        perception = self._blackboard.perception

        # Signal FSM : vrai fork (len(nexts)>1), PAS is_junction seul.
        branch_approaching = planning.branch_in_chain
        branch_dist_m      = planning.branch_distance_m

        # Destination atteinte (GlobalRoutePlanner) : STOP immediat, court-circuite le reste.
        if planning.route_complete:
            if self._fsm_state != "STOP":
                logger.info("FSM transition: %s -> STOP | raison: destination_atteinte",
                            self._fsm_state)
                print(f"[FSM] t={self._tick_count:>4}  DESTINATION ATTEINTE -> STOP")
            self._fsm_state = "STOP"
            self._reason    = "destination_atteinte"
            self._blackboard.publish_decision(DecisionState(
                fsm_state="STOP",
                target_speed=0.0,
                branch_heading=None,
                is_at_junction=False,
                reason="destination_atteinte",
            ))
            return

        nearest_obs_m  = self._nearest_obstacle_m(perception.detections)
        # Hysterèse : une fois en SLOW_DOWN/STOP pour obstacle, maintient obstacle_close=True
        # jusqu'à d > obstacle_dist + hysteresis_m (évite lâcher/reprendre au seuil).
        _hyst = self._obs_slow_hysteresis_m
        if self._fsm_state in ("SLOW_DOWN", "STOP") and nearest_obs_m < self._obstacle_dist + _hyst:
            obstacle_close = True
        else:
            obstacle_close = nearest_obs_m < self._obstacle_dist

        # Traffic light fields (publies par CarlaTrafficLightPerception au tick t-1)
        tl_state  = perception.traffic_light_state   # "Red"|"Yellow"|"Green"|"Off"|"None"
        tl_dist_m = perception.traffic_light_dist_m  # distance ligne d'arret (m)

        # Stop sign field (publie par CarlaStopSignPerception au tick t-1)
        stop_sign_dist_m = perception.stop_sign_dist_m

        # FSM transitions (mise a jour complete chaque tick)
        v_lead_kmh = self._nearest_obstacle_v_lead(perception.detections)
        prev_state = self._fsm_state
        target_speed = self._update_fsm(
            branch_approaching, branch_dist_m,
            obstacle_close, nearest_obs_m,
            tl_state, tl_dist_m,
            stop_sign_dist_m,
            ego_speed_ms,
            v_lead_kmh,
        )

        if self._fsm_state != prev_state:
            logger.info(
                "FSM transition: %s -> %s | raison: %s",
                prev_state, self._fsm_state, self._reason,
            )

        # --- Log [FSM] inconditionnel (compact + info TL / stop sign si pertinent) ---
        tl_info = ""
        if tl_dist_m < float("inf"):
            _v2a = (ego_speed_ms ** 2) / (2 * self._tl_decel_mps2)
            _bd  = _v2a + self._tl_brake_margin_m
            tl_info = (
                f"  tl={tl_state:<8}"
                f"  tl_dist={tl_dist_m:>5.1f}m"
                f"  brake@<={_bd:>4.1f}m"
            )
        stop_info = ""
        if stop_sign_dist_m < float("inf"):
            _sv2a = (ego_speed_ms ** 2) / (2 * self._tl_decel_mps2)
            _sbd  = _sv2a + self._stop_brake_margin_m
            held  = self._stop_held_ticks
            stop_info = (
                f"  stop_dist={stop_sign_dist_m:>5.1f}m"
                f"  brake@<={_sbd:>4.1f}m"
                f"  held={held}/{self._stop_hold_ticks}"
            )
        print(
            f"[FSM] t={self._tick_count:>4}"
            f"  v={ego_speed_kmh:>6.2f} km/h"
            f"  fwd={nearest_obs_m:>7.2f}m"
            f"  close={'T' if obstacle_close else 'F'}"
            f"  fsm={self._fsm_state:<12}"
            f"  tgt={target_speed:>5.1f}"
            f"{tl_info}"
            f"{stop_info}"
        )

        # Branch heading : publie SEULEMENT si vrai fork dans la trajectoire.
        branch_heading: Optional[float] = None
        if branch_approaching:
            branch_heading = self._branch_policy.select_branch_heading(
                ego_heading_deg,
                ego_speed_kmh,
                self._blackboard.decision,
            )

        self._blackboard.publish_decision(DecisionState(
            fsm_state=self._fsm_state,
            target_speed=target_speed,
            branch_heading=branch_heading,
            is_at_junction=branch_approaching,
            reason=self._reason,
        ))

    # ------------------------------------------------------------------
    # Transitions FSM
    # ------------------------------------------------------------------

    def _update_fsm(
        self,
        branch_approaching: bool,
        branch_dist_m: float,
        obstacle_close: bool,
        nearest_obs_m: float,
        tl_state: str,
        tl_dist_m: float,
        stop_sign_dist_m: float,
        ego_speed_ms: float,
        v_lead_kmh: float = 0.0,
    ) -> float:
        """Applique les regles de transition de la FSM et retourne target_speed (km/h).

        COMPOSITION target_speed = min(tl_target, stop_target, obs_target).
        Priorite descendante : RED_LIGHT=STOP_SIGN (0) > SLOW_DOWN (slow) > FOLLOW_LANE (normal).

        TL -- regles ETAPE C (detection separee du freinage)
        -------------------------------------------------------
        DETECTION (armement) : dist <= tl_detect_m (50m) — TL visible meme via junction.
        FREINAGE  : dist <= dist_freinage + tl_brake_margin_m (~10m @ 16 km/h).
          Red  ET dist dans [0, frein+marge]  : RED_LIGHT, target=0.
          Red  ET dist dans ]frein+marge, 50m]: TL vu mais freinage pas encore justifie -> continue.
          None ET dist <= frein+marge         : RED_LIGHT conservateur (zone aveugle proche).
          None ET dist > frein+marge          : continue (etat inconnu mais trop loin).
          Yellow ET dist <= detect ET dist >= dist_freinage : RED_LIGHT (peut s'arreter a temps).
          Yellow ET dist <= detect ET dist < dist_freinage  : FOLLOW_LANE (zone dilemme, trop tard).
          Yellow ET dist > detect             : ignore (hors zone — actor scan depasse horizon).
          Green / Off                         : sortie RED_LIGHT si pas d'autre contrainte.

        Stop sign (detection separee du freinage — symétrique au TL)
        --------------------------------------------------------------
        DETECTION : dist <= stop_detect_m (50m) — vu tôt.
        FREINAGE  : dist <= dist_freinage + stop_brake_margin_m (~10m @ 16 km/h) -> STOP_SIGN.
        Apres stop_hold_ticks ticks a v<0.2 m/s : cleared=True -> resume FOLLOW_LANE.
        cleared reset quand dist >= stop_clear_dist_m (panneau passe ou nouveau loin).

        Obstacle / fork
        ---------------
        SLOW_DOWN si obstacle entre obstacle_dist et obs_stop_m (tgt=slow_speed).
        STOP si obstacle < obs_stop_m (tgt=0) — arrêt FSM avant zone Safety MIN_SAFE_M.
          Sans ce seuil, l'ego approche à slow_speed et entre dans MIN_SAFE_M (7.5m) →
          Safety bloque définitivement (latch relancé chaque tick, jamais libéré).

        YIELD / EMERGENCY : reserves etapes futures.
        """
        inf = float("inf")

        # ---- Calcul de la distance de freinage a vitesse courante ----
        # v²/(2*a) = distance minimale pour s'arreter avec deceleration a.
        # Recalcule chaque tick -> decroit quand l'ego ralentit.
        dist_freinage = (ego_speed_ms ** 2) / (2.0 * self._tl_decel_mps2)

        # ---- Condition TL (prioritaire, target=0) ----
        # SEPARATION detection / freinage :
        #   - TL "arme" des que dist <= tl_detect_m (50m), mais PAS de freinage encore.
        #   - Freinage declanche quand dist <= dist_freinage + tl_brake_margin_m (~10m @ 16 km/h).
        #   - L'ego roule normalement jusqu'a ~10m puis freine et s'arrete a la ligne.
        tl_stop = False
        tl_reason = ""
        brake_dist = dist_freinage + self._tl_brake_margin_m
        if tl_dist_m < inf:
            if tl_state == "Red":
                if tl_dist_m <= self._tl_detect_m:
                    if tl_dist_m <= brake_dist:
                        # Dans la fenetre de freinage : RED_LIGHT
                        tl_stop = True
                        tl_reason = (
                            f"feu_rouge d={tl_dist_m:.1f}m"
                            f" <= frein+marge={brake_dist:.1f}m -> 0 km/h"
                        )
                    # else: TL rouge detecte mais trop loin pour freiner -> continue
            elif tl_state == "None":
                # Zone aveugle (etat illisible a grande distance).
                # Conservateur UNIQUEMENT dans la fenetre de freinage — a 50m l'ego
                # ne doit pas s'arreter pour un etat inconnu.
                if tl_dist_m <= self._tl_detect_m and tl_dist_m <= brake_dist:
                    tl_stop = True
                    tl_reason = (
                        f"feu_inconnu d={tl_dist_m:.1f}m"
                        f" <= frein+marge={brake_dist:.1f}m -> conservateur"
                    )
            elif tl_state == "Yellow":
                if tl_dist_m <= self._tl_detect_m:
                    if tl_dist_m >= dist_freinage:
                        # Dans la zone de detection, assez loin pour s'arreter en securite
                        tl_stop = True
                        tl_reason = (
                            f"feu_jaune d={tl_dist_m:.1f}m"
                            f" >= frein={dist_freinage:.1f}m -> s'arreter"
                        )
                    else:
                        # Zone de dilemme : trop pres pour freiner sans danger -> passer
                        tl_reason = (
                            f"feu_jaune DILEMME d={tl_dist_m:.1f}m"
                            f" < frein={dist_freinage:.1f}m -> passer"
                        )
            # Green / Off -> tl_stop=False (pas de contrainte TL)

        # Latch anti-oscillation : une fois que la condition de freinage est satisfaite,
        # ne pas relâcher simplement parce que v décroît (et donc brake@ décroît aussi).
        # L'ego reste en tl_stop=True jusqu'à ce que le feu passe vert/off OU
        # que le feu soit trop loin (ego l'a passé).
        if tl_stop:
            self._tl_latch = True
        elif tl_state in ("Green", "Off") or tl_dist_m > self._tl_detect_m:
            self._tl_latch = False
        # Applique le latch si feu toujours rouge/inconnu et dans la portée
        if self._tl_latch and tl_state not in ("Green", "Off") and tl_dist_m <= self._tl_detect_m:
            tl_stop = True
            if not tl_reason:
                tl_reason = f"latch feu_rouge d={tl_dist_m:.1f}m -> 0 km/h"

        tl_target = 0.0 if tl_stop else self._normal_speed

        # ---- Condition stop sign (STOP_SIGN, target=0) ----
        # Reset cleared quand le panneau est loin (passé ou nouveau) -- doit preceder le test
        if stop_sign_dist_m >= self._stop_clear_dist_m:
            if self._stop_sign_cleared:
                self._stop_sign_cleared = False
                self._stop_held_ticks   = 0
                self._stop_latch        = False

        # SEPARATION detection / freinage (symétrique au TL) :
        #   - Armé quand stop_dist <= stop_detect_m (50m), mais PAS de freinage encore.
        #   - Freinage quand stop_dist <= dist_freinage + stop_brake_margin_m (~10m @ 16 km/h).
        stop_brake_dist = dist_freinage + self._stop_brake_margin_m
        stop_sign_stop = (
            not self._stop_sign_cleared
            and stop_sign_dist_m <= self._stop_detect_m
            and stop_sign_dist_m <= stop_brake_dist
        )

        # Latch anti-oscillation (v decroit → brake@ decroit → condition peut se rouvrir)
        if stop_sign_stop:
            self._stop_latch = True
        elif self._stop_sign_cleared or stop_sign_dist_m > self._stop_detect_m:
            self._stop_latch = False
        if self._stop_latch and not self._stop_sign_cleared and stop_sign_dist_m <= self._stop_detect_m:
            stop_sign_stop = True

        stop_target = 0.0 if stop_sign_stop else self._normal_speed
        stop_reason = ""
        if stop_sign_stop:
            stop_reason = (
                f"stop_sign a {stop_sign_dist_m:.1f}m <= frein+marge={stop_brake_dist:.1f}m"
                f"  hold={self._stop_held_ticks}/{self._stop_hold_ticks}"
            )

        # ---- Condition obstacle / fork (SLOW_DOWN → STOP si très proche) ----
        # Deux niveaux :
        #   obs_stop : obstacle < obs_stop_m  → tgt=0  (arrêt FSM avant zone Safety MIN_SAFE_M)
        #   obs_slow : obstacle < obstacle_dist OU fork proche → tgt=slow_speed
        # La composition min() fait que obs_stop (tgt=0) est plus restrictif que obs_slow (tgt=6).
        obs_stop   = False
        obs_slow   = False
        obs_reason = ""

        if obstacle_close:
            if nearest_obs_m < self._obs_stop_m:
                obs_stop   = True
                obs_reason = (
                    f"obstacle a {nearest_obs_m:.1f}m < {self._obs_stop_m}m"
                    f" -> arret (evite zone Safety {self._obs_stop_m}m)"
                )
            else:
                obs_slow   = True
                obs_reason = (
                    f"obstacle a {nearest_obs_m:.1f}m < {self._obstacle_dist}m"
                    f" -> {self._slow_speed} km/h"
                )
        elif branch_approaching and branch_dist_m < self._branch_near_m:
            obs_slow   = True
            obs_reason = (
                f"vrai fork a {branch_dist_m:.1f}m < {self._branch_near_m}m"
                f" -> {self._slow_speed} km/h"
            )

        if obs_stop:
            obs_target = 0.0
        elif obstacle_close:
            # Car-following (IDM-simplifié) : v_target adaptatif basé sur v_lead.
            # Ne s'applique que si un obstacle est détecté (pas fork seul, pas nearest=inf).
            obs_target = self._car_follow_target(
                nearest_obs_m=nearest_obs_m,
                v_lead_kmh=v_lead_kmh,
                ego_speed_ms=ego_speed_ms,
                v_cruise_kmh=self._normal_speed,
            )
        elif obs_slow:
            # Fork seul sans obstacle détecté : vitesse réduite fixe (régression pré-Scénario3).
            obs_target = self._slow_speed
        else:
            obs_target = self._normal_speed

        # ---- Composition : vitesse cible = contrainte la plus restrictive ----
        target_speed = min(tl_target, stop_target, obs_target)

        # ---- Etat FSM = raison principale (log / blackboard) ----
        if tl_stop:
            new_state  = "RED_LIGHT"
            new_reason = tl_reason
        elif stop_sign_stop:
            new_state  = "STOP_SIGN"
            new_reason = stop_reason
        elif obs_stop:
            new_state  = "STOP"        # arrêt FSM pour obstacle trop proche (avant zone Safety)
            new_reason = obs_reason
        elif obs_slow:
            new_state  = "SLOW_DOWN"
            new_reason = obs_reason
        else:
            new_state  = "FOLLOW_LANE"
            new_reason = "libre"

        # ---- Hold counter STOP_SIGN : arret obligatoire avant reprise ----
        if new_state == "STOP_SIGN":
            if ego_speed_ms < 0.2:
                self._stop_held_ticks += 1
                if self._stop_held_ticks >= self._stop_hold_ticks:
                    self._stop_sign_cleared = True
                    print(
                        f"[STOP_SIGN] t={self._tick_count:>4}"
                        f"  arret maintenu {self._stop_held_ticks} ticks -> CLEARED, reprise"
                    )
            else:
                self._stop_held_ticks = 0  # encore en mouvement, on ne compte pas
        else:
            # En dehors de STOP_SIGN : reset le compteur (si on sort sans avoir tenu)
            if not stop_sign_stop:
                self._stop_held_ticks = 0

        # Log dilemme YELLOW (inconditionnel si present)
        if tl_reason and "DILEMME" in tl_reason:
            print(f"[FSM_TL] t={self._tick_count:>4}  {tl_reason}")

        if new_state != self._fsm_state:
            self._reason = new_reason
        self._fsm_state = new_state

        return target_speed

        # YIELD / EMERGENCY : reserves etapes futures

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_obstacle_m(detections: list) -> float:
        """Distance a l'obstacle le plus proche depuis les detections Perception.

        Retourne float('inf') si aucune detection ou champ distance_m absent.
        """
        if not detections:
            return float("inf")
        distances = [
            # forward_dist : cle CarlaActorPerception + PerceptionAgent._enrich_detection
            # distance_m   : fallback PerceptionAgent ancienne cle (compatibilite)
            float(d.get("forward_dist", d.get("distance_m", float("inf"))))
            for d in detections
            if isinstance(d, dict)
        ]
        return min(distances) if distances else float("inf")

    @staticmethod
    def _nearest_obstacle_v_lead(detections: list) -> float:
        """Vitesse (km/h) de l'obstacle le plus proche sur la voie.

        Utilise v_lead_kmh publié par CarlaActorPerception.
        Retourne 0.0 si champ absent (walkers statiques, obstacles sans vitesse).
        """
        if not detections:
            return 0.0
        nearest = min(
            (d for d in detections if isinstance(d, dict)),
            key=lambda d: float(d.get("forward_dist", d.get("distance_m", float("inf")))),
            default=None,
        )
        if nearest is None:
            return 0.0
        return float(nearest.get("v_lead_kmh", 0.0))

    def _car_follow_target(
        self,
        nearest_obs_m: float,
        v_lead_kmh: float,
        ego_speed_ms: float,
        v_cruise_kmh: float,
    ) -> float:
        """Loi IDM-simplifiée : vitesse cible [km/h] pour le car-following.

        Corrige 2 défauts d'une loi distance-seule :
        1. Rattrapage : quand loin (err_d > 0), v_target > v_lead → l'ego se rapproche.
        2. Anti-oscillation : terme k_vel × (v_ego - v_lead) amorti les oscillations
           (freine AVANT de coller, ne sur-réagit pas quand s'éloigne).

        Dégénère correctement vers 0 quand v_lead=0 (lead arrêté → non-régression Scén 1).
        """
        v_ego_kmh = ego_speed_ms * 3.6

        # Distance de sécurité : règle des 2 secondes, avec plancher obs_stop_m
        d_secu = max(self._obs_stop_m, self._cf_headway_s * ego_speed_ms)

        err_d = nearest_obs_m - d_secu     # + = trop loin (rattraper), - = trop près
        err_v = v_ego_kmh - v_lead_kmh     # + = on se rapproche (freiner), - = on s'éloigne

        v_target = v_lead_kmh + self._cf_k_dist * err_d - self._cf_k_vel * err_v
        return max(0.0, min(v_target, v_cruise_kmh))

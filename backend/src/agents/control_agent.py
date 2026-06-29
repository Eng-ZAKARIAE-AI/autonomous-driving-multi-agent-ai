"""Control Agent — PID longitudinal + latéral.

Rôle dans l'architecture
------------------------
Le Control Agent est le seul composant qui écrit des commandes sur le véhicule.
Il lit la trajectoire publiée par le Planning Agent dans le blackboard et produit
{steer, throttle, brake} à chaque tick CARLA.

Fréquence
---------
GPU mesuré (NVIDIA T1200, torch 2.4.1+cu124) :
  YOLOv8n @ 640x640 = 8.7 ms/frame → 41 ms de marge dans un tick de 50 ms.
La perception tourne à CHAQUE tick (20 Hz, perception_stride=1). La mitigation
de perception throttlée (perception_stride=2) n'est plus nécessaire sur GPU.

Règle de robustesse (CPU fallback)
-----------------------------------
Si perception_stride > 1 (CPU, ~60 ms/frame), le PID ne gèle JAMAIS sa commande :
il continue de calculer sur la dernière trajectoire connue dans le blackboard.

Extension MPC
-------------
LateralController et LongitudinalController sont des ABCs. Pour basculer vers MPC :
    agent.set_lateral_controller(MpcLateralController(horizon=10, dt=0.05))
L'orchestrateur n'a pas à changer.

Sélection de waypoint
---------------------
_select_target_waypoint() utilise un lookahead fixe (~5m). La liste reçue du
Planning Agent (étape 4) sera triée distance croissante ; le lookahead cible
naturellement le 2e waypoint (à ~6m). Même logique pour la DUMMY_TRAJECTORY
statique de l'étape 3 (les waypoints derrière l'ego sont ignorés automatiquement).
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from backend.src.agents.base_agent import BaseAgent
from backend.src.agents.blackboard import Blackboard, ControlState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Controller ABCs — point d'extension MPC (soutenance : interface prête)
# ---------------------------------------------------------------------------

class LateralController(ABC):
    """Interface pour les contrôleurs latéraux (direction).

    Extension point
    ---------------
    Pour ajouter MPC, pure-pursuit, Stanley etc. sans modifier ControlAgent :

        class MpcLateralController(LateralController):
            def compute(self, heading_error_norm, dt): ...

        agent.set_lateral_controller(MpcLateralController(...))

    Contrat de l'unique méthode abstraite :
        - heading_error_norm : erreur de cap normalisée dans [-1, 1] (degrés / 180)
        - dt                 : temps écoulé depuis le dernier appel (secondes)
        - retour             : commande steer dans [-max_steer, max_steer]
    """

    @abstractmethod
    def compute(self, heading_error_norm: float, dt: float) -> float:
        """Retourne la commande steer dans [-max_steer, max_steer]."""

    def reset(self) -> None:
        """Appelé en début d'épisode — surcharger si le contrôleur est stateful."""


class LongitudinalController(ABC):
    """Interface pour les contrôleurs longitudinaux (throttle / brake).

    Contrat :
        - speed_error_ms : target_speed_ms − current_speed_ms
                           Positif → accélérer ; négatif → freiner.
        - retour         : (throttle, brake), chacun dans [0, 1].
    """

    @abstractmethod
    def compute(self, speed_error_ms: float, dt: float) -> Tuple[float, float]:
        """Retourne (throttle, brake)."""

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# PidController — primitive scalaire (inchangé depuis étape 1)
# ---------------------------------------------------------------------------

class PidController:
    """PID scalaire avec anti-windup et suppression du pic dérivé au premier appel."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float,
        output_max: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = output_min
        self.out_max = output_max
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time: Optional[float] = None
        # Supprime le pic dérivé sur le premier appel (prev_error = 0 → D spike)
        self._first_call = True
        # Composantes exposées pour diagnostic (mises à jour à chaque compute)
        self._last_p: float = 0.0
        self._last_d: float = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0:
            dt = 0.05

        self._integral += error * dt
        # Anti-windup : clamp intégrateur pour que ki*integral reste dans [out_min, out_max]
        max_integral = (self.out_max - self.out_min) / max(self.ki, 1e-9)
        self._integral = max(-max_integral, min(max_integral, self._integral))

        # Premier appel : initialiser prev_error = error (évite D = error/dt au tick 0)
        if self._first_call:
            self._prev_error = error
            self._first_call = False

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        self._last_p = self.kp * error
        self._last_d = self.kd * derivative
        output = self._last_p + self.ki * self._integral + self._last_d
        return max(self.out_min, min(self.out_max, output))

    @property
    def integral(self) -> float:
        """Valeur courante de l'intégrateur (lecture seule, pour diagnostic)."""
        return self._integral

    @property
    def last_p(self) -> float:
        """Terme P du dernier appel compute() — diagnostic."""
        return self._last_p

    @property
    def last_d(self) -> float:
        """Terme D du dernier appel compute() — diagnostic."""
        return self._last_d

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_call = True
        self._last_p = 0.0
        self._last_d = 0.0


# ---------------------------------------------------------------------------
# PID implementations
# ---------------------------------------------------------------------------

class PidLateralController(LateralController):
    """PID latéral opérant sur l'erreur de cap normalisée.

    Gains de départ (documentés pour la soutenance) :
      kp = 1.2  : à 90° d'erreur (norm = 0.5) → steer = 0.6 (75% du max 0.8)
                  Réponse vive sans saturation immédiate.
      ki = 0.0  : pas d'intégrale latérale — écarté car sur une courbe
                  heading_error ≠ 0 est le signal de pilotage de la courbe,
                  PAS une perturbation constante. ki l'accumulerait → survirage.
      kd = 0.1  : amortit les oscillations de cap. Modéré pour ne pas
                  amplifier le bruit de heading CARLA (±0.5° typique →
                  pic D = 0.1 × 0.5/180 / 0.05 = 0.006, négligeable).

    Feedforward de courbure (géré par ControlAgent, pas ici) :
      steer_ff = k_ff × κ_local  ajouté APRÈS le PID pour anticiper les virages.
      Voir ControlAgent._compute_curvature_ff() et config steer_kff.
    """

    def __init__(self, kp: float, ki: float, kd: float, max_steer: float) -> None:
        self._pid = PidController(kp, ki, kd, -max_steer, max_steer)

    def compute(self, heading_error_norm: float, dt: float) -> float:
        return self._pid.compute(heading_error_norm, dt)

    @property
    def integral(self) -> float:
        return self._pid.integral

    @property
    def last_p(self) -> float:
        """Terme P latéral du dernier tick (diagnostic)."""
        return self._pid.last_p

    @property
    def last_d(self) -> float:
        """Terme D latéral du dernier tick (diagnostic)."""
        return self._pid.last_d

    def reset(self) -> None:
        self._pid.reset()


class PidLongitudinalController(LongitudinalController):
    """PID longitudinal avec normalisation de la vitesse (anti bang-bang).

    Problème sans normalisation
    ---------------------------
    speed_error_ms peut valoir 8–10+ m/s au démarrage. Avec le PID borné
    à [-1, 1] et kp ≥ 0.1, le terme P sature immédiatement → le contrôleur
    se comporte en bang-bang (throttle = 1.0 ou 0.0) et perd sa dynamique.

    Solution : normaliser speed_error_ms par speed_ref_ms avant le PID
    -------------------------------------------------------------------
    error_norm = speed_error_ms / speed_ref_ms  ∈ [-1, 1] approximativement.
    Le PID voit des entrées bornées ; les gains restent interprétables.

    Gains de départ :
      kp = 0.6  : à speed_ref d'erreur (11.1 m/s = 40 km/h), sortie = 0.6 (60%)
                  Au démarrage (erreur ≈ 8 m/s), throttle ≈ 0.43 — pas de bang-bang.
      ki = 0.02 : accumulation lente pour compenser les frottements. L'anti-windup
                  borne l'intégrale : ki × integral_max = 0.02 × 100 = 2 (clampé à 1).
      kd = 0.0  : pas de dérivée longitudinale — amplifierait le bruit de vitesse CARLA.
      speed_ref = 11.1 m/s (= 40 km/h) — référence de normalisation.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        max_throttle: float,
        max_brake: float,
        speed_ref_ms: float = 11.1,  # 40 km/h
    ) -> None:
        self._pid = PidController(kp, ki, kd, -1.0, 1.0)
        self._max_throttle = max_throttle
        self._max_brake = max_brake
        self._speed_ref = speed_ref_ms

    def compute(self, speed_error_ms: float, dt: float) -> Tuple[float, float]:
        error_norm = speed_error_ms / self._speed_ref  # ramène à ~[-1, 1]
        out = self._pid.compute(error_norm, dt)
        if out >= 0:
            return min(out, self._max_throttle), 0.0
        else:
            return 0.0, min(-out, self._max_brake)

    @property
    def integral(self) -> float:
        return self._pid.integral

    def reset(self) -> None:
        self._pid.reset()


# ---------------------------------------------------------------------------
# ControlAgent
# ---------------------------------------------------------------------------

_DEFAULT_LOOKAHEAD_M = 5.0  # lookahead waypoint selection (metres)


class ControlAgent(BaseAgent):
    """PID latéral + PID longitudinal → {steer, throttle, brake} dans le blackboard.

    Parameters
    ----------
    blackboard   : Blackboard partagé
    config       : dict de configuration complet
    lateral      : LateralController (None → PidLateralController depuis config)
    longitudinal : LongitudinalController (None → PidLongitudinalController depuis config)
    """

    def __init__(
        self,
        blackboard: Blackboard,
        config: Dict[str, Any],
        lateral: Optional[LateralController] = None,
        longitudinal: Optional[LongitudinalController] = None,
    ) -> None:
        super().__init__(blackboard, config)

        ctrl = config.get("agents", {}).get("control", {})
        self._fixed_dt = float(config.get("carla", {}).get("fixed_delta_seconds", 0.05))
        self._max_steer = float(ctrl.get("max_steer", 0.8))
        self._max_throttle = float(ctrl.get("max_throttle", 1.0))
        self._max_brake = float(ctrl.get("max_brake", 0.7))
        self._lookahead_m = float(ctrl.get("lookahead_distance", _DEFAULT_LOOKAHEAD_M))

        speed_ref_ms = float(ctrl.get("speed_ref_kmh", 40.0)) / 3.6

        self._lateral: LateralController = lateral or PidLateralController(
            kp=float(ctrl.get("steer_kp", 1.2)),
            ki=float(ctrl.get("steer_ki", 0.0)),
            kd=float(ctrl.get("steer_kd", 0.1)),
            max_steer=self._max_steer,
        )
        
        self._longitudinal: LongitudinalController = longitudinal or PidLongitudinalController(
            kp=float(ctrl.get("speed_kp", 0.6)),
            ki=float(ctrl.get("speed_ki", 0.02)),
            kd=float(ctrl.get("speed_kd", 0.0)),
            max_throttle=self._max_throttle,
            max_brake=self._max_brake,
            speed_ref_ms=speed_ref_ms,
        )

        # --- Feedforward de courbure ---
        # steer_ff = k_ff × κ_local  où κ = Δheading(rad)/dist(m) moyenné sur
        # les premiers curvature_smooth_n segments de la trajectoire.
        # Elimine le retard géométrique de poursuite en courbe (CTE ~0.55m → <0.3m)
        # sans ki (qui accumulerait le heading_error de pilotage → survirage).
        # Calibration : lancer spawn 60 --diag-all, vérifier colonne curv,
        #               ajuster steer_kff jusqu'à CTE<0.3m sur toute la courbe.
        plan_cfg = config.get("agents", {}).get("planning", {})
        self._steer_kff: float         = float(ctrl.get("steer_kff", 0.0))
        # Empattement du véhicule (m) — paramètre physique, non calibrable.
        # Tesla Model3 CARLA ≈ 2.87m. Modifier si autre véhicule.
        # Rôle : steer_ff = k_ff × κ × wheelbase  (modèle bicyclette Ackermann).
        # Sans wheelbase, k_ff absorberait le facteur d'empattement dans une constante
        # opaque qui change d'un véhicule à l'autre.
        self._wheelbase_m: float       = float(ctrl.get("wheelbase_m", 2.87))
        self._curvature_smooth_n: int  = int(ctrl.get("curvature_smooth_n", 5))
        self._curvature_max: float     = float(ctrl.get("curvature_max_radpm", 0.10))
        # Espacement nominal des waypoints (pour normalisation si dist_segment ~ 0)
        self._wp_spacing: float        = float(plan_cfg.get("waypoint_spacing", 2.0))
        # Rate-limiter : borne la variation de steer_ff par tick.
        # N'agit QUE pendant la montée curv 0→κ (transitoire d'entrée d'arc).
        # Transparent en régime établi (curv constant → steer_ff constant → delta=0).
        # 0.0 = désactivé (échelon pur, comportement pré-lissage).
        self._steer_ff_rate_max: float = float(ctrl.get("steer_ff_rate_max", 0.0))
        # État du rate-limiter : valeur ff du tick précédent
        self._steer_ff_prev: float = 0.0
        # Lookahead ff : nombre de waypoints à sauter avant la fenêtre de courbure.
        # ff calcule kappa à _ff_lookahead_wps devant l'ego → anticipe l'arc entrant.
        # Converti depuis ff_lookahead_m / waypoint_spacing, arrondi au wp entier.
        _ff_lookahead_m: float = float(ctrl.get("ff_lookahead_m", 0.0))
        self._ff_lookahead_wps: int = max(0, round(_ff_lookahead_m / max(self._wp_spacing, 0.1)))
        # Terme latéral Stanley : arctan(k_cte × CTE / max(v, v_min)).
        # Introduit le CTE dans la commande (PID de cap ne corrige que le heading).
        # 0.0 = désactivé (non-régression : comportement identique à avant).
        self._steer_k_cte: float = float(ctrl.get("steer_k_cte", 0.0))
        self._cte_v_min: float   = float(ctrl.get("cte_v_min", 1.0))
        print(
            f"[ControlAgent] kff={self._steer_kff:.3f}  "
            f"wheelbase={self._wheelbase_m:.2f}m  "
            f"rate_max={self._steer_ff_rate_max:.4f}  "
            f"curv_max={self._curvature_max:.3f}  "
            f"lookahead_wps={self._ff_lookahead_wps}  "
            f"k_cte={self._steer_k_cte:.3f}"
        )
        # Dernière estimation de courbure locale (exposée pour le diagnostic demo)
        self._last_curvature_radpm: float = 0.0
        # Dernière valeur du terme feedforward steer (avant clamp final)
        self._last_steer_ff: float = 0.0
        # Valeur cible ff AVANT rate-limiter (diagnostic ramp)
        self._last_steer_ff_target: float = 0.0
        # Terme Stanley et CTE au dernier tick (diagnostic)
        self._last_cte: float      = 0.0
        self._last_cte_term: float = 0.0
        # Compteur de ticks (diagnostic uniquement — numérotation dans les logs)
        self._tick_count: int = 0

        self._last_tick_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Controller swap — extension MPC sans modifier l'orchestrateur
    # ------------------------------------------------------------------

    @property
    def longitudinal_integral(self) -> float:
        """Intégrale courante du PID longitudinal (diagnostic uniquement)."""
        if isinstance(self._longitudinal, PidLongitudinalController):
            return self._longitudinal.integral
        return 0.0

    @property
    def last_steer_p(self) -> float:
        """Terme P du PID latéral au dernier tick (avant clamp + ff).

        steer_pid = last_steer_p + last_steer_d  (ki=0 donc pas de terme I latéral).
        Exposé pour décomposer steer_total = steer_p + steer_d + steer_ff.
        """
        if isinstance(self._lateral, PidLateralController):
            return self._lateral.last_p
        return 0.0

    @property
    def last_steer_d(self) -> float:
        """Terme D du PID latéral au dernier tick (diagnostic)."""
        if isinstance(self._lateral, PidLateralController):
            return self._lateral.last_d
        return 0.0

    @property
    def last_steer_ff(self) -> float:
        """Terme feedforward de courbure au dernier tick (diagnostic)."""
        return self._last_steer_ff

    @property
    def last_steer_ff_target(self) -> float:
        """Valeur ff cible AVANT rate-limiter : k_ff × κ × L (diagnostic ramp)."""
        return self._last_steer_ff_target

    @property
    def last_curvature_radpm(self) -> float:
        """Dernière estimation de courbure locale (rad/m), mise à jour chaque tick.

        Valeur brute (après clamp à curvature_max_radpm, avant k_ff).
        Positive = courbe à droite (CARLA convention, y=south).
        Exposée pour le diagnostic demo : colonne curv= en diag.
        """
        return self._last_curvature_radpm

    @property
    def last_cte(self) -> float:
        """Dernière erreur cross-track calculée (m) — diagnostic Stanley."""
        return self._last_cte

    @property
    def last_cte_term(self) -> float:
        """Dernier terme Stanley appliqué au steer (arctan saturé) — diagnostic."""
        return self._last_cte_term

    @property
    def longitudinal_integral_cap(self) -> float:
        """Borne anti-windup : ki*integral <= 1.0 => integral_max = 1/ki."""
        if isinstance(self._longitudinal, PidLongitudinalController):
            ki = self._longitudinal._pid.ki
            return 2.0 / max(ki, 1e-9)
        return float("inf")

    def set_lateral_controller(self, ctrl: LateralController) -> None:
        """Swap latéral à chaud (ex : MPC). L'orchestrateur n'a pas à changer."""
        ctrl.reset()
        self._lateral = ctrl

    def set_longitudinal_controller(self, ctrl: LongitudinalController) -> None:
        """Swap longitudinal à chaud."""
        ctrl.reset()
        self._longitudinal = ctrl

    # ------------------------------------------------------------------
    # Interface principale — appelée à chaque tick par l'Orchestrateur
    # ------------------------------------------------------------------

    def run(
        self,
        current_speed_ms: float,
        current_heading_deg: float,
        ego_position: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Calcule et publie les commandes à chaque tick CARLA.

        Parameters
        ----------
        current_speed_ms    : vitesse du véhicule en m/s
        current_heading_deg : cap (yaw) en degrés, issu de CARLA transform
        ego_position        : (x, y) coordonnées monde.
                              Passé directement par l'Orchestrateur (préféré).
                              Si None → lit blackboard.perception.lane_geometry
                              (quand PerceptionAgent tourne, étapes 4+).
        """
        dt = self._get_dt()
        self._tick_count += 1
        waypoints = self.blackboard.planning.waypoints

        if not waypoints:
            # Pas de trajectoire — freinage doux de sécurité
            self.blackboard.publish_control(ControlState(steer=0.0, throttle=0.0, brake=0.3))
            return

        # Résolution de la position ego
        if ego_position is None:
            lane_geo = getattr(self.blackboard.perception, "lane_geometry", {})
            ego_position = lane_geo.get("ego_position", (0.0, 0.0))
        ego_x, ego_y = float(ego_position[0]), float(ego_position[1])

        # Sélection du waypoint cible
        target_wp = self._select_target_waypoint(waypoints, ego_x, ego_y, current_heading_deg)

        if target_wp is None:
            # Trajectoire terminee : freinage doux, steer=0 (ligne droite)
            logger.debug("Tous les waypoints sont depassés — freinage fin de trajectoire")
            self.blackboard.publish_control(ControlState(steer=0.0, throttle=0.0, brake=0.3))
            return

        # PID latéral + feedforward de courbure + terme Stanley CTE
        steer_pid = self._compute_lateral(target_wp, ego_x, ego_y, current_heading_deg, dt)
        steer_ff  = self._compute_curvature_ff(waypoints)
        cte       = self._compute_cte(waypoints, ego_x, ego_y)
        steer_cte = self._compute_stanley_cte_term(cte, current_speed_ms)
        steer     = max(-self._max_steer, min(self._max_steer, steer_pid + steer_ff + steer_cte))
        # [STANLEY] inconditionnel — prouve exécution et signe du terme CTE à chaque tick.
        # CTE<0 = intérieur virage → cte_term<0 attendu. Filtre : Select-String "\[STANLEY\]"
        print(
            f"[STANLEY] t={self._tick_count:>4}  CTE={cte:+.3f}  v={current_speed_ms:.2f}  "
            f"cte_term={steer_cte:+.4f}  sp={self.last_steer_p:+.4f}  "
            f"sff={self._last_steer_ff:+.4f}  steer={steer:+.4f}"
        )

        # PID longitudinal
        target_speed_kmh = float(target_wp[2]) if len(target_wp) > 2 else 16.0
        target_speed_ms = target_speed_kmh / 3.6
        if target_speed_ms <= 0.1:
            throttle, brake = 0.0, self._max_brake
        else:
            throttle, brake = self._longitudinal.compute(target_speed_ms - current_speed_ms, dt)

        self.blackboard.publish_control(ControlState(steer=steer, throttle=throttle, brake=brake))

    def reset(self) -> None:
        """Réinitialise les contrôleurs en début d'épisode."""
        self._lateral.reset()
        self._longitudinal.reset()
        self._last_tick_time = None
        self._steer_ff_prev = 0.0   # rate-limiter : repart de 0 à chaque épisode
        self._last_cte = 0.0
        self._last_cte_term = 0.0
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_target_waypoint(
        self,
        waypoints: List[Tuple],
        ego_x: float,
        ego_y: float,
        ego_heading_deg: float,
    ) -> Optional[Tuple]:
        """Selectionne le waypoint cible avec lookahead fixe (~5m).

        Retourne None quand tous les waypoints sont derriere l'ego (trajectoire terminee).
        Dans ce cas, run() publie un freinage doux.

        Regle :
          1. Calcule la projection de chaque waypoint sur l'axe avant de l'ego.
          2. Ignore les waypoints clairement derriere (projection < -1m, tolerance 1m).
          3. Parmi les candidats restants, retourne le premier a distance >= lookahead.
          4. Si tous sont plus proches, retourne le plus eloigne disponible devant.
          5. Si tous sont derriere : retourne None (mission terminee -> freinage).

        Interface Planning Agent (etape 4) :
          La liste sera triee distance croissante, tous devant l'ego.
          Avec spacing=3m : waypoints[0]~3m, waypoints[1]~6m -> lookahead=5m
          cible naturellement waypoints[1]. Pas d'effet de bord sur le blackboard.
        """
        fwd_x = math.cos(math.radians(ego_heading_deg))
        fwd_y = math.sin(math.radians(ego_heading_deg))

        candidates: List[Tuple[float, Tuple]] = []
        for wp in waypoints:
            dx = float(wp[0]) - ego_x
            dy = float(wp[1]) - ego_y
            proj = dx * fwd_x + dy * fwd_y
            dist = math.hypot(dx, dy)
            if proj > -1.0:
                candidates.append((dist, wp))

        if not candidates:
            # Tous les waypoints sont derriere : trajectoire terminee
            return None

        candidates.sort(key=lambda t: t[0])
        for dist, wp in candidates:
            if dist >= self._lookahead_m:
                return wp
        return candidates[-1][1]

    def _compute_lateral(
        self,
        target_wp: Tuple,
        ego_x: float,
        ego_y: float,
        current_heading_deg: float,
        dt: float,
    ) -> float:
        """Erreur de cap vers le waypoint cible → steer via PID latéral."""
        dx = float(target_wp[0]) - ego_x
        dy = float(target_wp[1]) - ego_y

        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return 0.0

        target_heading_deg = math.degrees(math.atan2(dy, dx))
        heading_error = target_heading_deg - current_heading_deg

        # Normalise à [-180, 180]
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360

        heading_error_norm = heading_error / 180.0
        return self._lateral.compute(heading_error_norm, dt)

    def _compute_curvature_ff(self, waypoints: List[Tuple]) -> float:
        """Feedforward de courbure : steer_ff = k_ff × κ_local (rad/m).

        Algorithme
        ----------
        κ_local = moyenne de κ_i = Δheading_i(rad) / dist_i(m)
                  sur les premiers curvature_smooth_n segments de la trajectoire.

        Convention de signe (cohérente avec _compute_lateral) :
          κ > 0 = courbe à droite (CARLA : y pointe au Sud, yaw croissant = horaire).
          steer_ff > 0 = ordre droite. k_ff > 0 est le signe correct pour compenser
          le retard de poursuite sur une courbe droite (et automatiquement gauche
          si κ < 0).

        Protection contre waypoints aberrants
        --------------------------------------
          1. Segments trop courts (dist < 0.1m) : ignorés (division par zéro).
          2. κ_local clampé à [-curvature_max, +curvature_max] avant application
             de k_ff. Exemple : curvature_max=0.10 rad/m = rayon 10m (virage serré).
          3. steer_ff clampé à ±50% de max_steer : le FF ne peut pas dominer le
             signal P+D même si k_ff est sur-calibré.

        Retourne 0.0 si k_ff=0 ou moins de 3 waypoints (pas de Δheading estimable).
        """
        if self._steer_kff == 0.0 or len(waypoints) < 3:
            self._last_curvature_radpm = 0.0
            return 0.0

        # Lookahead : décale la fenêtre de courbure en avant de l'ego.
        # ff voit la courbure à venir (skip wps) plutôt qu'à la position courante.
        # Skip borné pour garder curvature_smooth_n+2 waypoints disponibles.
        skip = min(self._ff_lookahead_wps, max(0, len(waypoints) - self._curvature_smooth_n - 2))
        wps = waypoints[skip:] if skip > 0 else waypoints
        # [LOOKAHEAD] tick%10 — skip confirmé fonctionnel (diff=+2.000 prouvé).
        if self._tick_count % 10 == 0 and len(wps) >= 1:
            print(
                f"[LOOKAHEAD] t={self._tick_count:>4}  "
                f"skip={skip}  "
                f"wps0_x={float(wps[0][0]):.3f}  "
                f"raw0_x={float(waypoints[0][0]):.3f}  "
                f"diff={float(wps[0][0])-float(waypoints[0][0]):+.3f}"
            )

        kappas: List[float] = []
        # On a besoin de N+1 waypoints pour N segments, et N-1 paires de segments.
        # Donc pour curvature_smooth_n paires, il faut curvature_smooth_n+2 waypoints.
        n_pairs = min(self._curvature_smooth_n, len(wps) - 2)

        for i in range(n_pairs):
            ax, ay = float(wps[i][0]),   float(wps[i][1])
            bx, by = float(wps[i+1][0]), float(wps[i+1][1])
            cx, cy = float(wps[i+2][0]), float(wps[i+2][1])

            d_ab = math.hypot(bx - ax, by - ay)
            if d_ab < 0.1:
                continue  # segment dégénéré (deux waypoints superposés)

            h_ab = math.atan2(by - ay, bx - ax)  # en radians
            h_bc = math.atan2(cy - by, cx - bx)

            dh = h_bc - h_ab
            # Normalisation à [-π, π]
            dh = (dh + math.pi) % (2.0 * math.pi) - math.pi

            kappas.append(dh / d_ab)  # Δheading(rad) / longueur_segment(m)

        if not kappas:
            self._last_curvature_radpm = 0.0
            return 0.0

        kappa_local = sum(kappas) / len(kappas)

        # Clamp anti-aberrant (waypoint isolé au bord de segment CARLA)
        kappa_local = max(-self._curvature_max, min(self._curvature_max, kappa_local))
        self._last_curvature_radpm = kappa_local

        # steer_ff = k_ff × κ × L  (modèle bicyclette Ackermann)
        # k_ff ≈ 1/δ_max_roue (gain pur, calibrable sur run CARLA)
        # L = wheelbase_m (constante physique du véhicule, NON calibrée)
        steer_ff_target = self._steer_kff * kappa_local * self._wheelbase_m
        self._last_steer_ff_target = steer_ff_target  # exposé pour diagnostic

        # Rate-limiter : lisse le ff pendant le transitoire d'entrée d'arc.
        # N'agit QUE quand steer_ff varie (curv 0→κ ou κ→0).
        # En régime établi (curv constant) : target==prev → delta=0 → transparent.
        if self._steer_ff_rate_max > 0.0:
            delta_raw = steer_ff_target - self._steer_ff_prev
            delta = max(-self._steer_ff_rate_max, min(self._steer_ff_rate_max, delta_raw))
            steer_ff_limited = self._steer_ff_prev + delta
            if abs(delta) < abs(delta_raw) - 1e-9:
                # Le rate-limiter a clampé : print une fois par tick où il mord
                print(
                    f"[RL-FF t={self._tick_count:>4}] "
                    f"target={steer_ff_target:+.4f}  "
                    f"prev={self._steer_ff_prev:+.4f}  "
                    f"delta_raw={delta_raw:+.4f}  "
                    f"clamped={delta:+.4f}  "
                    f"-> ff={steer_ff_limited:+.4f}"
                )
        else:
            steer_ff_limited = steer_ff_target  # désactivé : échelon pur

        self._steer_ff_prev = steer_ff_limited  # mémorise pour le prochain tick

        # Le FF ne doit pas saturer le steer seul (borne à 50% de max_steer)
        ff_cap = self._max_steer * 0.5
        self._last_steer_ff = max(-ff_cap, min(ff_cap, steer_ff_limited))
        return self._last_steer_ff

    def _compute_cte(
        self,
        waypoints: List[Tuple],
        ego_x: float,
        ego_y: float,
    ) -> float:
        """Distance signée ego → segment [waypoints[0]→waypoints[1]].

        Formule : (ex*dy - ey*dx) / seg_len   (convention cross_track_error du demo)
          Négatif = ego à l'intérieur (côté sud) d'un virage droit CARLA.
          Positif = ego à l'extérieur (côté nord).
        Expose _last_cte pour la propriété last_cte (diagnostic).
        """
        if len(waypoints) < 2:
            if waypoints:
                self._last_cte = math.hypot(
                    ego_x - float(waypoints[0][0]),
                    ego_y - float(waypoints[0][1]),
                )
            else:
                self._last_cte = 0.0
            return self._last_cte
        ax, ay = float(waypoints[0][0]), float(waypoints[0][1])
        bx, by = float(waypoints[1][0]), float(waypoints[1][1])
        dx, dy  = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-3:
            self._last_cte = math.hypot(ego_x - ax, ego_y - ay)
            return self._last_cte
        self._last_cte = ((ego_x - ax) * dy - (ego_y - ay) * dx) / seg_len
        return self._last_cte

    def _compute_stanley_cte_term(self, cte: float, v: float) -> float:
        """Terme latéral Stanley : arctan(k_cte × CTE / max(v, v_min)).

        CTE < 0 → terme < 0 → steer gauche → ramène vers le centre.
        Corrige l'erreur de POSITION (que le PID de cap ne régule pas directement).
        k_cte = 0.0 → retourne 0.0 (non-régression totale, désactivé par défaut).
        Borné à ±max_steer : ne peut pas saturer le steer à lui seul.
        """
        if self._steer_k_cte == 0.0:
            self._last_cte_term = 0.0
            return 0.0
        v_eff = max(v, self._cte_v_min)
        raw = math.atan(self._steer_k_cte * cte / v_eff)
        self._last_cte_term = max(-self._max_steer, min(self._max_steer, raw))
        return self._last_cte_term

    def _get_dt(self) -> float:
        # En mode synchrone CARLA (fixed_delta_seconds=0.05), chaque world.tick()
        # avance la simulation de exactement 0.05s independamment du temps reel.
        # On utilise toujours fixed_dt pour que ki/kd soient calibres sur le temps
        # simulation et non sur le temps CPU (qui varie selon la charge machine).
        return self._fixed_dt

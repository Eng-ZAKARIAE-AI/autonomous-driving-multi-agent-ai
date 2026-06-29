"""SafetyAgent -- veto transversal pour le freinage d'urgence.

Architecture
------------
SafetyAgent lit blackboard.perception.detections a chaque tick pour calculer
le Time-To-Collision (TTC). Quand TTC < TTC_EMERGENCY, il publie safety.override=True
(veto). L'Orchestrateur (ou la boucle demo) verifie ce flag AVANT d'appliquer les
commandes CARLA, et applique brake=0.7 / throttle=0 si le veto est actif.

SafetyAgent ecrit UNIQUEMENT dans blackboard.safety. Il ne touche pas aux slots
control, planning ou decision -- c'est l'appelant qui agit sur le veto.

Design pattern : veto par flag, pas par surcharge de commandes.
L'appelant reste responsable de l'application CARLA; le Safety Agent est un
"oracle" dont le flag est prioritaire dans la boucle.

Calibration (mesure reelle CARLA Town10HD_Opt, spawn 60, brake=0.7) :
  d_stop = 2.99m a v0=4.85 m/s -> TTC_kinematique = 2.99/4.85 + 0.5 = 1.12s
  TTC_EMERGENCY = 1.5s (marge 4.3m supplementaire, zone tampon confortable)
  TTC_CLEAR     = 3.0s (hysteresis de liberation du latch)
  latch_ticks   = 20  (~1s a 20 Hz -- evite des librations si obstacle oscille)

Communication
-------------
  Entree : blackboard.perception.detections  (CarlaActorPerception ou PerceptionAgent)
  Sortie : blackboard.safety (override, ttc, reason, interventions_count, log)

Logs [SAFETY]
-------------
  Tous les ticks si TTC < 5m (information utile)
  Tous les 20 ticks sinon (bruit minimal en regime nominal)
  [SAFETY] VETO : debut d'intervention
  [SAFETY] CLEAR : fin d'intervention (apres latch + TTC_CLEAR)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.src.agents.blackboard import Blackboard, SafetyState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RuleBasedSafetyPolicy — politique TTC sans etat (stateless)
# ---------------------------------------------------------------------------

class RuleBasedSafetyPolicy:
    """Evalue le TTC sur la liste de detections et decide si le freinage est necessaire.

    Stateless : tout l'etat (latch, compteurs) est gere par SafetyAgent.
    Separee de SafetyAgent pour permettre le remplacement par une politique RL
    (PPO lane-decision hook) sans modifier SafetyAgent.

    Extension point
    ---------------
    Pour remplacer par une politique apprentie :
        class PPOSafetyPolicy(RuleBasedSafetyPolicy):
            def evaluate(self, detections, ego_speed_ms) -> Tuple[bool, str, float]:
                ...  # inference PPO
    Passez l'instance a SafetyAgent(policy=PPOSafetyPolicy(...)).
    """

    def __init__(
        self,
        ttc_emergency: float = 1.5,
        ttc_clear: float = 3.0,
    ) -> None:
        self.ttc_emergency = ttc_emergency
        self.ttc_clear = ttc_clear

    # Distance minimum de securite (centre-a-centre) : meme a l'arret, le Safety Agent
    # maintient le veto si l'obstacle est a moins de MIN_SAFE_M.
    # Evite le cycle : ego arrete -> CLEAR -> normal control -> accelere vers obstacle.
    # Calibre sur mesure CARLA : ego s'arrete a 6.60m du centre obstacle apres VETO a 9.6m.
    # Valeur = 7.5m (> 6.60m) : garde le veto quand l'ego est immobilise pres de l'obstacle.
    MIN_SAFE_M: float = 7.5

    def evaluate(
        self,
        detections: List[Dict[str, Any]],
        ego_speed_ms: float,
    ) -> Tuple[bool, str, float]:
        """Calcule le TTC minimum parmi les detections sur-voie.

        Returns
        -------
        (should_brake, reason, min_ttc)
        should_brake=True si :
          - un acteur a TTC < ttc_emergency (ego en mouvement), OU
          - un acteur est a < MIN_SAFE_M centre-a-centre (ego a l'arret ou quasi).
        min_ttc = float('inf') si aucun acteur en avant sur la voie.
        """
        if not detections:
            return False, "", float("inf")

        min_fwd = float("inf")
        closest_class = ""

        for det in detections:
            fwd = det.get("forward_dist")
            if fwd is None or fwd <= 0:
                continue
            if fwd < min_fwd:
                min_fwd = fwd
                closest_class = det.get("class_name", "obstacle")

        if min_fwd == float("inf"):
            return False, "", float("inf")

        # Cas 1 : obstacle trop proche meme a l'arret (evite cycle CLEAR -> reinfoncement)
        if min_fwd < self.MIN_SAFE_M:
            reason = (
                f"distance={min_fwd:.1f}m < {self.MIN_SAFE_M}m"
                f" ({closest_class}) -- maintien veto"
            )
            return True, reason, float("inf")

        # Cas 2 : TTC dynamique (ego en mouvement)
        if ego_speed_ms < 0.1:
            return False, "", float("inf")

        min_ttc = min_fwd / ego_speed_ms
        if min_ttc < self.ttc_emergency:
            reason = (
                f"TTC={min_ttc:.2f}s < seuil={self.ttc_emergency}s"
                f" ({closest_class} a {min_fwd:.1f}m)"
            )
            return True, reason, min_ttc

        return False, "", min_ttc


# ---------------------------------------------------------------------------
# SafetyAgent
# ---------------------------------------------------------------------------

class SafetyAgent:
    """Safety Agent -- veto transversal de freinage d'urgence.

    Appeler run() apres ControlAgent, avant apply_control().
    Verifier blackboard.safety.override pour appliquer le frein d'urgence.

    Exemple d'integration dans la boucle de demo :
        safety_agent.run(ego_speed_ms=speed_ms, tick=tick)
        ctrl = blackboard.control
        if blackboard.safety.override:
            ego.apply_control(carla.VehicleControl(
                throttle=0.0,
                steer=float(ctrl.steer),   # cap maintenu
                brake=safety_agent.brake_force,
            ))
        else:
            ego.apply_control(carla.VehicleControl(
                throttle=float(ctrl.throttle),
                steer=float(ctrl.steer),
                brake=float(ctrl.brake),
            ))

    Parameters
    ----------
    blackboard    : shared Blackboard
    config        : full config dict (lit la section 'safety' si presente)
    ttc_emergency : seuil de declenchement en secondes (default 1.5s)
    ttc_clear     : seuil de liberation du latch en secondes (default 3.0s)
    latch_ticks   : ticks minimum en veto apres declenchement (default 20)
    brake_force   : pression de frein pendant le veto (default 0.7)
    policy        : politique de decision (default: RuleBasedSafetyPolicy)
    """

    def __init__(
        self,
        blackboard: Blackboard,
        config: Optional[Dict[str, Any]] = None,
        ttc_emergency: float = 1.5,
        ttc_clear: float = 3.0,
        latch_ticks: int = 20,
        brake_force: float = 0.7,
        policy: Optional[RuleBasedSafetyPolicy] = None,
    ) -> None:
        self._bb = blackboard
        self.brake_force = brake_force
        self._latch_ticks = latch_ticks

        # Surcharge depuis config si section 'safety' presente
        if config:
            safety_cfg = config.get("safety", {})
            ttc_emergency  = float(safety_cfg.get("ttc_emergency", ttc_emergency))
            ttc_clear      = float(safety_cfg.get("ttc_clear",     ttc_clear))
            latch_ticks    = int(  safety_cfg.get("latch_ticks",   latch_ticks))
            self.brake_force = float(safety_cfg.get("brake_force", brake_force))
            self._latch_ticks = latch_ticks

        self._policy: RuleBasedSafetyPolicy = policy or RuleBasedSafetyPolicy(
            ttc_emergency=ttc_emergency,
            ttc_clear=ttc_clear,
        )

        # Etat interne
        self._currently_overriding = False
        self._latch_remaining = 0

    # ------------------------------------------------------------------
    # Main tick method
    # ------------------------------------------------------------------

    def run(self, ego_speed_ms: float, tick: int) -> None:
        """Evalue TTC, gere le latch, publie blackboard.safety."""
        detections = self._bb.perception.detections

        should_brake, reason, min_ttc = self._policy.evaluate(detections, ego_speed_ms)

        prev_overriding = self._currently_overriding

        # --- Logique latch ---
        if should_brake:
            if not self._currently_overriding:
                # Nouvelle intervention : logue dans le blackboard
                self._bb.log_safety_intervention(reason=reason, ttc=min_ttc)
                logger.warning(
                    "[SafetyAgent] VETO t=%d  TTC=%.2fs  %s",
                    tick, min_ttc, reason,
                )
                print(f"[SAFETY] VETO t={tick:>4}  {reason}")
            # Recharge le latch a chaque tick de danger
            self._latch_remaining = self._latch_ticks
            self._currently_overriding = True

        elif self._currently_overriding:
            # Obstacle plus en zone critique -- latch en decompte
            self._latch_remaining -= 1
            if self._latch_remaining <= 0 and min_ttc > self._policy.ttc_clear:
                self._currently_overriding = False
                self._latch_remaining = 0
                print(
                    f"[SAFETY] CLEAR t={tick:>4}"
                    f"  TTC={min_ttc:.2f}s"
                    f"  (veto leve, latch expire)"
                )

        # --- Publie l'etat de securite ---
        state = SafetyState(
            override=self._currently_overriding,
            ttc=round(min_ttc, 2) if min_ttc != float("inf") else float("inf"),
            reason=reason if self._currently_overriding else "",
            sensor_fault=self._bb.safety.sensor_fault,
            interventions_count=self._bb.safety.interventions_count,
            interventions_log=self._bb.safety.interventions_log,
        )
        self._bb.publish_safety(state)

        # --- Log TTC (annule si override freinage test actif, log utile seulement) ---
        if min_ttc < 5.0 or tick % 20 == 0:
            print(
                f"[SAFETY] t={tick:>4}"
                f"  TTC={min_ttc:>6.2f}s"
                f"  override={self._currently_overriding}"
                f"  latch={self._latch_remaining}"
                f"  interventions={self._bb.safety.interventions_count}"
            )

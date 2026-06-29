"""Tests unitaires PlanningAgent -- sans CARLA.

Structure
---------
- carla mocke en tete de fichier (avant tout import de planning_agent)
- MockStraightMap     : route droite vers l'est, waypoints reguliers
- MockJunctionMap     : 2 branches apres chaque wp (simule un carrefour)
- MockOppositeMap     : 1er snap correct (Est), snaps suivants en sens inverse (Ouest)
                        -> reproduit le bug road939 de Town10HD_Opt

Tests
-----
1. test_trajectory_format_and_length      -- 3-tuple (x, y, speed), longueur = n_waypoints
2. test_waypoints_ahead_of_ego            -- tous les wp devant l'ego (x > ego_x)
3. test_continuity_tight                  -- saut entre replans < 1m sur 25 ticks
4. test_junction_warning                  -- JUNCTION logue quand len(nexts) > 1
5. test_fallback_perception               -- utilise blackboard.perception sans carte CARLA
6. test_recovery_after_trajectory_exhaustion -- None case -> regen -> ControlAgent OK
7. test_angle_normalization               -- _delta_yaw et _safe_yaw sur valeurs hors [-180,180]
8. test_wrong_way_snap_recovery           -- snap voie inverse -> anchor.next() fallback
"""

import logging
import math
import sys
import types
from typing import Any, List

# ---------------------------------------------------------------------------
# Mock carla -- AVANT tout import qui ferait `import carla`
# ---------------------------------------------------------------------------
_mock_carla = types.ModuleType("carla")

class _Location:
    """Remplace carla.Location dans les tests."""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

_mock_carla.Location = _Location  # type: ignore
sys.modules["carla"] = _mock_carla

# Maintenant safe d'importer PlanningAgent (son `import carla` utilisera le mock)
from backend.src.agents.blackboard import Blackboard, PerceptionState  # noqa: E402
from backend.src.agents.planning_agent import PlanningAgent              # noqa: E402


# ---------------------------------------------------------------------------
# Mocks CARLA Map
# ---------------------------------------------------------------------------

class _MockTransform:
    def __init__(self, x: float, y: float, yaw: float = 0.0) -> None:
        self.location = _Location(x=x, y=y)
        self.rotation = type("Rotation", (), {"yaw": yaw})()


class _MockWaypoint:
    """Waypoint CARLA minimal.

    yaw=0.0  -> route vers l'est (+X), .next() avance en +X (coherent avec yaw=0).
    """

    def __init__(
        self, x: float, y: float, n_branches: int = 1, yaw: float = 0.0
    ) -> None:
        self.transform = _MockTransform(x, y, yaw)
        self.lane_width = 3.5
        self.is_junction = n_branches > 1
        self._n_branches = n_branches

    def next(self, spacing: float) -> "List[_MockWaypoint]":
        next_x = self.transform.location.x + spacing
        next_y = self.transform.location.y
        # Branches legerement decalees en Y pour simuler une intersection
        return [
            _MockWaypoint(next_x, next_y + i * 0.001, 1, yaw=0.0)
            for i in range(self._n_branches)
        ]


class _MockWaypointReverse(_MockWaypoint):
    """Waypoint pointant vers l'ouest (yaw=180 deg).

    Simule le snap incohérent observe sur road939 (Town10HD_Opt) :
    road_yaw ~ +185 deg pendant que ego_yaw ~ -11 deg -> delta_yaw ~ 163 deg.
    .next() va vers l'ouest (-X) pour etre physiquement coherent avec le cap 180 deg.
    Mais en pratique, .next() n'est JAMAIS appele sur ce wp (il est rejete par
    le filtre de coherence de cap avant la generation de trajectoire).
    """

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y, n_branches=1, yaw=180.0)

    def next(self, spacing: float) -> "List[_MockWaypoint]":
        # Va vers l'ouest (x decroissant)
        return [_MockWaypoint(
            self.transform.location.x - spacing,
            self.transform.location.y,
            1,
            yaw=180.0,
        )]


class MockStraightMap:
    """Route droite infinie vers l'est -- waypoint exact a la position demandee."""

    def get_waypoint(self, location: Any, project_to_road: bool = True) -> _MockWaypoint:
        return _MockWaypoint(location.x, location.y)


class MockJunctionMap:
    """Chaque waypoint a n_branches branches (simule un carrefour permanent)."""

    def __init__(self, n_branches: int = 2) -> None:
        self._n = n_branches

    def get_waypoint(self, location: Any, project_to_road: bool = True) -> _MockWaypoint:
        return _MockWaypoint(location.x, location.y, n_branches=self._n)


class MockOppositeMap:
    """Premier appel get_waypoint : snap correct vers l'est (yaw=0).
    Tous les appels suivants : snap vers l'ouest (yaw=180) -- sens inverse.

    Reproduit le comportement de CARLA sur Town10HD_Opt ticks 237-249 :
    road_yaw=+185 deg alors que ego_yaw ~ -11 deg (delta_yaw ~ 163 deg).
    """

    def __init__(self) -> None:
        self._calls = 0

    def get_waypoint(
        self, location: Any, project_to_road: bool = True
    ) -> _MockWaypoint:
        self._calls += 1
        if self._calls <= 1:
            return _MockWaypoint(location.x, location.y, yaw=0.0)   # correct : Est
        return _MockWaypointReverse(location.x, location.y)          # bug : Ouest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(
    n_waypoints: int = 10,
    spacing: float = 2.0,
    target_speed: float = 16.0,
) -> PlanningAgent:
    bb = Blackboard()
    cfg = {
        "agents": {
            "planning": {
                "lookahead_waypoints": n_waypoints,
                "waypoint_spacing": spacing,
                "target_speed": target_speed,
                "slow_speed": 6.0,
            }
        }
    }
    return PlanningAgent(bb, cfg)


class _CapturingHandler(logging.Handler):
    """Handler qui stocke les records pour assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# ---------------------------------------------------------------------------
# Test 1 -- Format et longueur
# ---------------------------------------------------------------------------

def test_trajectory_format_and_length() -> None:
    """La trajectoire doit contenir exactement n_waypoints 3-tuples (x, y, speed)."""
    agent = _make_agent(n_waypoints=10, spacing=2.0, target_speed=16.0)
    agent.set_carla_map(MockStraightMap())

    agent.run(0.0, 0.0)
    wps = agent._blackboard.planning.waypoints

    assert len(wps) == 10, f"Expected 10 waypoints, got {len(wps)}"

    for i, wp in enumerate(wps):
        assert len(wp) == 3, f"wp[{i}] doit etre un 3-tuple, got {len(wp)}-tuple"
        x, y, speed = wp
        assert isinstance(x, float), f"wp[{i}].x doit etre float, got {type(x)}"
        assert isinstance(y, float), f"wp[{i}].y doit etre float, got {type(y)}"
        assert speed == 16.0, f"wp[{i}].speed={speed} != target_speed=16.0"

    print(
        f"  n=10, spacing=2m : "
        f"x in [{wps[0][0]:.2f}, {wps[-1][0]:.2f}]m, speed={wps[0][2]:.1f} km/h"
    )
    print("[PASS] test_trajectory_format_and_length")


# ---------------------------------------------------------------------------
# Test 2 -- Waypoints devant l'ego
# ---------------------------------------------------------------------------

def test_waypoints_ahead_of_ego() -> None:
    """Tous les waypoints generes doivent etre DEVANT l'ego (x > ego_x sur route droite)."""
    agent = _make_agent(n_waypoints=10, spacing=2.0)
    agent.set_carla_map(MockStraightMap())

    ego_x, ego_y = 50.0, 3.0
    agent.run(ego_x, ego_y)
    wps = agent._blackboard.planning.waypoints

    assert len(wps) == 10, f"Expected 10 wps, got {len(wps)}"
    for i, (x, y, _) in enumerate(wps):
        assert x > ego_x, (
            f"wp[{i}] a x={x:.2f} est derriere l'ego a x={ego_x} "
            f"-- le planificateur doit generer des wp devant l'ego"
        )

    print(
        f"  ego_x={ego_x}, wps x in [{wps[0][0]:.1f}, {wps[-1][0]:.1f}] "
        f"(tous > {ego_x})"
    )
    print("[PASS] test_waypoints_ahead_of_ego")


# ---------------------------------------------------------------------------
# Test 3 -- Continuite resserree < 1m
# ---------------------------------------------------------------------------

def test_continuity_tight() -> None:
    """Le premier waypoint ne doit pas sauter de plus de 1m entre deux replans consecutifs.

    A 16 km/h et DT=0.05s, l'ego avance 0.222m par tick.
    Sur route droite, le premier wp se deplace de ~0.222m entre replans.
    Un saut > 1m indiquerait une discontinuite inacceptable pour le controleur.
    """
    DT = 0.05
    SPEED_MS = 16.0 / 3.6   # ~4.44 m/s
    N_TICKS = 25
    MAX_JUMP_M = 1.0         # seuil reserre : contrat etape 4

    agent = _make_agent(n_waypoints=10, spacing=2.0)
    agent.set_carla_map(MockStraightMap())

    ego_x = 0.0
    prev_first: tuple = None  # type: ignore
    max_observed = 0.0

    for tick in range(N_TICKS):
        agent.run(ego_x, 0.0)
        wps = agent._blackboard.planning.waypoints
        assert len(wps) > 0, f"Tick {tick}: trajectoire vide inattendue"

        curr_first = (wps[0][0], wps[0][1])

        if prev_first is not None:
            delta = math.hypot(
                curr_first[0] - prev_first[0],
                curr_first[1] - prev_first[1],
            )
            max_observed = max(max_observed, delta)
            assert delta < MAX_JUMP_M, (
                f"Continuite brisee au tick {tick}: "
                f"saut premier-wp = {delta:.4f}m > {MAX_JUMP_M}m "
                f"(ego avance {SPEED_MS * DT:.3f}m/tick)"
            )

        prev_first = curr_first
        ego_x += SPEED_MS * DT

    print(
        f"  {N_TICKS} ticks, ego parcourt {ego_x:.2f}m total, "
        f"saut max premier-wp = {max_observed:.4f}m (seuil < {MAX_JUMP_M}m)"
    )
    print("[PASS] test_continuity_tight")


# ---------------------------------------------------------------------------
# Test 4 -- Avertissement intersection
# ---------------------------------------------------------------------------

def test_junction_warning() -> None:
    """FORK doit etre logue quand len(nexts) > 1, trajectoire non vide.

    Historique: "JUNCTION" -> "FORK" depuis etape 5 fix pour distinguer
    is_junction (zone geometrique CARLA, diagnostic) de len(nexts)>1 (vrai fork, FSM).
    """
    agent = _make_agent(n_waypoints=5, spacing=2.0)
    agent.set_carla_map(MockJunctionMap(n_branches=2))

    handler = _CapturingHandler()
    plan_logger = logging.getLogger("backend.src.agents.planning_agent")
    original_level = plan_logger.level
    plan_logger.addHandler(handler)
    plan_logger.setLevel(logging.WARNING)

    try:
        agent.run(0.0, 0.0)
    finally:
        plan_logger.removeHandler(handler)
        plan_logger.setLevel(original_level)

    # Depuis etape 5 fix : "FORK" (len>1, signal FSM) remplace "JUNCTION" (is_junction, diag)
    fork_records = [r for r in handler.records if "FORK" in r.getMessage()]
    assert len(fork_records) > 0, (
        f"Aucun avertissement FORK -- messages recus : "
        f"{[r.getMessage() for r in handler.records]}"
    )

    # branch_in_chain=True doit etre publie (len(nexts)>1)
    assert agent._blackboard.planning.branch_in_chain is True, (
        "branch_in_chain doit etre True quand len(nexts)>1 dans la trajectoire"
    )

    wps = agent._blackboard.planning.waypoints
    assert len(wps) > 0, (
        "La trajectoire ne doit pas etre vide a une intersection "
        "(nexts[0] doit etre selectionne)"
    )

    print(f"  {len(fork_records)} FORK warning(s) loguees")
    print(f"  Premier: {fork_records[0].getMessage()[:90]}...")
    print(f"  Trajectoire generee : {len(wps)} waypoints, branch_in_chain=True")
    print("[PASS] test_junction_warning")


# ---------------------------------------------------------------------------
# Test 5 -- Fallback perception
# ---------------------------------------------------------------------------

def test_fallback_perception() -> None:
    """Sans carte CARLA, PlanningAgent utilise blackboard.perception.lane_geometry."""
    bb = Blackboard()
    cfg = {
        "agents": {
            "planning": {
                "lookahead_waypoints": 5,
                "waypoint_spacing": 2.0,
                "target_speed": 16.0,
                "slow_speed": 6.0,
            }
        }
    }
    agent = PlanningAgent(bb, cfg, carla_map=None)  # pas de carte

    # Simuler une publication PerceptionAgent avec waypoints_ahead (x, y, yaw_deg)
    fake_wps_ahead = [(float(i * 2), 0.0, 0.0) for i in range(1, 6)]
    bb.publish_perception(PerceptionState(
        lane_geometry={"waypoints_ahead": fake_wps_ahead}
    ))

    agent.run(0.0, 0.0)
    wps = bb.planning.waypoints

    assert len(wps) == 5, f"Expected 5 wps depuis perception fallback, got {len(wps)}"
    for i, (x, y, speed) in enumerate(wps):
        assert speed == 16.0, f"Speed doit etre target_speed=16.0, got {speed}"
        assert abs(x - float((i + 1) * 2)) < 1e-9, f"x[{i}] attendu {(i+1)*2}, got {x}"

    print(f"  Fallback perception : {len(wps)} wps (x, y, yaw) -> (x, y, 16.0 km/h)")
    print("[PASS] test_fallback_perception")


# ---------------------------------------------------------------------------
# Test 6 -- Recuperation apres trajectoire epuisee (None case)
# ---------------------------------------------------------------------------

def test_recovery_after_trajectory_exhaustion() -> None:
    """Trajectoire epuisee (tous wps derriere ego) -> regeneration -> target valide.

    Simule le scenario observe en CARLA ticks 150-160 :
      1. Ancienne trajectoire : wps a x=2..10 (ego encore a x=0)
      2. Ego avance a x=15 : tous les wps derriere -> ControlAgent retourne None
      3. PlanningAgent regenere depuis x=15 -> wps a x=17..25
      4. ControlAgent retrouve un target a >= 5m devant l'ego

    Ce test ECHOUE si la regeneration produit des wps derriere l'ego
    (la raison probable du bug CARLA : get_waypoint snappe voie opposee).
    """
    from backend.src.agents.blackboard import PlanningState
    from backend.src.agents.control_agent import ControlAgent

    bb = Blackboard()
    full_cfg = {
        "agents": {
            "planning": {
                "lookahead_waypoints": 5,
                "waypoint_spacing": 2.0,
                "target_speed": 16.0,
                "slow_speed": 6.0,
            },
            "control": {
                "lookahead_distance": 5.0,
                "steer_kp": 1.2, "steer_ki": 0.0, "steer_kd": 0.1,
                "max_steer": 0.8,
                "speed_kp": 0.8, "speed_ki": 0.12, "speed_kd": 0.0,
                "speed_ref_kmh": 30.0,
                "max_throttle": 1.0, "max_brake": 0.7,
            },
        },
        "carla": {"fixed_delta_seconds": 0.05},
    }

    planning_agent = PlanningAgent(bb, full_cfg)
    planning_agent.set_carla_map(MockStraightMap())
    control_agent = ControlAgent(bb, full_cfg)

    # -- Etape 1 : trajectoire initiale depuis ego=(0,0) --
    planning_agent.run(0.0, 0.0)
    traj_initial = list(bb.planning.waypoints)
    # Sur MockStraightMap, wps a x=2,4,6,8,10 (spacing=2m, n=5)
    assert len(traj_initial) == 5

    # -- Etape 2 : ego avance a x=15, tous les wps derriere --
    # Republier manuellement l'ancienne trajectoire pour simuler l'etat "non regenere"
    bb.publish_planning(PlanningState(waypoints=traj_initial))
    target_before = control_agent._select_target_waypoint(traj_initial, 15.0, 0.0, 0.0)
    assert target_before is None, (
        f"Wps a x=2..10 vs ego x=15, cap=0 -> doit retourner None, got {target_before}"
    )

    # -- Etape 3 : PlanningAgent regenere depuis ego=(15, 0) --
    planning_agent.run(15.0, 0.0)
    traj_recovered = list(bb.planning.waypoints)

    assert len(traj_recovered) == 5, f"Expected 5 wps apres recuperation, got {len(traj_recovered)}"

    # Tous les nouveaux wps doivent etre devant ego x=15
    behind = [(x, y) for x, y, _ in traj_recovered if x <= 15.0]
    assert not behind, (
        f"Wps derriere ego x=15 apres recuperation: {behind} "
        f"-- signe probable d'un snap sur voie opposee dans CARLA"
    )

    # -- Etape 4 : ControlAgent retrouve un target valide --
    target_recovered = control_agent._select_target_waypoint(traj_recovered, 15.0, 0.0, 0.0)
    assert target_recovered is not None, (
        "Apres recuperation PlanningAgent, ControlAgent doit trouver un target. "
        "Si None : la regeneration produit des wps derriere l'ego (bug heading-snap)."
    )

    jump = math.hypot(
        traj_recovered[0][0] - traj_initial[-1][0],
        traj_recovered[0][1] - traj_initial[-1][1],
    )

    print(f"  Ancienne traj : x=[{traj_initial[0][0]:.0f}..{traj_initial[-1][0]:.0f}]m")
    print(f"  Ego avance x=15 -> None confirme")
    print(f"  Nouvelle traj : x=[{traj_recovered[0][0]:.0f}..{traj_recovered[-1][0]:.0f}]m")
    print(f"  Jump geographique : {jump:.2f}m (= ego a avance de 5m depuis dernier wp)")
    print(f"  Target recupere : ({target_recovered[0]:.1f}, {target_recovered[1]:.1f})")
    print("[PASS] test_recovery_after_trajectory_exhaustion")


# ---------------------------------------------------------------------------
# Test 7 -- Normalisation d'angle dans _delta_yaw et _safe_yaw
# ---------------------------------------------------------------------------

def test_angle_normalization() -> None:
    """_delta_yaw doit retourner la PLUS PETITE difference angulaire dans [-180, 180].

    CARLA stocke parfois des road_yaw hors [-180, 180] :
      - +360.0   (= 0°, rotation accumulee)
      - -270.2   (= +89.8°)
      - -539.6   (= +180.4°, quasi-inverse)

    La formule ((a - b + 180) % 360) - 180 gere ces cas SANS pre-normalisation,
    car Python % renvoie toujours un resultat positif pour diviseur positif.

    Ce test ECHOUERAIT si la formule retournait des valeurs hors [-180, 180] :
      ex: ego=170, road=-170 -> difference reelle = 20 deg
          si formule naive : 170 - (-170) = 340 deg [FAUX]
          si formule correcte : 340 normalise = 340 - 360 = -20 deg [OK]

    Contexte t=158 du run diagnostique :
      road_yaw_raw=-539.6, delta_yaw_affiché=-39.2
      road_yaw_normalise = ((−539.6 + 180) % 360) − 180 = 0.4 − 180 = −179.6°
      _delta_yaw(141.2, −539.6) = _delta_yaw(141.2, −179.6) = −39.2° (identique)
      Conclusion : la formule etait correcte, seul l'affichage etait brut.
      FALLBK ne devait PAS se declencher a t=158 (39.2 < 45 deg).
    """
    from backend.src.agents.planning_agent import PlanningAgent

    dYaw = PlanningAgent._delta_yaw

    # --- Cas utilisateur ---

    # Cas A : ego=+170, road=-170 -> difference reelle = 20 deg (direction W-ish vs SW-ish)
    # Formule naive donnerait 340 deg [FAUX]. Correcte : -20 deg.
    d_a = dYaw(170.0, -170.0)
    assert abs(abs(d_a) - 20.0) < 0.01, (
        f"ego=170, road=-170 : |delta| attendu 20°, got {d_a:.4f}° "
        f"(formule naive donnerait 340, signe d'absence de wrapping)"
    )
    assert abs(d_a) < 45, "Ces directions sont coherentes (20° d'ecart) — pas de FALLBK"

    # Cas B : ego=+90, road=-270 (= +90, direction identique) -> delta = 0
    d_b = dYaw(90.0, -270.0)
    assert abs(d_b) < 0.01, (
        f"ego=90, road=-270 (≡ +90) : delta attendu 0°, got {d_b:.4f}°"
    )

    # --- Valeurs observees dans le run CARLA ---

    # Cas C : road=+360.0 (= 0°), ego≈0 (meme direction) -> delta ≈ 0
    d_c = dYaw(0.0, 360.0)
    assert abs(d_c) < 0.01, (
        f"ego=0, road=+360 (≡ 0) : delta attendu 0°, got {d_c:.4f}°"
    )

    # Cas D : road=-270.2 (= +89.8°), ego=-90 (= +270°, quasi-oppose) -> |delta| ≈ 179.8 > 45
    # En CARLA : road a +89.8 (NE-ish), ego a -90 (SW-ish) -> directions inverses
    d_d = dYaw(-90.0, -270.2)
    assert abs(d_d) > 45, (
        f"ego=-90, road=-270.2 (=+89.8) : |delta| doit > 45 (quasi-inverse), got {d_d:.4f}°"
    )
    assert abs(abs(d_d) - 179.8) < 0.5, (
        f"ego=-90, road=-270.2 : |delta| attendu ~179.8°, got {abs(d_d):.4f}°"
    )

    # Cas E : road=-539.6 (= +180.4°, quasi-inverse a ego=0) -> |delta| ≈ 179.6 > 45
    # Ce cas est exactement celui observe a t=158 (road_yaw_raw=-539.6)
    d_e = dYaw(0.0, -539.6)
    assert abs(d_e) > 45, (
        f"ego=0, road=-539.6 (=+180.4, quasi-inverse) : |delta| doit > 45, got {d_e:.4f}°"
    )
    assert abs(abs(d_e) - 179.6) < 0.5, (
        f"ego=0, road=-539.6 : |delta| attendu ~179.6°, got {abs(d_e):.4f}°"
    )

    # --- Verification du recalcul t=158 ---
    # road_raw=-539.6, dYaw_affiche=-39.2, ego_back-solve≈141.2°
    # Avec raw : _delta_yaw(141.2, -539.6) doit = -39.2
    # Avec normalise (-179.6) : _delta_yaw(141.2, -179.6) doit = -39.2 aussi
    d_t158_raw  = dYaw(141.2, -539.6)
    d_t158_norm = dYaw(141.2, -179.6)
    assert abs(d_t158_raw - (-39.2)) < 0.1, (
        f"Recalcul t=158 (road raw) : attendu -39.2°, got {d_t158_raw:.4f}°"
    )
    assert abs(d_t158_norm - (-39.2)) < 0.1, (
        f"Recalcul t=158 (road normalise) : attendu -39.2°, got {d_t158_norm:.4f}°"
    )
    assert abs(d_t158_raw) < 45, (
        f"FALLBK ne devait PAS se declencher a t=158 (|delta|={abs(d_t158_raw):.1f} < 45)"
    )

    # --- _safe_yaw normalise dans [-180, 180) ---
    # Verifier que _safe_yaw retourne des valeurs normalisees
    from backend.src.agents.planning_agent import PlanningAgent as _PA
    class _WpWithYaw:
        class transform:
            class rotation:
                yaw = 0.0  # will be set per test
        def __init__(self, y):
            class _R:
                pass
            r = _R(); r.yaw = y
            class _T:
                pass
            t = _T(); t.rotation = r
            self.transform = t

    def safe(raw_yaw):
        return _PA._safe_yaw(_WpWithYaw(raw_yaw))

    assert abs(safe(360.0) - 0.0) < 0.001,    f"safe_yaw(360)   attendu 0,    got {safe(360.0)}"
    assert abs(safe(-270.2) - 89.8) < 0.001,  f"safe_yaw(-270.2) attendu 89.8, got {safe(-270.2)}"
    assert abs(safe(-539.6) - (-179.6)) < 0.1, f"safe_yaw(-539.6) attendu -179.6, got {safe(-539.6)}"
    assert abs(safe(0.0) - 0.0) < 0.001,       f"safe_yaw(0.0)   attendu 0,    got {safe(0.0)}"
    assert abs(safe(180.0) - (-180.0)) < 0.001 or abs(safe(180.0) - 180.0) < 0.001, \
        f"safe_yaw(180)   attendu ±180,  got {safe(180.0)}"

    print(f"  Cas A ego=170,  road=-170  -> dYaw={d_a:+.3f}° (attendu ±20)")
    print(f"  Cas B ego=90,   road=-270  -> dYaw={d_b:+.3f}° (attendu  0)")
    print(f"  Cas C ego=0,    road=+360  -> dYaw={d_c:+.3f}° (attendu  0)")
    print(f"  Cas D ego=-90,  road=-270.2-> dYaw={d_d:+.3f}° (|d|>45, inverse)")
    print(f"  Cas E ego=0,    road=-539.6-> dYaw={d_e:+.3f}° (|d|>45, inverse)")
    print(f"  t=158 raw  : dYaw={d_t158_raw:+.3f}° | normalise : {d_t158_norm:+.3f}° (identiques -> calcul etait correct)")
    print(f"  _safe_yaw : 360->0 | -270.2->89.8 | -539.6->-179.6 | 180->-180")
    print("[PASS] test_angle_normalization")


# ---------------------------------------------------------------------------
# Test 8 -- Recuperation apres snap voie inverse (bug road939)
# ---------------------------------------------------------------------------

def test_wrong_way_snap_recovery() -> None:
    """Le PlanningAgent NE DOIT PAS generer une trajectoire a rebours apres un snap incohérent.

    Scenario reproduit (Town10HD_Opt, ticks 237-249 diagnostiques) :
      - Tick 1 : get_waypoint renvoie un wp vers l'Est (yaw=0 deg) -> anchor initialise
      - Tick 2+ : get_waypoint renvoie un wp vers l'Ouest (yaw=180 deg) -> delta_yaw=180
      - Attendu : PlanningAgent detecte |delta_yaw|=180 > 45 deg -> fallback anchor.next()
                  -> trajectoire vers l'Est generee depuis l'anchor
      - Interdit : trajectoire a rebours (x < ego_x) OU trajectoire vide (None -> brake)

    Ce test aurait ECHOUE avant le fix : le snap incohérent etait utilise tel quel,
    generant des wps vers l'Ouest -> proj <= -1 -> ControlAgent None -> brake 0.3.
    """
    agent = _make_agent(n_waypoints=5, spacing=2.0, target_speed=16.0)
    agent.set_carla_map(MockOppositeMap())

    # -- Tick 1 : snap correct (Est), anchor initialise --
    agent.run(0.0, 0.0, ego_heading_deg=0.0)
    traj1 = list(agent._blackboard.planning.waypoints)

    assert len(traj1) == 5, f"Tick 1: attendu 5 wp, got {len(traj1)}"
    assert all(x > 0 for x, y, _ in traj1), (
        f"Tick 1: tous wp doivent etre a l'Est (x > 0), got {[(x, y) for x, y, _ in traj1]}"
    )
    assert agent._last_diag["snap_coherent"] is True, (
        "Tick 1: snap doit etre marque coherent dans _last_diag"
    )

    # -- Tick 2 : snap incohérent (Ouest, yaw=180), fallback anchor attendu --
    handler = _CapturingHandler()
    plan_logger = logging.getLogger("backend.src.agents.planning_agent")
    original_level = plan_logger.level
    plan_logger.addHandler(handler)
    plan_logger.setLevel(logging.WARNING)

    try:
        agent.run(0.22, 0.0, ego_heading_deg=0.0)
    finally:
        plan_logger.removeHandler(handler)
        plan_logger.setLevel(original_level)

    traj2 = list(agent._blackboard.planning.waypoints)

    # Le warning WRONG-WAY doit etre logue
    wrong_way_records = [
        r for r in handler.records if "WRONG-WAY" in r.getMessage()
    ]
    assert len(wrong_way_records) > 0, (
        f"Attendu warning WRONG-WAY SNAP dans les logs. Messages recus : "
        f"{[r.getMessage() for r in handler.records]}"
    )

    # La trajectoire ne doit pas etre vide (pas de None -> brake)
    assert len(traj2) > 0, (
        "Apres snap incohérent, la trajectoire NE DOIT PAS etre vide. "
        "Un vide silencieux = None -> brake 0.3 indefiniment (bug original)."
    )

    # Aucun waypoint ne doit etre derriere ou au niveau de l'ego (x <= 0.22)
    backwards = [(x, y) for x, y, _ in traj2 if x <= 0.22]
    assert not backwards, (
        f"Apres snap incohérent (road=Ouest), AUCUN waypoint ne doit etre derriere "
        f"l'ego (x=0.22). Waypoints a rebours : {backwards}. "
        f"Trajectoire complete : {[(round(x, 2), round(y, 2)) for x, y, _ in traj2]}"
    )

    # snap_coherent doit etre False pour que le Safety Agent puisse le detecter
    assert agent._last_diag["snap_coherent"] is False, (
        "Tick 2: snap_coherent doit etre False dans _last_diag (signal pour Safety Agent)"
    )

    print(f"  Tick 1 : snap Est  -> traj x=[{traj1[0][0]:.0f}..{traj1[-1][0]:.0f}]m")
    print(f"  Tick 2 : snap Ouest (delta_yaw=180) -> fallback anchor.next()")
    print(f"           traj x=[{traj2[0][0]:.2f}..{traj2[-1][0]:.2f}]m (tous > ego 0.22) OK")
    print(f"  Warning logue : {wrong_way_records[0].getMessage()[:80]}...")
    print("[PASS] test_wrong_way_snap_recovery")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PlanningAgent tests (sans CARLA) ===")
    test_trajectory_format_and_length()
    test_waypoints_ahead_of_ego()
    test_continuity_tight()
    test_junction_warning()
    test_fallback_perception()
    test_recovery_after_trajectory_exhaustion()
    test_angle_normalization()
    test_wrong_way_snap_recovery()
    print()
    print("[OK] Tous les tests PlanningAgent passent.")

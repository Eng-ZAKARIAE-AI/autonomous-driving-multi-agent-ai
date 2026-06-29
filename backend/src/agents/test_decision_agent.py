"""Tests unitaires -- DecisionAgent (etape 5).

Tests couverts :
  1. Etat initial FSM = FOLLOW_LANE, decision publiee avec valeurs par defaut
  2. FOLLOW_LANE -> SLOW_DOWN sur junction_in_chain
  3. SLOW_DOWN -> FOLLOW_LANE quand junction disparait
  4. FOLLOW_LANE -> SLOW_DOWN sur obstacle proche
  5. SLOW_DOWN reste si obstacle ET junction simultanement presents
  6. GoStraightPolicy retourne ego_heading inchange
  7. branch_heading = None quand pas de junction (FOLLOW_LANE)
  8. Choix de branche deterministe dans PlanningAgent (avec branch_heading fixe,
     meme branche choisie sur ticks successifs)
  9. Fallback nexts[0] quand branch_heading = None (non-regression etape 4)

Pour lancer :
  .\\venv\\Scripts\\python.exe -m backend.src.agents.test_decision_agent
"""

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.src.agents.blackboard import (
    Blackboard,
    DecisionState,
    PerceptionState,
    PlanningState,
)
from backend.src.agents.decision_agent import DecisionAgent, GoStraightPolicy

# ---------------------------------------------------------------------------
# Mocks partages avec test_planning_agent.py
# ---------------------------------------------------------------------------

_CFG_MIN = {
    "agents": {
        "planning": {
            "target_speed": 16.0,
            "slow_speed":    6.0,
            "obstacle_distance": 15.0,
            "lookahead_waypoints": 5,
            "waypoint_spacing":    2.0,
        }
    }
}


def _make_detection(dist_m: float) -> dict:
    return {"class": "car", "distance_m": dist_m, "confidence": 0.9}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_decision(
    bb: Blackboard,
    ego_speed_ms: float = 5.0,
    ego_heading_deg: float = 90.0,
    cfg: dict = None,
) -> Blackboard:
    """Instancie et execute DecisionAgent.run() une fois."""
    agent = DecisionAgent(bb, cfg or _CFG_MIN)
    agent.run(ego_speed_ms, ego_heading_deg, (0.0, 0.0))
    return bb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_state_and_default_publish():
    """Etat initial FOLLOW_LANE, valeurs par defaut publiees."""
    bb = Blackboard()
    _run_decision(bb)
    d = bb.decision
    assert d.fsm_state == "FOLLOW_LANE", f"Got {d.fsm_state}"
    assert d.target_speed == 16.0,       f"Got {d.target_speed}"
    assert d.branch_heading is None,     f"Got {d.branch_heading}"
    assert d.is_at_junction is False,    f"Got {d.is_at_junction}"
    assert d.timestamp > 0.0,            "Timestamp non mis a jour"
    print("[PASS] test_initial_state_and_default_publish")


def test_follow_lane_to_slow_down_on_branch():
    """FOLLOW_LANE -> SLOW_DOWN quand branch_in_chain=True ET branch_distance_m < 15m.

    Signal FSM = vrai fork (len(nexts)>1), PAS is_junction seul.
    """
    bb = Blackboard()
    bb.publish_planning(PlanningState(
        waypoints=[],
        branch_in_chain=True,       # vrai fork present
        branch_distance_m=10.0,     # a 10m < 15m (seuil branch_near_m)
        junction_in_chain=True,     # junction aussi, mais ce n'est pas lui qui compte
        junction_distance_m=0.0,
    ))
    _run_decision(bb, ego_heading_deg=90.0)
    d = bb.decision
    assert d.fsm_state == "SLOW_DOWN",   f"Got {d.fsm_state}"
    assert d.target_speed == 6.0,        f"Got {d.target_speed}"
    assert d.is_at_junction is True,     f"Got {d.is_at_junction}"
    # GoStraightPolicy: branch_heading = ego_heading = 90.0 (publie car fork reel)
    assert d.branch_heading == 90.0,     f"Got {d.branch_heading}"
    print("[PASS] test_follow_lane_to_slow_down_on_branch")


def test_slow_down_returns_to_follow_lane():
    """SLOW_DOWN -> FOLLOW_LANE quand le fork disparait."""
    bb = Blackboard()
    agent = DecisionAgent(bb, _CFG_MIN)

    # Tick 1 : fork present et proche -> SLOW_DOWN
    bb.publish_planning(PlanningState(
        waypoints=[], branch_in_chain=True, branch_distance_m=10.0
    ))
    agent.run(5.0, 90.0, (0.0, 0.0))
    assert agent.fsm_state == "SLOW_DOWN"

    # Tick 2 : plus de fork dans 40m -> FOLLOW_LANE
    bb.publish_planning(PlanningState(
        waypoints=[], branch_in_chain=False, branch_distance_m=float("inf")
    ))
    agent.run(5.0, 90.0, (0.0, 0.0))
    assert agent.fsm_state == "FOLLOW_LANE",       f"Got {agent.fsm_state}"
    assert bb.decision.fsm_state == "FOLLOW_LANE", f"Got {bb.decision.fsm_state}"
    assert bb.decision.target_speed == 16.0,       f"Got {bb.decision.target_speed}"
    assert bb.decision.branch_heading is None,     f"branch_heading doit etre None sans fork"
    print("[PASS] test_slow_down_returns_to_follow_lane")


def test_follow_lane_to_slow_down_on_obstacle():
    """FOLLOW_LANE -> SLOW_DOWN sur obstacle a 8m (< seuil 15m)."""
    bb = Blackboard()
    bb.publish_perception(PerceptionState(
        detections=[_make_detection(8.0)], timestamp=1.0
    ))
    _run_decision(bb, ego_heading_deg=45.0)
    d = bb.decision
    assert d.fsm_state == "SLOW_DOWN",  f"Got {d.fsm_state}"
    assert d.target_speed == 6.0,       f"Got {d.target_speed}"
    # Pas de junction -> branch_heading = None
    assert d.branch_heading is None,    f"Got {d.branch_heading}"
    assert d.is_at_junction is False,   f"Got {d.is_at_junction}"
    print("[PASS] test_follow_lane_to_slow_down_on_obstacle")


def test_slow_down_stays_if_branch_and_obstacle():
    """SLOW_DOWN reste si fork disparu mais obstacle toujours present."""
    bb = Blackboard()
    agent = DecisionAgent(bb, _CFG_MIN)

    # Fork + obstacle simultanement -> SLOW_DOWN
    bb.publish_planning(PlanningState(waypoints=[], branch_in_chain=True, branch_distance_m=8.0))
    bb.publish_perception(PerceptionState(detections=[_make_detection(5.0)]))
    agent.run(5.0, 0.0, (0.0, 0.0))
    assert agent.fsm_state == "SLOW_DOWN"

    # Fork disparu mais obstacle reste -> SLOW_DOWN toujours
    bb.publish_planning(PlanningState(waypoints=[], branch_in_chain=False))
    agent.run(5.0, 0.0, (0.0, 0.0))
    assert agent.fsm_state == "SLOW_DOWN", (
        f"Doit rester SLOW_DOWN (obstacle present), got {agent.fsm_state}"
    )
    print("[PASS] test_slow_down_stays_if_branch_and_obstacle")


def test_no_slow_down_on_junction_without_branch():
    """FSM reste FOLLOW_LANE si junction_in_chain=True MAIS branch_in_chain=False.

    Reproduit le scenario Town10HD_Opt : jwc=9-16/20, bc=0, SLOW_DOWN permanent.
    Avec le fix, junction seule NE declenche PAS SLOW_DOWN.
    """
    bb = Blackboard()
    bb.publish_planning(PlanningState(
        waypoints=[],
        junction_in_chain=True,          # comme Town10 : is_junction partout
        junction_distance_m=0.0,         # junc=0m en permanence
        branch_in_chain=False,            # MAIS aucun vrai fork (bc=0)
        branch_distance_m=float("inf"),
    ))
    _run_decision(bb, ego_heading_deg=0.0)
    d = bb.decision
    assert d.fsm_state == "FOLLOW_LANE", (
        f"junction seule NE doit PAS declencher SLOW_DOWN, got {d.fsm_state}"
    )
    assert d.target_speed == 16.0,   f"Vitesse doit rester normale, got {d.target_speed}"
    assert d.branch_heading is None, f"branch_heading doit etre None sans fork"
    print("[PASS] test_no_slow_down_on_junction_without_branch")


def test_is_branch_only_when_len_nexts_gt1():
    """_select_next_waypoint : is_branch=True seulement si len(nexts)>1.

    Cas 1 : len==2 -> is_branch=True, is_junc_only=False
    Cas 2 : len==1 ET is_junction=True -> is_branch=False, is_junc_only=True
    Cas 3 : len==1 ET is_junction=False -> is_branch=False, is_junc_only=False
    """
    from backend.src.agents.planning_agent import PlanningAgent

    class _Loc:
        def __init__(self, x, y): self.x = x; self.y = y
    class _Rot:
        def __init__(self, yaw): self.yaw = yaw
    class _Tf:
        def __init__(self, x, y, yaw=0.0):
            self.location = _Loc(x, y)
            self.rotation = _Rot(yaw)

    class _WP:
        def __init__(self, x, y, n_next=1, is_junction=False):
            self.transform = _Tf(x, y)
            self.lane_width = 3.5
            self.is_junction = is_junction
            self.road_id = 10; self.lane_id = -1
            self._n = n_next
        def next(self, spacing):
            return [_WP(self.transform.location.x + spacing + i*0.001, 0, 1, False)
                    for i in range(self._n)]

    class _WPJuncOnly(_WP):
        """Single branch, is_junction=True -> is_junc_only=True."""
        def next(self, spacing):
            nxt = _WP(self.transform.location.x + spacing, 0, 1, True)  # is_junction=True
            return [nxt]

    bb = Blackboard()
    cfg = {"agents": {"planning": {"lookahead_waypoints": 2, "waypoint_spacing": 2.0,
                                   "target_speed": 16.0, "slow_speed": 6.0}}}

    try:
        import carla as _c  # noqa
    except ModuleNotFoundError:
        import types
        cm = types.ModuleType("carla")
        class _CL:
            def __init__(self, x=0, y=0, z=0): self.x=x; self.y=y; self.z=z
        cm.Location = _CL
        sys.modules["carla"] = cm

    class _Map1:
        def get_waypoint(self, loc, project_to_road=True): return _WP(loc.x, loc.y, n_next=2)
    class _Map2:
        def get_waypoint(self, loc, project_to_road=True): return _WPJuncOnly(loc.x, loc.y)
    class _Map3:
        def get_waypoint(self, loc, project_to_road=True): return _WP(loc.x, loc.y, n_next=1, is_junction=False)

    # Cas 1 : len==2 -> branch_in_chain=True
    a1 = PlanningAgent(bb, cfg, carla_map=_Map1())
    a1.run(0.0, 0.0, 0.0)
    assert bb.planning.branch_in_chain is True, f"Cas 1: attendu True, got {bb.planning.branch_in_chain}"
    assert bb.planning.branch_distance_m < float("inf"), "Cas 1: branch_distance_m doit etre fini"

    # Cas 2 : len==1, is_junction=True -> branch_in_chain=False (junc_only)
    a2 = PlanningAgent(bb, cfg, carla_map=_Map2())
    a2.run(0.0, 0.0, 0.0)
    assert bb.planning.branch_in_chain is False, f"Cas 2: attendu False, got {bb.planning.branch_in_chain}"
    assert bb.planning.junction_in_chain is True, f"Cas 2: junction_in_chain doit etre True"

    # Cas 3 : len==1, is_junction=False -> branch=False, junction=False
    a3 = PlanningAgent(bb, cfg, carla_map=_Map3())
    a3.run(0.0, 0.0, 0.0)
    assert bb.planning.branch_in_chain is False,   f"Cas 3: attendu False"
    assert bb.planning.junction_in_chain is False, f"Cas 3: junction attendu False"

    print("[PASS] test_is_branch_only_when_len_nexts_gt1")


def test_branch_near_threshold_not_triggered_if_fork_far():
    """FSM ne passe pas en SLOW_DOWN si branch_distance_m >= branch_near_m (15m).

    branch_heading est quand meme publie car PlanningAgent a besoin du cap pour filtrer
    nexts[] meme si la FSM reste FOLLOW_LANE (fork loin = anticipation, pas ralentissement).
    Ce qui ne doit PAS changer : l'etat FSM et la vitesse cible.
    """
    bb = Blackboard()
    bb.publish_planning(PlanningState(
        waypoints=[],
        branch_in_chain=True,
        branch_distance_m=30.0,   # fork present mais LOIN (> 15m)
    ))
    _run_decision(bb, ego_heading_deg=45.0)
    assert bb.decision.fsm_state == "FOLLOW_LANE", (
        f"Fork loin (30m) ne doit pas declencher SLOW_DOWN, got {bb.decision.fsm_state}"
    )
    assert bb.decision.target_speed == 16.0, (
        f"Vitesse doit rester normale (16), got {bb.decision.target_speed}"
    )
    # branch_heading peut etre publie (GoStraightPolicy: 45.0) car bc=True
    # PlanningAgent a besoin du cap pour filtrer nexts[] meme si FSM = FOLLOW_LANE
    print("[PASS] test_branch_near_threshold_not_triggered_if_fork_far")


def test_go_straight_policy_returns_ego_heading():
    """GoStraightPolicy retourne exactement ego_heading pour tout cap."""
    policy = GoStraightPolicy()
    dummy_ctx = DecisionState()
    for heading in [0.0, 90.0, -45.0, 135.5, -180.0, 360.0]:
        result = policy.select_branch_heading(heading, 20.0, dummy_ctx)
        assert result == heading, f"heading={heading}: got {result}"
    print("[PASS] test_go_straight_policy_returns_ego_heading")


def test_branch_heading_none_without_junction():
    """branch_heading = None quand junction_in_chain=False (etat FOLLOW_LANE)."""
    bb = Blackboard()
    bb.publish_planning(PlanningState(waypoints=[], junction_in_chain=False))
    _run_decision(bb, ego_heading_deg=30.0)
    assert bb.decision.branch_heading is None, f"Got {bb.decision.branch_heading}"
    assert bb.decision.fsm_state == "FOLLOW_LANE"
    print("[PASS] test_branch_heading_none_without_junction")


def test_planning_branch_selection_deterministic():
    """PlanningAgent choisit la meme branche sur ticks successifs avec branch_heading fixe.

    Scenario : 2 branches (yaw=0 et yaw=180). branch_heading=0 -> toujours yaw=0.
    Simule le comportement au carrefour des ticks 128/143 de Town10HD_Opt.
    """
    from backend.src.agents.planning_agent import PlanningAgent

    # ---- Mocks ----
    class _Loc:
        def __init__(self, x, y): self.x = x; self.y = y

    class _Rot:
        def __init__(self, yaw): self.yaw = yaw; self.pitch = 0.0; self.roll = 0.0

    class _Tf:
        def __init__(self, x, y, yaw):
            self.location = _Loc(x, y)
            self.rotation = _Rot(yaw)

    class _BranchWP:
        """Waypoint avec 2 branches : yaw=0 (est) et yaw=180 (ouest)."""
        def __init__(self, x, y, yaw=0.0):
            self.transform = _Tf(x, y, yaw)
            self.lane_width = 3.5
            self.is_junction = False
            self.road_id = 10
            self.lane_id = -1
        def next(self, spacing):
            # Branche A : cap est (yaw=0)
            wa = _BranchWP(self.transform.location.x + spacing, self.transform.location.y, 0.0)
            wa.is_junction = True
            # Branche B : cap ouest (yaw=180)
            wb = _BranchWP(self.transform.location.x - spacing, self.transform.location.y, 180.0)
            wb.is_junction = True
            return [wa, wb]

    class _StableWP:
        """Waypoint simple (une seule branche) apres le choix."""
        def __init__(self, x, y, yaw=0.0):
            self.transform = _Tf(x, y, yaw)
            self.lane_width = 3.5
            self.is_junction = True
            self.road_id = 10
            self.lane_id = -1
        def next(self, spacing):
            return [_StableWP(self.transform.location.x + spacing, self.transform.location.y, 0.0)]

    # _BranchWP.next() retourne 2 branches, mais les branches suivantes sont stables.
    # Pour la simplicite du test, on fait que la branche A (yaw=0) a des .next() stables.

    class _MockBranchMap:
        def get_waypoint(self, location, project_to_road=True):
            return _BranchWP(location.x, location.y, 0.0)

    class _MockLoc:
        def __init__(self, x, y): self.x = x; self.y = y

    try:
        import carla as _carla  # noqa: F401
    except ModuleNotFoundError:
        # Stub carla.Location pour PlanningAgent._generate_from_carla_map
        import types
        carla_stub = types.ModuleType("carla")
        class _CLocation:
            def __init__(self, x=0, y=0, z=0): self.x=x; self.y=y; self.z=z
        carla_stub.Location = _CLocation
        sys.modules["carla"] = carla_stub

    bb = Blackboard()
    cfg = {
        "agents": {"planning": {"lookahead_waypoints": 5, "waypoint_spacing": 2.0,
                                "target_speed": 16.0, "slow_speed": 6.0}}
    }
    # Publier branch_heading=0 (cap est = branche A)
    bb.publish_decision(DecisionState(
        fsm_state="SLOW_DOWN",
        target_speed=6.0,
        branch_heading=0.0,   # <-- FSM a decide cap=0 (est)
        is_at_junction=True,
    ))

    agent = PlanningAgent(bb, cfg, carla_map=_MockBranchMap())

    chosen_xs = []
    for tick in range(3):
        agent.run(0.0, 0.0, ego_heading_deg=0.0)
        traj = bb.planning.waypoints
        if traj:
            chosen_xs.append(traj[0][0])

    # Toutes les trajectoires doivent commencer sur la branche A (x > 0)
    for i, x in enumerate(chosen_xs):
        assert x > 0, (
            f"Tick {i}: x={x:.3f} <= 0 -> branche B (cap ouest) choisie au lieu de A. "
            f"FALLBK en rafale non resolu."
        )
    # Verifier la stabilite : meme x a chaque tick
    if len(chosen_xs) >= 2:
        for i in range(1, len(chosen_xs)):
            assert abs(chosen_xs[i] - chosen_xs[0]) < 0.1, (
                f"Branche instable : tick 0 x={chosen_xs[0]:.3f}, tick {i} x={chosen_xs[i]:.3f}"
            )
    print(f"[PASS] test_planning_branch_selection_deterministic (xs={[f'{x:.2f}' for x in chosen_xs]})")


def test_planning_fallback_nexts0_without_branch_heading():
    """Sans branch_heading, PlanningAgent utilise nexts[0] (non-regression etape 4)."""
    from backend.src.agents.planning_agent import PlanningAgent

    class _Loc:
        def __init__(self, x, y): self.x = x; self.y = y
    class _Rot:
        def __init__(self, yaw): self.yaw = yaw
    class _Tf:
        def __init__(self, x, y, yaw):
            self.location = _Loc(x, y)
            self.rotation = _Rot(yaw)
    class _TwoWP:
        def __init__(self, x, y):
            self.transform = _Tf(x, y, 0.0)
            self.lane_width = 3.5
            self.is_junction = True
            self.road_id = 5
            self.lane_id = -1
        def next(self, spacing):
            # 2 branches, nexts[0] a x positif
            a = _TwoWP(self.transform.location.x + spacing, self.transform.location.y)
            a.is_junction = False
            b = _TwoWP(self.transform.location.x, self.transform.location.y + spacing)
            b.is_junction = False
            return [a, b]

    class _MockMap2:
        def get_waypoint(self, location, project_to_road=True):
            return _TwoWP(location.x, location.y)

    try:
        import carla as _c  # noqa
    except ModuleNotFoundError:
        import types
        cm = types.ModuleType("carla")
        class _CL:
            def __init__(self, x=0, y=0, z=0): self.x=x; self.y=y; self.z=z
        cm.Location = _CL
        sys.modules["carla"] = cm

    bb = Blackboard()
    # decision.branch_heading = None (par defaut)
    cfg = {"agents": {"planning": {"lookahead_waypoints": 3, "waypoint_spacing": 2.0,
                                   "target_speed": 16.0, "slow_speed": 6.0}}}
    agent = PlanningAgent(bb, cfg, carla_map=_MockMap2())
    agent.run(0.0, 0.0, ego_heading_deg=0.0)
    traj = bb.planning.waypoints
    assert traj, "Trajectoire vide"
    # nexts[0] est la branche A (x progressif)
    assert traj[0][0] > 0, f"nexts[0] attendu x>0, got {traj[0][0]}"
    assert traj[0][1] == 0.0, f"nexts[0] attendu y=0, got {traj[0][1]}"
    print("[PASS] test_planning_fallback_nexts0_without_branch_heading")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [
        test_initial_state_and_default_publish,
        test_follow_lane_to_slow_down_on_branch,
        test_slow_down_returns_to_follow_lane,
        test_follow_lane_to_slow_down_on_obstacle,
        test_slow_down_stays_if_branch_and_obstacle,
        # Nouveaux tests etape 5 fix
        test_no_slow_down_on_junction_without_branch,
        test_is_branch_only_when_len_nexts_gt1,
        test_branch_near_threshold_not_triggered_if_fork_far,
        # Existants
        test_go_straight_policy_returns_ego_heading,
        test_branch_heading_none_without_junction,
        test_planning_branch_selection_deterministic,
        test_planning_fallback_nexts0_without_branch_heading,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            failed.append(t.__name__)
    print()
    if failed:
        print(f"ECHEC : {len(failed)}/{len(tests)} tests ont echoue : {failed}")
        sys.exit(1)
    print(f"[OK] Tous les {len(tests)} tests DecisionAgent passent.")
    print("     (dont 3 nouveaux : no-junc-only, is_branch-only-len>1, threshold-far)")


if __name__ == "__main__":
    _run_all()

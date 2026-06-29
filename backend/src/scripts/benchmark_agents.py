"""Benchmark architecture 5-agents : securite + fluidite + controle lateral.

30 episodes : 10 spawns x 3 conditions x 1 rep (CARLA deterministe, prouve).
Connexion CARLA unique reutilisee sur tous les episodes (gain ~8s/episode).
Output : benchmark_results.csv + benchmark_summary.csv + resume console.

Conditions
----------
  nominal  : aucun obstacle -- mesure flux pur + CTE sur geometrie variee
  obstacle : obstacle statique 40m -- SLOW_DOWN -> Safety-distance
  vanish   : obstacle 40m + vanish t=165 -- SLOW_DOWN seul, 0 veto attendu

Critere "evitable"
------------------
  Un veto Safety est EVITABLE si :
    (1) veto_type == "TTC"  (pas la garde distance -- celle-ci est by-design)
    (2) first_detect_dist >= 15m  (SLOW_DOWN avait le temps de s'engager)
  A haute valeur => D_slow=15m trop court ou decel PID trop faible.

CTE
---
  Ne sont pris en compte que les ticks en FOLLOW_LANE ET sans override Safety.
  Les ticks SLOW_DOWN + VETO sont exclus (suivi non actif pendant freinage).

ttc_min / dist_min
------------------
  "n/a" en condition nominale (pas d'obstacle). Exclus des agregats conditon=nominal.

Faux positif nominal
--------------------
  Tout veto en condition nominale => notes="CRITICAL_FALSE_POSITIVE" dans CSV,
  flag rouge en console. C'est la non-regression du scenario c de l'etape 7.

Usage
-----
  .\\venv37\\Scripts\\python.exe backend/src/scripts/benchmark_agents.py
  .\\venv37\\Scripts\\python.exe backend/src/scripts/benchmark_agents.py --out-dir results/
"""

import argparse
import contextlib
import csv
import io
import logging
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.src.agents.blackboard import Blackboard
from backend.src.agents.carla_actor_perception import CarlaActorPerception
from backend.src.agents.control_agent import ControlAgent
from backend.src.agents.decision_agent import DecisionAgent
from backend.src.agents.planning_agent import PlanningAgent
from backend.src.agents.safety_agent import SafetyAgent
from backend.src.scripts.demo_control_carla import (
    cross_track_error,
    ego_heading_deg,
    ego_position,
    ego_speed_ms,
    load_config,
)
from simulator.envs.carla_env import CarlaGymEnv

try:
    import carla  # type: ignore
except ModuleNotFoundError:
    carla = None  # type: ignore

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration benchmark
# ---------------------------------------------------------------------------

SPAWNS = [60, 75, 30, 50, 10, 40, 145, 15, 5, 25]
CONDITIONS = ["nominal", "obstacle", "vanish"]
TICKS: Dict[str, int] = {"nominal": 200, "obstacle": 280, "vanish": 280}
OBSTACLE_DIST_M    = 40.0   # metres devant ego (forward_dist Euclidien)
VANISH_AT_TICK     = 165    # tick de destruction (condition vanish)
V_TARGET_KMH       = 16.0   # vitesse cible reelle (agents.planning.target_speed dans config)
D_SLOW_M           = 15.0   # seuil obstacle_distance (config) -- critere evitable
INF = float("inf")


# ---------------------------------------------------------------------------
# EpisodeMetrics
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    episode_id: int
    spawn_index: int
    condition: str
    status: str = "ok"              # ok | retry | failed

    # --- Securite ---
    collision_count: int = 0        # 0 ou 1 par episode (binaire)
    ttc_min: float = INF            # INF = pas d'obstacle (nominal)
    dist_min_m: float = INF         # INF = pas d'obstacle (nominal)
    veto_type: str = "NONE"         # TTC | DISTANCE | NONE

    # --- Couche proactive ---
    safety_interventions_count: int = 0
    safety_interventions_evitables: int = 0
    first_detect_dist_m: float = INF

    # --- Fluidite ---
    v_mean_kmh: float = 0.0
    v_target_kmh: float = V_TARGET_KMH
    v_fluidity: float = 0.0         # v_mean / v_target
    pct_slow_down: float = 0.0      # % ticks en SLOW_DOWN

    # --- Controle lateral (FOLLOW_LANE + pas de veto) ---
    cte_mean_m: float = 0.0
    cte_max_m: float = 0.0
    cte_n_ticks: int = 0

    duration_ticks: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers episode
# ---------------------------------------------------------------------------

def _spawn_ego(env: Any, spawn_index: int) -> Any:
    world = env.world
    bp_lib = world.get_blueprint_library()
    ego_filter = env.config["carla"]["ego_filter"]
    bps = list(bp_lib.filter(ego_filter))
    if not bps:
        raise RuntimeError(f"Blueprint introuvable : '{ego_filter}'")
    ego_bp = random.choice(bps)

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Aucun spawn point disponible.")

    idx = spawn_index % len(spawn_points)
    ego = world.try_spawn_actor(ego_bp, spawn_points[idx])
    if ego is None:
        raise RuntimeError(f"Spawn echoue a l'index {idx}.")

    env.ego = ego
    env.actors = [ego]
    return ego


def _spawn_obstacle_forward_m(world: Any, ego: Any, forward_m: float = 40.0) -> Optional[Any]:
    """Spawn obstacle at exactly forward_m ahead in ego's Euclidean forward direction.

    Unlike spawn_obstacle_ahead() which walks along the waypoint chain, this places
    the obstacle directly at (ego_pos + fwd_vector * forward_m) in world space.
    This guarantees forward_dist ≈ forward_m regardless of road curvature,
    which is what CarlaActorPerception and the FSM/Safety actually measure.

    Trade-off: on tight curves the obstacle may be slightly off the road surface.
    For the benchmark this is intentional -- we test the ego's response to an obstacle
    at a known forward_dist, not to an obstacle placed along the exact road arc.
    """
    transform = ego.get_transform()
    fwd = transform.get_forward_vector()
    loc = transform.location

    target_loc = carla.Location(
        x=loc.x + fwd.x * forward_m,
        y=loc.y + fwd.y * forward_m,
        z=loc.z + 0.5,
    )

    bp_lib = world.get_blueprint_library()
    bp = None
    for veh_id in (
        "vehicle.tesla.model3",
        "vehicle.audi.a2",
        "vehicle.lincoln.mkz_2020",
        "vehicle.mercedes.coupe",
    ):
        try:
            bp = bp_lib.find(veh_id)
            break
        except (IndexError, RuntimeError):
            continue
    if bp is None:
        vbps = list(bp_lib.filter("vehicle.*"))
        bp = vbps[0] if vbps else None
    if bp is None:
        return None

    obstacle_tf = carla.Transform(target_loc, transform.rotation)
    actor = world.try_spawn_actor(bp, obstacle_tf)
    if actor is None:
        return None

    actor.set_autopilot(False)
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass
    return actor


def _cleanup(env: Any, world: Any, actors: List[Any]) -> None:
    """Destroy all episode actors and reset env state.

    Sensors are stopped before destroy to prevent callback-during-destroy warnings.
    The collision_sensor is included in actors list (added in run_episode).
    """
    for actor in actors:
        try:
            if hasattr(actor, "stop"):
                actor.stop()
            actor.destroy()
        except Exception:
            pass
    env.collision = False
    env.collision_sensor = None
    env.actors = []
    try:
        world.tick()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------

def run_episode(
    env: Any,
    episode_id: int,
    spawn_index: int,
    condition: str,
    cfg: dict,
) -> EpisodeMetrics:
    """Boucle CARLA complete pour un episode. Retourne les metriques brutes."""
    n_ticks = TICKS[condition]
    world = env.world
    actors_to_destroy: List[Any] = []
    m = EpisodeMetrics(episode_id=episode_id, spawn_index=spawn_index, condition=condition)

    try:
        ego = _spawn_ego(env, spawn_index)
        actors_to_destroy.append(ego)
        env._setup_collision_sensor()
        if getattr(env, "collision_sensor", None) is not None:
            actors_to_destroy.append(env.collision_sensor)
        env.collision = False

        for _ in range(5):   # init ticks
            world.tick()

        obs_actor = None
        if condition in ("obstacle", "vanish"):
            obs_actor = _spawn_obstacle_forward_m(world, ego, OBSTACLE_DIST_M)
            if obs_actor is not None:
                actors_to_destroy.append(obs_actor)
                world.tick()

        blackboard = Blackboard()
        carla_map = world.get_map()

        decision_agent  = DecisionAgent(blackboard, cfg)
        planning_agent  = PlanningAgent(blackboard, cfg, carla_map=carla_map)
        control_agent   = ControlAgent(blackboard, cfg)
        safety_agent    = SafetyAgent(blackboard, cfg)
        actor_perception = CarlaActorPerception(
            world=world, ego=ego, blackboard=blackboard,
            lane_half_width_m=1.5, max_range_m=50.0,
        )

        # Accumulateurs tick-level
        v_list:      List[float] = []
        cte_fl_list: List[float] = []   # CTE filtre FOLLOW_LANE
        slow_ticks = 0
        first_detect_dist = INF
        veto_events: List[Dict] = []
        ttc_min_ep = INF
        dist_min_ep = INF
        prev_override = False

        for tick in range(n_ticks):
            speed_ms = ego_speed_ms(ego)
            heading  = ego_heading_deg(ego)
            pos      = ego_position(ego)

            # Agents -- stdout supprime pour ne pas noyer la console benchmark
            with contextlib.redirect_stdout(io.StringIO()):
                actor_perception.run(ego_x=pos[0], ego_y=pos[1],
                                     ego_heading_deg=heading, tick=tick)
                decision_agent.run(ego_speed_ms=speed_ms,
                                   ego_heading_deg=heading, ego_position=pos)
                planning_agent.run(pos[0], pos[1], heading)
                control_agent.run(current_speed_ms=speed_ms,
                                  current_heading_deg=heading, ego_position=pos)
                safety_agent.run(ego_speed_ms=speed_ms, tick=tick)

            # Actuation (Safety prioritaire)
            ctrl = blackboard.control
            if blackboard.safety.override:
                ego.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    steer=float(ctrl.steer),
                    brake=float(safety_agent.brake_force),
                ))
            else:
                ego.apply_control(carla.VehicleControl(
                    throttle=float(ctrl.throttle),
                    steer=float(ctrl.steer),
                    brake=float(ctrl.brake),
                ))

            world.tick()

            if condition == "vanish" and obs_actor is not None and tick == VANISH_AT_TICK:
                try:
                    obs_actor.destroy()
                except Exception:
                    pass
                obs_actor = None

            # ---- Collecte metriques ----

            v_kmh = speed_ms * 3.6
            v_list.append(v_kmh)

            fsm = blackboard.decision.fsm_state
            if fsm == "SLOW_DOWN":
                slow_ticks += 1

            # CTE : FOLLOW_LANE uniquement, sans veto Safety
            if fsm == "FOLLOW_LANE" and not blackboard.safety.override:
                traj = blackboard.planning.waypoints
                if len(traj) >= 2:
                    cte = abs(cross_track_error(
                        pos[0], pos[1],
                        traj[0][0], traj[0][1],
                        traj[1][0], traj[1][1],
                    ))
                    cte_fl_list.append(cte)

            # Premier obstacle detecte + dist_min
            detections = blackboard.perception.detections
            if detections:
                fwd = min(d.get("forward_dist", INF) for d in detections)
                if fwd < INF:
                    if first_detect_dist == INF:
                        first_detect_dist = fwd   # premier tick de detection
                    dist_min_ep = min(dist_min_ep, fwd)

            # TTC minimum (depuis blackboard Safety)
            safety_ttc = blackboard.safety.ttc
            if safety_ttc < INF:
                ttc_min_ep = min(ttc_min_ep, safety_ttc)

            # Detection front montant veto (nouvel evenement)
            cur_override = blackboard.safety.override
            if cur_override and not prev_override:
                reason = blackboard.safety.reason
                if "TTC=" in reason:
                    vt = "TTC"
                elif "distance=" in reason:
                    vt = "DISTANCE"
                else:
                    vt = "UNKNOWN"
                veto_events.append({
                    "tick": tick,
                    "type": vt,
                    "first_detect_dist": first_detect_dist,
                    "reason": reason,
                })
            prev_override = cur_override

        # Collision : binaire par episode
        m.collision_count = 1 if env.collision else 0
        m.duration_ticks = n_ticks

        # Fluidite
        m.v_mean_kmh   = round(statistics.mean(v_list), 3) if v_list else 0.0
        m.v_fluidity   = round(m.v_mean_kmh / V_TARGET_KMH, 4)
        m.pct_slow_down = round(100.0 * slow_ticks / n_ticks, 2)

        # Controle lateral
        m.cte_mean_m  = round(statistics.mean(cte_fl_list), 4) if cte_fl_list else 0.0
        m.cte_max_m   = round(max(cte_fl_list), 4) if cte_fl_list else 0.0
        m.cte_n_ticks = len(cte_fl_list)

        # Obstacle / Safety
        m.first_detect_dist_m = round(first_detect_dist, 2) if first_detect_dist < INF else INF
        m.dist_min_m = round(dist_min_ep, 2) if dist_min_ep < INF else INF
        m.ttc_min    = round(ttc_min_ep, 3) if ttc_min_ep < INF else INF
        m.safety_interventions_count = len(veto_events)
        m.veto_type  = veto_events[0]["type"] if veto_events else "NONE"

        m.safety_interventions_evitables = sum(
            1 for v in veto_events
            if v["type"] == "TTC" and v["first_detect_dist"] >= D_SLOW_M
        )

        # Non-regression : veto en scene nominale = bug grave
        if condition == "nominal" and m.safety_interventions_count > 0:
            m.notes = "CRITICAL_FALSE_POSITIVE"

    except Exception as exc:
        m.status = "failed"
        m.notes  = str(exc)[:200]
        logger.error("Ep %d spawn=%d %s FAILED: %s", episode_id, spawn_index, condition, exc)
    finally:
        _cleanup(env, world, actors_to_destroy)

    return m


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "episode_id", "spawn_index", "condition", "status",
    "collision_count", "ttc_min", "dist_min_m", "veto_type",
    "safety_interventions_count", "safety_interventions_evitables",
    "first_detect_dist_m",
    "v_mean_kmh", "v_target_kmh", "v_fluidity", "pct_slow_down",
    "cte_mean_m", "cte_max_m", "cte_n_ticks",
    "duration_ticks", "notes",
]

SUMMARY_METRICS = [
    ("collision_count",                 False),   # (field, exclude_nominal_inf)
    ("ttc_min",                         True),    # exclure nominal (INF)
    ("dist_min_m",                      True),
    ("safety_interventions_count",      False),
    ("safety_interventions_evitables",  False),
    ("v_mean_kmh",                      False),
    ("v_fluidity",                      False),
    ("pct_slow_down",                   False),
    ("cte_mean_m",                      False),
    ("cte_max_m",                       False),
]


def _inf_str(v: float) -> str:
    return "n/a" if v == INF else str(round(v, 4))


def write_results_csv(results: List[EpisodeMetrics], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for m in results:
            row = {k: getattr(m, k) for k in CSV_FIELDS}
            # INF -> "n/a" pour lisibilite
            for k in ("ttc_min", "dist_min_m", "first_detect_dist_m"):
                if row[k] == INF:
                    row[k] = "n/a"
            w.writerow(row)


def write_summary_csv(results: List[EpisodeMetrics], path: Path) -> None:
    rows = []
    for cond in CONDITIONS:
        ep = [m for m in results if m.condition == cond and m.status != "failed"]
        if not ep:
            continue
        for field, exclude_inf in SUMMARY_METRICS:
            vals = [getattr(m, field) for m in ep]
            if exclude_inf:
                vals = [v for v in vals if v < INF]
            if not vals:
                rows.append({"condition": cond, "metric": field,
                             "mean": "n/a", "std": "n/a", "min": "n/a", "max": "n/a",
                             "n_episodes": 0})
                continue
            rows.append({
                "condition": cond,
                "metric": field,
                "mean": round(statistics.mean(vals), 4),
                "std":  round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "min":  round(min(vals), 4),
                "max":  round(max(vals), 4),
                "n_episodes": len(vals),
            })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "metric", "mean", "std", "min", "max", "n_episodes"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(results: List[EpisodeMetrics]) -> None:
    print()
    print("=" * 72)
    print("  BENCHMARK SUMMARY")
    print("=" * 72)

    for cond in CONDITIONS:
        ep = [m for m in results if m.condition == cond]
        ok = [m for m in ep if m.status != "failed"]
        n  = len(ok)
        print(f"\nCONDITION: {cond}  ({n}/{len(ep)} episodes ok)")

        if not ok:
            print("  AUCUN episode valide.")
            continue

        def _agg(field, exclude_inf=False):
            vals = [getattr(m, field) for m in ok]
            if exclude_inf:
                vals = [v for v in vals if v < INF]
            if not vals:
                return "n/a"
            mn = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{mn:.3f} +/- {sd:.3f}  [min={min(vals):.3f}  max={max(vals):.3f}]"

        coll = sum(m.collision_count for m in ok)
        veto = sum(m.safety_interventions_count for m in ok)
        evit = sum(m.safety_interventions_evitables for m in ok)
        fp   = [m for m in ok if "CRITICAL_FALSE_POSITIVE" in m.notes]

        print(f"  Safety   : collisions={coll}/{n}  veto={veto}/{n}  evitables={evit}")
        if cond != "nominal":
            print(f"             ttc_min = {_agg('ttc_min', exclude_inf=True)}")
            print(f"             dist_min = {_agg('dist_min_m', exclude_inf=True)}")
        print(f"  Fluidite : v_mean = {_agg('v_mean_kmh')}  fluidity = {_agg('v_fluidity')}")
        print(f"             pct_SLOW_DOWN = {_agg('pct_slow_down')} %")
        print(f"  Controle : cte_mean = {_agg('cte_mean_m')}  cte_max = {_agg('cte_max_m')}")
        print(f"             cte_n_ticks = {_agg('cte_n_ticks')} (ticks FOLLOW_LANE)")

        if fp:
            print(f"\n  *** [CRITICAL] {len(fp)} faux positif(s) en nominal ***")
            for m in fp:
                print(f"      spawn={m.spawn_index}  veto={m.safety_interventions_count}  notes={m.notes}")

    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------

def run_benchmark(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    env = CarlaGymEnv(cfg)
    if not env.connect():
        print("[ERROR] Connexion CARLA impossible. Lance CarlaUE4.exe d'abord.")
        return

    total = len(SPAWNS) * len(CONDITIONS)
    results: List[EpisodeMetrics] = []
    ep_id = 0

    print(f"BENCHMARK AGENTS -- {total} episodes ({len(SPAWNS)} spawns x {len(CONDITIONS)} conditions)")
    print(f"Spawns : {SPAWNS}")
    print(f"Out    : {out_dir}")
    print()

    try:
        for spawn_idx in SPAWNS:
            for cond in CONDITIONS:
                ep_id += 1
                ticks = TICKS[cond]
                print(f"[{ep_id:>2}/{total}] spawn={spawn_idx:>3}  cond={cond:<8}  ticks={ticks}  ...",
                      end="", flush=True)

                m = run_episode(env, ep_id, spawn_idx, cond, cfg)

                if m.status == "failed":
                    print(f"  FAILED -- retry ...")
                    m2 = run_episode(env, ep_id, spawn_idx, cond, cfg)
                    if m2.status != "failed":
                        m2.status = "retry"
                        m = m2
                    else:
                        m.notes = f"2x failed: {m.notes}"

                # Indicateurs inline
                coll_flag  = " [COLLISION!]" if m.collision_count > 0 else ""
                fp_flag    = " [FALSE-POS!]" if "CRITICAL_FALSE_POSITIVE" in m.notes else ""
                evit_flag  = f" [EVITABLE={m.safety_interventions_evitables}]" if m.safety_interventions_evitables > 0 else ""
                veto_flag  = f" veto={m.safety_interventions_count}({m.veto_type})" if m.safety_interventions_count > 0 else " veto=0"
                cte_str    = f"cte_max={m.cte_max_m:.3f}m" if m.cte_n_ticks > 0 else "cte=n/a"
                v_str      = f"v_mean={m.v_mean_kmh:.1f} km/h  fluidity={m.v_fluidity:.2f}"
                print(f"  {m.status}  {v_str}  {cte_str}{veto_flag}{coll_flag}{fp_flag}{evit_flag}")

                results.append(m)

    finally:
        env.close()

    # Ecriture CSV
    results_path = out_dir / "benchmark_results.csv"
    summary_path = out_dir / "benchmark_summary.csv"
    write_results_csv(results, results_path)
    write_summary_csv(results, summary_path)
    print(f"\nCSV ecrit : {results_path}")
    print(f"CSV ecrit : {summary_path}")

    print_summary(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark architecture 5-agents : securite + fluidite + CTE"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("benchmark_out"),
        help="Repertoire de sortie pour les CSV (defaut: benchmark_out/)",
    )
    args = parser.parse_args()
    run_benchmark(out_dir=args.out_dir)


if __name__ == "__main__":
    main()

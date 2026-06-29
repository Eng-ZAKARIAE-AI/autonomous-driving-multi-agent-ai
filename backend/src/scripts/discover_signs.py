"""ETAPE A — Découverte ground-truth panneaux/feux sur Town10HD_Opt.

Lance un run de N ticks sur un spawn donné et dump TROIS sources :
  1. waypoint.get_landmarks(horizon_m)  → codes OpenDRIVE (274=speed limit, 206=stop…)
  2. ego.get_speed_limit()              → limite courante fournie par CARLA directement
  3. world.get_actors("traffic.traffic_light*") → feux sur la map + état ego

Usage :
  .\\venv37\\Scripts\\python.exe backend/src/scripts/discover_signs.py
  .\\venv37\\Scripts\\python.exe backend/src/scripts/discover_signs.py --spawn-index 25 --ticks 300
  .\\venv37\\Scripts\\python.exe backend/src/scripts/discover_signs.py --scan-all   # scan 20 spawns, 50 ticks chacun
"""

import argparse
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.src.scripts.demo_control_carla import (
    ego_heading_deg, ego_position, ego_speed_ms, load_config,
)
from simulator.envs.carla_env import CarlaGymEnv

try:
    import carla  # type: ignore
except ModuleNotFoundError:
    carla = None  # type: ignore

# Codes OpenDRIVE → noms lisibles
SIGN_TYPES = {
    "206": "STOP",
    "205": "YIELD",
    "274": "SPEED_LIMIT",
    "275": "DERESTRICTION",
    "269": "PEDESTRIAN_CROSSING",
    "263": "CROSSING",
    "267": "NO_OVERTAKING",
    "306": "PRIORITY_ROAD",
    "301": "GIVE_WAY",
    "330": "NO_ENTRY",
}

LANDMARK_HORIZON_M = 60.0   # distance de recherche en avant (m)
LOG_EVERY_N = 10            # log tick-par-tick toutes les N ticks (sinon trop verbeux)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sign_name(lm_type: str) -> str:
    return SIGN_TYPES.get(str(lm_type), f"type={lm_type}")


def _dump_landmarks(wp, tick: int, log_every: int = 1) -> list:
    """Appelle get_landmarks() sur le waypoint courant, dump et retourne la liste."""
    try:
        landmarks = wp.get_landmarks(LANDMARK_HORIZON_M, stop_at_junction=False)
    except Exception as exc:
        if tick % log_every == 0:
            print(f"  [SIGN]  get_landmarks() -> ERREUR : {exc}")
        return []

    if tick % log_every == 0:
        if landmarks:
            for lm in landmarks:
                name  = _sign_name(lm.type)
                value = f"  val={lm.value:.1f}" if hasattr(lm, "value") and lm.value else ""
                print(f"  [SIGN]  t={tick:>4}  landmark: {name:<20} type={lm.type:<6}"
                      f"  id={lm.id:<8}  dist={lm.distance:.1f}m{value}")
        else:
            print(f"  [SIGN]  t={tick:>4}  (aucun landmark dans {LANDMARK_HORIZON_M}m)")
    return landmarks


def _dump_speed_limit(ego, tick: int, log_every: int = 1) -> float:
    """get_speed_limit() retourne km/h ou 0 si inconnu."""
    try:
        sl = ego.get_speed_limit()
    except Exception:
        sl = -1.0
    if tick % log_every == 0:
        print(f"  [SPEED_LIMIT]  t={tick:>4}  ego.get_speed_limit() = {sl:.1f} km/h")
    return sl


def _dump_traffic_lights_global(world) -> None:
    """Dump tous les feux présents sur la map (une seule fois au départ)."""
    tls = list(world.get_actors().filter("traffic.traffic_light*"))
    print(f"\n[TL-GLOBAL]  Feux sur la map : {len(tls)}")
    for tl in tls[:15]:   # max 15 pour ne pas noyer
        loc = tl.get_location()
        print(f"  id={tl.id:<6}  state={tl.state}  "
              f"pos=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
    if len(tls) > 15:
        print(f"  ... et {len(tls)-15} autres")
    print()


def _dump_ego_traffic_light(ego, tick: int, log_every: int = 1) -> None:
    """Vérifie si l'ego est influencé par un feu courant."""
    try:
        if ego.is_at_traffic_light():
            tl = ego.get_traffic_light()
            state = tl.get_state() if tl else "?"
            print(f"  [TL-EGO]  t={tick:>4}  A UN FEU  state={state}")
        else:
            if tick % log_every == 0:
                print(f"  [TL-EGO]  t={tick:>4}  (pas de feu actif)")
    except Exception as exc:
        if tick % log_every == 0:
            print(f"  [TL-EGO]  t={tick:>4}  erreur: {exc}")


# ---------------------------------------------------------------------------
# Run single spawn
# ---------------------------------------------------------------------------

def run_discovery(env, spawn_index: int, n_ticks: int, log_every: int = LOG_EVERY_N):
    world = env.world
    carla_map = world.get_map()

    # Spawn ego
    bp_lib = world.get_blueprint_library()
    ego_filter = env.config["carla"]["ego_filter"]
    bps = list(bp_lib.filter(ego_filter))
    if not bps:
        print(f"[ERR] blueprint '{ego_filter}' introuvable")
        return

    import random
    ego_bp = random.choice(bps)
    spawn_points = carla_map.get_spawn_points()
    idx = spawn_index % len(spawn_points)
    ego = world.try_spawn_actor(ego_bp, spawn_points[idx])
    if ego is None:
        print(f"[ERR] spawn echoue a l'index {idx}")
        return

    env.ego = ego
    env.actors = [ego]
    print(f"\n{'='*70}")
    print(f"DISCOVERY  spawn_index={spawn_index}  ticks={n_ticks}  "
          f"landmark_horizon={LANDMARK_HORIZON_M}m  log_every={log_every}")
    print(f"{'='*70}")

    # Feux globaux (une fois)
    _dump_traffic_lights_global(world)

    # Autopilot CARLA : fait rouler l'ego pour qu'il traverse la route
    # et expose les changements de speed_limit / landmarks
    try:
        ego.set_autopilot(True)
        tm = env.client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.vehicle_percentage_speed_difference(ego, -20)   # roule a ~20% au-dessus limite
    except Exception as exc:
        print(f"[WARN] autopilot non disponible : {exc}  (ego roulera sans controle)")

    for _ in range(5):
        world.tick()

    # Collecte pour résumé
    speed_limits_seen = set()
    landmark_types_seen = {}   # type -> set(valeurs)
    tl_states_seen = []

    try:
        for tick in range(n_ticks):
            world.tick()
            pos     = ego_position(ego)
            heading = ego_heading_deg(ego)
            speed   = ego_speed_ms(ego) * 3.6

            # Waypoint courant
            wp = carla_map.get_waypoint(
                ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
            )

            if tick % log_every == 0:
                print(f"\n--- tick={tick}  v={speed:.1f}km/h  "
                      f"pos=({pos[0]:.1f},{pos[1]:.1f})  heading={heading:.1f}° ---")

            # Source 1 : landmarks
            landmarks = _dump_landmarks(wp, tick, log_every)
            for lm in landmarks:
                t = str(lm.type)
                if t not in landmark_types_seen:
                    landmark_types_seen[t] = set()
                if hasattr(lm, "value") and lm.value:
                    landmark_types_seen[t].add(round(lm.value, 1))

            # Source 2 : get_speed_limit
            sl = _dump_speed_limit(ego, tick, log_every)
            if sl > 0:
                speed_limits_seen.add(round(sl, 1))

            # Source 3 : feux ego
            _dump_ego_traffic_light(ego, tick, log_every)

    finally:
        try:
            ego.destroy()
        except Exception:
            pass
        env.ego   = None
        env.actors = []
        try:
            world.tick()
        except Exception:
            pass

    # Résumé
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ spawn={spawn_index}")
    print(f"{'='*70}")
    print(f"  get_speed_limit() valeurs vues : {sorted(speed_limits_seen) or 'AUCUNE'}")
    print(f"  Landmarks détectés :")
    if landmark_types_seen:
        for t, vals in sorted(landmark_types_seen.items()):
            name = _sign_name(t)
            print(f"    type={t} ({name:<20})  valeurs={sorted(vals) if vals else 'n/a'}")
    else:
        print("    (aucun)")
    print()


# ---------------------------------------------------------------------------
# Scan multi-spawns
# ---------------------------------------------------------------------------

def run_scan_all(env, n_spawns: int = 20, n_ticks: int = 60):
    """Lance une découverte courte sur les N premiers spawns, montre un résumé compact."""
    world   = env.world
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    n_spawns = min(n_spawns, len(spawn_points))

    print(f"\nSCAN {n_spawns} spawns x {n_ticks} ticks")
    print(f"{'spawn':<8} {'speed_limits':<30} {'landmarks'}")
    print("-"*70)

    import random
    bp_lib = world.get_blueprint_library()
    ego_filter = env.config["carla"]["ego_filter"]
    bps = list(bp_lib.filter(ego_filter))

    for idx in range(n_spawns):
        ego_bp = random.choice(bps)
        ego = world.try_spawn_actor(ego_bp, spawn_points[idx])
        if ego is None:
            print(f"{idx:<8} spawn_failed")
            continue

        speed_limits = set()
        lm_types = {}

        try:
            ego.set_autopilot(True)
            tm = env.client.get_trafficmanager()
            tm.set_synchronous_mode(True)
        except Exception:
            pass

        for _ in range(5):
            world.tick()

        for _ in range(n_ticks):
            world.tick()
            sl = ego.get_speed_limit()
            if sl > 0:
                speed_limits.add(round(sl, 1))

            wp = carla_map.get_waypoint(
                ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
            )
            try:
                lms = wp.get_landmarks(40.0, stop_at_junction=False)
                for lm in lms:
                    t = str(lm.type)
                    if t not in lm_types:
                        lm_types[t] = set()
                    if hasattr(lm, "value") and lm.value:
                        lm_types[t].add(round(lm.value, 1))
            except Exception:
                pass

        try:
            ego.destroy()
        except Exception:
            pass
        try:
            world.tick()
        except Exception:
            pass

        lm_str = "  ".join(
            f"{_sign_name(t)}={sorted(v)}" for t, v in sorted(lm_types.items())
        ) or "(aucun)"
        sl_str = str(sorted(speed_limits)) if speed_limits else "(aucun)"
        print(f"{idx:<8} {sl_str:<30} {lm_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Découverte panneaux/feux CARLA")
    parser.add_argument("--spawn-index", type=int, default=60)
    parser.add_argument("--ticks",       type=int, default=200)
    parser.add_argument("--log-every",   type=int, default=20,
                        help="Affiche les sources toutes les N ticks (défaut 20)")
    parser.add_argument("--scan-all",    action="store_true",
                        help="Scan les 20 premiers spawns (résumé compact)")
    parser.add_argument("--n-spawns",    type=int, default=20,
                        help="Nombre de spawns à scanner avec --scan-all")
    args = parser.parse_args()

    cfg = load_config()
    env = CarlaGymEnv(cfg)
    if not env.connect():
        print("[ERR] CARLA non disponible — lance CarlaUE4.exe d'abord.")
        return

    try:
        if args.scan_all:
            run_scan_all(env, n_spawns=args.n_spawns, n_ticks=args.ticks)
        else:
            run_discovery(env, args.spawn_index, args.ticks, args.log_every)
    finally:
        env.close()


if __name__ == "__main__":
    main()

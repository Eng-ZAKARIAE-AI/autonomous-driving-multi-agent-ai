"""Etapes 4+5 -- Demo ControlAgent + PlanningAgent + DecisionAgent dans CARLA.

Le PlanningAgent remplace la trajectoire factice de l'etape 3.
Il regenere une trajectoire de suivi de voie (waypoints carte CARLA) a chaque tick.

Pipeline
--------
1. Connexion CARLA, spawn ego (random OU deterministique via --spawn-index / --spawn-road)
2. Creation PlanningAgent (carte CARLA wiree) + ControlAgent (PID)
3. Boucle de ticks :
     a. PlanningAgent.run(ego_x, ego_y, heading) -- genere N waypoints lane-following
     b. ControlAgent.run(speed, heading, pos) -- PID lat+long
     c. apply_control() + world.tick()
     d. Log vitesse, CTE (vs trajectoire COURANTE), integral

Options diagnostiques
---------------------
--list-spawns             : liste tous les spawn points avec road_id/lane_id, puis quitte
--spawn-index N           : spawn deterministique a l'index N de get_spawn_points()
--spawn-road R            : detecte un waypoint sur road R et teleporte l'ego dessus
                            Exemple: --spawn-road 900 pour forcer la zone road 900->939
--diag-all                : log de diagnostic a CHAQUE tick (pas seulement 120-230)
--inject-wrongway-at N    : au tick N, injecte UNE FOIS un snap de cap inverse (+180°)
                            dans la vraie boucle CARLA. Prouve que le fallback se declenche
                            et recupere sans frein, independamment de la geometrie de la carte.
                            Recommande : --inject-wrongway-at 100 --diag-all --ticks 150

Les cas FALLBK et NONE sont TOUJOURS loggues, quelle que soit la zone de diagnostic.

Injection wrong-way (approche B)
---------------------------------
Principe : _WrongWayInjectionMap est un proxy autour de carla.Map. Au tick exact N,
get_waypoint() retourne un _FlippedWaypoint (cap reel + 180°) au lieu du vrai waypoint.
L'injection est one-shot (un seul tick). Tout le reste (control, blackboard, anchor
.next()) est reel CARLA. Cela prouve :
  - a t=N   : WRONG-WAY SNAP detecte, FALLBK declenche, anchor.next() sur la vraie route
  - a t=N+1 : retour snap=OK automatique (injection epuisee)
  - pendant : pas de brake, CTE stable, road anchor coherent

Metriques de diagnostic
-----------------------
jump_f2f : saut du PREMIER wp entre deux replans consecutifs (~0.22m/tick, reel)
jump_l2f : saut du DERNIER wp precedent -> PREMIER wp courant (~38m droite, ARTEFACT
           = horizon 40m - avance ego 0.22m/tick. Pas une discontinuite.)
CTE : vs 2 premiers wp trajectoire courante. Attendu < 1m voie, < 2m virage.

Usage
-----
1. Lancer CARLA :
   CarlaUE4.exe -windowed -ResX=1280 -ResY=720 -quality-level=Low

2. Test A -- un seul try sur road 900 :
   .\\venv37\\Scripts\\python.exe -m backend.src.scripts.demo_control_carla \\
       --spawn-road 900 --ticks 200 --diag-all

3. Test B -- injection deterministe au tick 100 (garanti) :
   .\\venv37\\Scripts\\python.exe -m backend.src.scripts.demo_control_carla \\
       --inject-wrongway-at 100 --ticks 150 --diag-all

4. Lister les spawns :
   .\\venv37\\Scripts\\python.exe -m backend.src.scripts.demo_control_carla --list-spawns
"""

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from backend.src.agents.blackboard import Blackboard
from backend.src.agents.carla_actor_perception import CarlaActorPerception
from backend.src.agents.carla_traffic_light_perception import CarlaTrafficLightPerception
from backend.src.agents.carla_stop_sign_perception import CarlaStopSignPerception
from backend.src.agents.control_agent import ControlAgent
from backend.src.agents.decision_agent import DecisionAgent
from backend.src.agents.planning_agent import PlanningAgent
from backend.src.agents.safety_agent import SafetyAgent
from backend.src.dashboard.ws_server import DashboardServer
from simulator.envs.carla_env import CarlaGymEnv

try:
    import carla  # type: ignore
except ModuleNotFoundError:
    carla = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proxy d'injection wrong-way (Test B)
# ---------------------------------------------------------------------------

class _FlippedRotation:
    """Rotation CARLA avec yaw retourne de 180 deg."""

    def __init__(self, yaw: float) -> None:
        self.yaw = yaw
        self.pitch = 0.0
        self.roll = 0.0


class _FlippedTransform:
    """Transform CARLA avec rotation a cap inverse."""

    def __init__(self, real_transform: Any, flipped_yaw: float) -> None:
        self.location = real_transform.location
        self.rotation = _FlippedRotation(flipped_yaw)


class _FlippedWaypoint:
    """Proxy d'un waypoint CARLA avec rotation.yaw + 180 deg.

    next() deleguee au VRAI waypoint : la chaine reste sur la vraie route.
    road_id / lane_id identiques au vrai wp (seul le cap est inverse).
    C'est intentionnel : on simule le snap geometrique sur la lane opposee
    de la MEME route (comme road939 qui longe road900 en sens inverse).
    """

    def __init__(self, real_wp: Any) -> None:
        self._real = real_wp
        # Copie les attributs CARLA utiles
        for attr in ("road_id", "lane_id", "lane_width", "is_junction"):
            if hasattr(real_wp, attr):
                setattr(self, attr, getattr(real_wp, attr))
        # Cap retourne de 180 deg
        real_yaw = real_wp.transform.rotation.yaw
        self.transform = _FlippedTransform(real_wp.transform, real_yaw + 180.0)

    def next(self, spacing: float) -> Any:
        """Suit la VRAIE route (pas la route inverse).

        Sans ca, anchor.next() apres FALLBK partirait sur la vraie route de toute
        facon (l'anchor est le waypoint du tick precedent, pas le FlippedWaypoint),
        mais cette implementation correcte evite tout bug si next() est appele par
        erreur sur le waypoint injected.
        """
        return self._real.next(spacing)


class _WrongWayInjectionMap:
    """Proxy autour de carla.Map : retourne un _FlippedWaypoint au tick inject_at.

    Principe : one-shot. Une seule injection, puis retour au comportement normal.
    Le reste de l'API (generate_waypoints, get_spawn_points) passe au vrai Map.

    Utilisation :
        proxy = _WrongWayInjectionMap(world.get_map(), inject_at=100)
        planning_agent.set_carla_map(proxy)
        # Dans la boucle, avant planning_agent.run() :
        proxy.set_tick(tick)
    """

    def __init__(self, real_map: Any, inject_at: int) -> None:
        self._real = real_map
        self._inject_at = inject_at
        self._current_tick = -1
        self._injected = False

    def set_tick(self, tick: int) -> None:
        self._current_tick = tick

    def get_waypoint(self, location: Any, project_to_road: bool = True) -> Any:
        if self._current_tick == self._inject_at and not self._injected:
            self._injected = True
            real_wp = self._real.get_waypoint(location, project_to_road=project_to_road)
            real_yaw = real_wp.transform.rotation.yaw
            injected_yaw = real_yaw + 180.0
            print(
                f"\n  [INJECT t={self._current_tick}] Snap artificiel cap+180° : "
                f"real_yaw={real_yaw:.1f}°  injected_yaw={injected_yaw:.1f}°  "
                f"road={getattr(real_wp, 'road_id', '?')} lane={getattr(real_wp, 'lane_id', '?')}"
                f"\n  -> PlanningAgent doit detecter |delta_yaw|~180 deg > 45 deg et activer FALLBK"
            )
            return _FlippedWaypoint(real_wp)
        return self._real.get_waypoint(location, project_to_road=project_to_road)

    # Delegation des autres methodes CARLA Map (generate_waypoints, get_spawn_points...)
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Helpers vehicule
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg_path = _ROOT / "backend" / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ego_speed_ms(ego: Any) -> float:
    v = ego.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def ego_heading_deg(ego: Any) -> float:
    return ego.get_transform().rotation.yaw


def ego_position(ego: Any) -> Tuple[float, float]:
    loc = ego.get_location()
    return (loc.x, loc.y)


def cross_track_error(
    ego_x: float, ego_y: float,
    wp_x: float, wp_y: float,
    next_wp_x: float, next_wp_y: float,
) -> float:
    """Distance perpendiculaire de l'ego au segment [wp -> next_wp].
    Positif = a droite de la voie, negatif = a gauche.
    """
    dx = next_wp_x - wp_x
    dy = next_wp_y - wp_y
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-3:
        return math.hypot(ego_x - wp_x, ego_y - wp_y)
    ex = ego_x - wp_x
    ey = ego_y - wp_y
    return (ex * dy - ey * dx) / seg_len


# ---------------------------------------------------------------------------
# Spawn deterministique
# ---------------------------------------------------------------------------

def list_spawns(env: CarlaGymEnv) -> None:
    """Affiche tous les spawn points avec road_id/lane_id et quitte."""
    carla_map = env.world.get_map()
    spawn_points = carla_map.get_spawn_points()
    print(f"\n=== {len(spawn_points)} spawn points disponibles ===")
    print(f"  {'#':>4}  {'x':>9}  {'y':>9}  {'yaw':>7}  {'road_id':>8}  {'lane_id':>8}")
    print(f"  {'':->4}  {'':->9}  {'':->9}  {'':->7}  {'':->8}  {'':->8}")
    for i, sp in enumerate(spawn_points):
        wp = carla_map.get_waypoint(sp.location, project_to_road=True)
        print(
            f"  {i:>4}  {sp.location.x:>9.1f}  {sp.location.y:>9.1f}  "
            f"{sp.rotation.yaw:>7.1f}  {wp.road_id:>8}  {wp.lane_id:>8}"
        )
    print("=== Fin de la liste ===\n")


def scan_forks(env: CarlaGymEnv, horizon: int = 20, spacing: float = 2.0) -> None:
    """Scan tous les spawn points et identifie ceux dont la trajectoire contient un vrai fork.

    Un fork = len(nexts) > 1 dans les `horizon` premiers waypoints depuis le spawn.
    Affiche la distance au premier fork et le spawn index pour filtrer rapidement.

    Usage :
      .\\venv37\\Scripts\\python.exe -m backend.src.scripts.demo_control_carla --scan-forks
    """
    carla_map = env.world.get_map()
    spawn_points = carla_map.get_spawn_points()
    print(f"\n=== SCAN FORKS : {len(spawn_points)} spawns x horizon={horizon} wps ===")
    print(f"  {'#':>4}  {'road_id':>8}  {'fork?':>6}  {'dist_m':>8}  {'n_candidates':>13}")
    print(f"  {'':->4}  {'':->8}  {'':->6}  {'':->8}  {'':->13}")

    fork_spawns = []
    for i, sp in enumerate(spawn_points):
        wp = carla_map.get_waypoint(sp.location, project_to_road=True)
        if wp is None:
            continue
        fork_dist = float("inf")
        n_candidates = 0
        for step in range(horizon):
            nexts = wp.next(spacing)
            if not nexts:
                break
            if len(nexts) > 1:
                fork_dist = (step + 1) * spacing
                n_candidates = len(nexts)
                break
            wp = nexts[0]
        has_fork = fork_dist < float("inf")
        if has_fork:
            fork_spawns.append((i, fork_dist, n_candidates))
        print(
            f"  {i:>4}  {carla_map.get_waypoint(sp.location, project_to_road=True).road_id:>8}  "
            f"{'YES' if has_fork else 'no':>6}  "
            f"{fork_dist if fork_dist < 9999 else float('inf'):>8.1f}  "
            f"{n_candidates if has_fork else 0:>13}"
        )

    print(f"\n=== SPAWNS AVEC FORK ({len(fork_spawns)} trouves) ===")
    if not fork_spawns:
        print("  AUCUN fork dans l'horizon 40m -- essaie --horizon N avec N > 20")
    else:
        for idx, dist, nc in sorted(fork_spawns, key=lambda t: t[1]):
            print(f"  spawn={idx:>3}  fork a {dist:.1f}m  ({nc} branches)")
        print(f"\n  Recommande : --spawn-index {fork_spawns[0][0]} --ticks 400 --diag-all")
    print("=== Fin du scan ===\n")


def _find_target_on_road(carla_map: Any, road_id: int, margin_wps: int = 15) -> Optional[Any]:
    """Trouve un transform sur road_id, `margin_wps` avant la fin/jonction.

    Retourne le carla.Transform ou None si road introuvable.
    """
    all_wps = [
        wp for wp in carla_map.generate_waypoints(2.0)
        if wp.road_id == road_id and wp.lane_id < 0
    ]
    if not all_wps:
        all_wps = [wp for wp in carla_map.generate_waypoints(2.0) if wp.road_id == road_id]
    if not all_wps:
        return None

    # Construire la chaine depuis le premier waypoint trouve
    chain = [all_wps[0]]
    wp = all_wps[0]
    for _ in range(300):
        nexts = wp.next(2.0)
        if not nexts or len(nexts) > 1:
            break
        chain.append(nexts[0])
        wp = nexts[0]

    target_idx = max(0, len(chain) - margin_wps)
    target_wp = chain[target_idx]
    tf = target_wp.transform
    print(
        f"[INFO] road {road_id}: chaine de {len(chain)} wps, "
        f"cible wp #{target_idx} "
        f"pos=({tf.location.x:.1f}, {tf.location.y:.1f}) "
        f"yaw={tf.rotation.yaw:.1f}°  lane={target_wp.lane_id}"
    )
    return tf


def demo_spawn_ego(
    env: CarlaGymEnv,
    spawn_index: Optional[int] = None,
    spawn_road: Optional[int] = None,
) -> Any:
    """Spawn l'ego : random / index fixe / teleport sur road R."""
    blueprint_library = env.world.get_blueprint_library()
    bps = list(blueprint_library.filter(env.config["carla"]["ego_filter"]))
    if not bps:
        raise RuntimeError(f"No blueprint for filter '{env.config['carla']['ego_filter']}'")
    ego_bp = random.choice(bps)

    carla_map = env.world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available.")

    if spawn_index is not None:
        idx = spawn_index % len(spawn_points)
        sp = spawn_points[idx]
        wp = carla_map.get_waypoint(sp.location, project_to_road=True)
        print(
            f"[INFO] Spawn fixe #{idx}: "
            f"pos=({sp.location.x:.1f}, {sp.location.y:.1f})  "
            f"yaw={sp.rotation.yaw:.1f}°  road={wp.road_id}  lane={wp.lane_id}"
        )
        ego = env.world.try_spawn_actor(ego_bp, sp)
    else:
        random.shuffle(spawn_points)
        ego = None
        for sp in spawn_points:
            ego = env.world.try_spawn_actor(ego_bp, sp)
            if ego is not None:
                break

    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    if spawn_road is not None:
        target_tf = _find_target_on_road(carla_map, spawn_road, margin_wps=15)
        if target_tf is not None:
            target_tf.location.z += 0.3
            ego.set_transform(target_tf)
            env.world.tick()
            env.world.tick()
            final_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
            print(
                f"[INFO] Teleport OK -> road={final_wp.road_id}  lane={final_wp.lane_id}  "
                f"pos=({ego.get_location().x:.1f}, {ego.get_location().y:.1f})  "
                f"yaw={ego.get_transform().rotation.yaw:.1f}°"
            )
        else:
            logger.error(
                "Road %d introuvable sur la carte -- spawn aleatoire conserve.", spawn_road
            )
            print(
                "[WARN] --spawn-road invalide. "
                "Essaie --list-spawns pour voir les road_id disponibles."
            )

    env.ego = ego
    env.actors = [ego]
    return ego


# ---------------------------------------------------------------------------
# Obstacle de test pour Safety Agent (--obstacle-at N)
# ---------------------------------------------------------------------------

def spawn_obstacle_ahead(world: Any, ego: Any, distance_m: float = 40.0) -> Any:
    """Spawn un vehicule arrete sur la meme voie que l'ego, distance_m metres devant.

    Utilise un VEHICULE (pas un prop statique) : seuls les vehicules remontent dans
    world.get_actors().filter("vehicle.*"), ce qu'attend CarlaActorPerception.
    set_simulate_physics(False) garde le vehicule immobile.

    Suit la chaine de waypoints CARLA pour rester sur la voie.
    Retourne l'acteur spawne ou None en cas d'echec.
    """
    carla_map = world.get_map()
    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)

    dist_walked = 0.0
    step = 2.0
    while dist_walked + step <= distance_m:
        nexts = wp.next(step)
        if not nexts:
            break
        wp = nexts[0]
        dist_walked += step

    bp_lib = world.get_blueprint_library()

    # Priorite : vehicule (visible dans vehicle.* filter)
    bp = None
    for veh_id in (
        "vehicle.tesla.model3",
        "vehicle.audi.a2",
        "vehicle.lincoln.mkz_2020",
        "vehicle.mercedes.coupe",
        "vehicle.dodge.charger_2020",
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
        print("[OBSTACLE] Aucun blueprint vehicule -- test sans obstacle.")
        return None

    spawn_tf = wp.transform
    spawn_tf.location.z += 0.5   # evite le clipping avec la route

    print(
        f"[OBSTACLE] Cible waypoint : ({spawn_tf.location.x:.1f}, "
        f"{spawn_tf.location.y:.1f})  dist_marche={dist_walked:.1f}m"
    )

    obstacle = world.try_spawn_actor(bp, spawn_tf)
    if obstacle is None:
        print(
            f"[OBSTACLE] Spawn echoue a ({spawn_tf.location.x:.1f}, "
            f"{spawn_tf.location.y:.1f}) -- essaie un autre spawn-index."
        )
        return None

    obstacle.set_autopilot(False)
    try:
        obstacle.set_simulate_physics(False)
    except Exception:
        pass

    world.tick()   # serveur enregistre l'acteur avant la lecture de position
    loc = obstacle.get_location()
    print(
        f"[OBSTACLE] Spawn OK : {bp.id}"
        f"  pos=({loc.x:.1f}, {loc.y:.1f})"
        f"  cible={distance_m:.0f}m devant ego"
    )
    return obstacle


# ---------------------------------------------------------------------------
# Piéton traversant (--pedestrian-cross-at N) — Scénario 2
# ---------------------------------------------------------------------------

def spawn_pedestrian_crossing_ahead(
    world: Any,
    ego: Any,
    distance_m: float = 10.0,
    side_offset_m: float = 3.5,
) -> Optional[tuple]:
    """Spawn un piéton à distance_m metres devant l'ego, côté droit de la voie.

    Le piéton traversera perpendiculairement (droite → gauche) quand on lui
    appliquera un WalkerControl à chaque tick.

    Retourne (walker_actor, walk_direction: carla.Vector3D) ou None si échec.

    Limite euclidienne : fonctionne uniquement sur une portion DROITE de la route.
    Sur une courbe, le lateral_offset euclidien peut rater le piéton (même limite
    que CarlaActorPerception).
    """
    carla_map = world.get_map()
    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                lane_type=carla.LaneType.Driving)

    # -- Avancer le long des waypoints --
    dist_walked = 0.0
    step = 1.0
    while dist_walked < distance_m:
        nexts = wp.next(step)
        if not nexts:
            break
        wp = nexts[0]
        dist_walked += step

    crossing = wp.transform.location

    # -- Direction perpendiculaire à l'ego (vecteur "droite" CARLA left-handed) --
    ego_yaw = ego.get_transform().rotation.yaw
    yaw_rad = math.radians(ego_yaw)
    right_x = -math.sin(yaw_rad)   # +X quand yaw=270° (heading -Y)
    right_y =  math.cos(yaw_rad)

    # Spawn côté droit de la route
    spawn_loc = carla.Location(
        x=crossing.x + side_offset_m * right_x,
        y=crossing.y + side_offset_m * right_y,
        z=crossing.z + 0.5,          # légèrement au-dessus du sol
    )

    walker_bps = world.get_blueprint_library().filter("walker.pedestrian.*")
    if not walker_bps:
        print("[PEDESTRIAN] Aucun blueprint walker disponible.")
        return None
    import random as _random
    walker_bp = _random.choice(list(walker_bps))

    walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc))
    if walker is None:
        spawn_loc.z += 1.0
        walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc))
    if walker is None:
        print(f"[PEDESTRIAN] Spawn échoué à ({spawn_loc.x:.1f}, {spawn_loc.y:.1f}). "
              "Essaie un autre spawn-index ou distance.")
        return None

    # Direction : gauche = -droite (de droite à gauche de la route)
    walk_dir = carla.Vector3D(-right_x, -right_y, 0.0)

    world.tick()
    loc = walker.get_location()
    print(
        f"[PEDESTRIAN] Spawn OK : {walker.type_id}"
        f"  pos=({loc.x:.1f}, {loc.y:.1f})"
        f"  cible={distance_m:.0f}m devant ego"
        f"  dir=({walk_dir.x:.2f},{walk_dir.y:.2f})"
    )
    print(
        f"[PEDESTRIAN] Traversée à 0.5 m/s depuis t=--pedestrian-start-tick"
        f"  durée totale ~{2*side_offset_m/0.5:.0f}s, dans voie ~{(2*side_offset_m-2*1.5)/0.5:.0f}s"
    )
    return walker, walk_dir


# ---------------------------------------------------------------------------
# Trafic lent devant l'ego (--traffic-ahead) — Scénario 3 car-following
# ---------------------------------------------------------------------------

def spawn_traffic_ahead(
    world: Any,
    ego: Any,
    client: Any,
    distance_m: float = 20.0,
    speed_kmh: float = 10.0,
) -> Optional[tuple]:
    """Spawn un véhicule lead à distance_m devant l'ego sur la même voie.

    Pour speed_kmh > 0 : autopilot avec TrafficManager (suit la route proprement).
    Pour speed_kmh == 0 : set_simulate_physics(False) = statique (non-régression Scén1).

    Retourne (actor, use_autopilot: bool) ou None si échec.
    """
    carla_map = world.get_map()
    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)

    # Avancer le long des waypoints (step=1m)
    dist_walked = 0.0
    while dist_walked < distance_m:
        nexts = wp.next(1.0)
        if not nexts:
            break
        wp = nexts[0]
        dist_walked += 1.0

    bps = world.get_blueprint_library().filter("vehicle.*")
    car_bps = [b for b in bps if b.get_attribute("number_of_wheels").as_int() == 4]
    if not car_bps:
        car_bps = list(bps)

    bp = random.choice(car_bps)
    if bp.has_attribute("color"):
        bp.set_attribute("color", "255,128,0")  # orange visible

    spawn_tf = wp.transform
    actor = world.try_spawn_actor(bp, spawn_tf)
    if actor is None:
        spawn_tf.location.z += 0.5
        actor = world.try_spawn_actor(bp, carla.Transform(spawn_tf.location, spawn_tf.rotation))
    if actor is None:
        print(f"[TRAFFIC] Spawn échoué à {distance_m:.0f}m devant ego")
        return None

    if speed_kmh <= 0:
        # Véhicule statique : désactive la physique (non-régression Scénario 1)
        actor.set_autopilot(False)
        actor.set_simulate_physics(False)
        use_autopilot = False
        print(
            f"[TRAFFIC] Spawn OK (STATIQUE) : {actor.type_id}"
            f"  pos=({spawn_tf.location.x:.1f},{spawn_tf.location.y:.1f})"
            f"  cible={distance_m:.0f}m devant ego"
        )
    else:
        # Véhicule en mouvement : autopilot + TM (suit la route proprement)
        # TM configuré APRÈS spawn pour ne pas perturber try_spawn_actor
        actor.set_autopilot(False)
        actor.set_simulate_physics(True)
        world.tick()  # laisser CARLA enregistrer l'acteur avant TM setup
        try:
            tm = client.get_trafficmanager()
            # set_synchronous_mode(True) cause un crash natif CARLA (0xC0000409) après
            # ~285 ticks — omis volontairement. TM async fonctionne sur route droite
            # (road 900) sans fork, seul contexte validé pour car-following.
            tm.set_global_distance_to_leading_vehicle(5.0)
            actor.set_autopilot(True)
            tm.set_desired_speed(actor, speed_kmh)
            tm.ignore_lights_percentage(actor, 100)   # ne s'arrête pas aux feux
            use_autopilot = True
            print(
                f"[TRAFFIC] Spawn OK (AUTOPILOT) : {actor.type_id}"
                f"  pos=({spawn_tf.location.x:.1f},{spawn_tf.location.y:.1f})"
                f"  cible={distance_m:.0f}m  vitesse_cible={speed_kmh:.0f} km/h"
            )
        except Exception as exc:
            print(f"[TRAFFIC] WARN autopilot échoué ({exc}) — mode statique de repli")
            actor.set_autopilot(False)
            actor.set_simulate_physics(False)
            use_autopilot = False

    world.tick()
    return actor, use_autopilot


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def run_demo(
    n_ticks: int = 600,
    target_kmh: float = 20.0,
    n_waypoints_override: Optional[int] = None,
    spacing_override: Optional[float] = None,
    spawn_index: Optional[int] = None,
    spawn_road: Optional[int] = None,
    do_list_spawns: bool = False,
    do_scan_forks: bool = False,
    diag_all: bool = False,
    inject_wrongway_at: Optional[int] = None,
    obstacle_at: Optional[float] = None,
    obstacle_vanish_at: Optional[int] = None,
    brake_test_at: Optional[int] = None,
    force_red_until: Optional[int] = None,
    tl_red_time: float = 5.0,
    tl_green_time: float = 15.0,
    dashboard_port: Optional[int] = None,
    clear_npcs: bool = False,
    spectator_mode: Optional[str] = None,
    spawn_offset_back: float = 0.0,
    branch_policy: str = "straight",
    route_to: Optional[int] = None,
    use_global_planner: bool = False,
    pedestrian_cross_at: Optional[float] = None,
    pedestrian_start_tick: int = 0,
    traffic_ahead: bool = False,
    traffic_speed_kmh: float = 10.0,
    traffic_distance_m: float = 20.0,
    traffic_brake_at_tick: int = -1,
) -> None:
    cfg = load_config()

    plan_cfg = cfg.setdefault("agents", {}).setdefault("planning", {})
    plan_cfg["target_speed"] = target_kmh
    if n_waypoints_override is not None:
        plan_cfg["lookahead_waypoints"] = n_waypoints_override
    if spacing_override is not None:
        plan_cfg["waypoint_spacing"] = spacing_override

    n_wps = plan_cfg.get("lookahead_waypoints", 20)
    spacing = plan_cfg.get("waypoint_spacing", 2.0)

    env = CarlaGymEnv(cfg)
    if not env.connect():
        logger.error("Impossible de se connecter a CARLA. Lance CarlaUE4.exe d'abord.")
        return

    _dashboard: Optional[DashboardServer] = None   # init avant try pour que finally y accède
    try:
        if do_list_spawns:
            list_spawns(env)
            return

        if do_scan_forks:
            scan_forks(env, horizon=int(plan_cfg.get("lookahead_waypoints", 20)),
                       spacing=float(plan_cfg.get("waypoint_spacing", 2.0)))
            return

        # -- Nettoyage NPCs orphelins (--clear-npcs) --
        # Detruit les vehicules tiers qui trainent d'une session CARLA precedente.
        if clear_npcs:
            world_pre = env.world
            npc_actors = [a for a in world_pre.get_actors().filter("vehicle.*")]
            if npc_actors:
                print(f"[NPC] --clear-npcs : destruction de {len(npc_actors)} vehicule(s) tiers ...")
                for a in npc_actors:
                    try:
                        a.destroy()
                    except Exception:
                        pass
                world_pre.tick()
                print(f"[NPC] fait.")

        ego = demo_spawn_ego(env, spawn_index=spawn_index, spawn_road=spawn_road)
        env._setup_collision_sensor()
        world = env.world

        start_wp = world.get_map().get_waypoint(ego.get_location(), project_to_road=True)
        print(f"[INFO] Ego spawne sur road={start_wp.road_id} lane={start_wp.lane_id}")

        # -- Recul optionnel avant démarrage (--spawn-offset-back N) --
        # Téléporte l'ego N mètres en arrière sur la même route pour allonger
        # l'approche visible avant un feu rouge ou un panneau STOP.
        if spawn_offset_back > 0.0:
            _step = 2.0
            _wp = world.get_map().get_waypoint(
                ego.get_location(), project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            _backed = 0.0
            while _backed < spawn_offset_back:
                _prevs = _wp.previous(_step)
                if not _prevs:
                    break
                _wp = _prevs[0]
                _backed += _step
            ego.set_transform(_wp.transform)
            world.tick()
            print(f"[OFFSET] Ego recule de {_backed:.0f}m -> "
                  f"road={_wp.road_id} lane={_wp.lane_id}")

        # -- Degel de securite : annule tout freeze residuel d'un run precedent --
        # world.freeze_all_traffic_lights(True) persiste entre sessions CARLA.
        # Sans ce degel, les feux restes frozen-Green (fin de --force-red-until)
        # ne cycleront jamais dans le run courant.
        try:
            world.freeze_all_traffic_lights(False)
            print("[TL] Degel de securite : freeze_all_traffic_lights(False)")
        except Exception as _exc:
            print(f"[TL] WARN degel securite : {_exc}")

        # -- Calibration des durées de phase (set_*_time ne fige pas le cycle) --
        # Ordre OBLIGATOIRE : degel d'abord, durées ensuite (un freeze posterieur
        # ecraserait les durées mais pas l'inverse).
        # Rouge court (5s par defaut) pour la démo : l'ego n'attend pas 30s au feu.
        _tl_green  = max(1.0, float(tl_green_time))
        _tl_yellow =  2.0
        _tl_red    = max(1.0, float(tl_red_time))   # garde au moins 1s
        try:
            for _tl in world.get_actors().filter("traffic.traffic_light*"):
                _tl.set_green_time(_tl_green)
                _tl.set_yellow_time(_tl_yellow)
                _tl.set_red_time(_tl_red)
            print(
                f"[TL] Durées réglées : rouge={_tl_red:.0f}s"
                f"  vert={_tl_green:.0f}s  jaune={_tl_yellow:.0f}s"
            )
        except Exception as _exc:
            print(f"[TL] WARN set_*_time : {_exc}")

        # -- Forçage feux tricolores (--force-red-until T) --
        # Gele tous les feux en Rouge de t=0 jusqu'au tick T.
        # Permet un test RED_LIGHT deterministe sans attendre le cycle CARLA.
        if force_red_until is not None:
            try:
                world.freeze_all_traffic_lights(True)
                for tl in world.get_actors().filter("traffic.traffic_light*"):
                    tl.set_state(carla.TrafficLightState.Red)
                print(
                    f"[TL-FORCE] Tous les feux geles en ROUGE jusqu'au tick {force_red_until}."
                    f" Liberation au tick {force_red_until + 1}."
                )
            except Exception as exc:
                print(f"[TL-FORCE] WARN : impossible de geler les feux ({exc})")

        for _ in range(5):
            world.tick()

        # -- Obstacle de test Safety Agent (optionnel) --
        if obstacle_at is not None:
            obs_actor = spawn_obstacle_ahead(world, ego, distance_m=float(obstacle_at))
            if obs_actor is not None:
                env.actors.append(obs_actor)
                world.tick()  # laisse la physique stabiliser le prop
        else:
            obs_actor = None

        # -- Piéton traversant (--pedestrian-cross-at N, scénario 2) --
        ped_actor: Optional[Any] = None
        ped_walk_dir: Optional[Any] = None
        if pedestrian_cross_at is not None:
            result = spawn_pedestrian_crossing_ahead(
                world, ego,
                distance_m=float(pedestrian_cross_at),
                side_offset_m=3.5,
            )
            if result is not None:
                ped_actor, ped_walk_dir = result
                env.actors.append(ped_actor)

        # -- Trafic lent (--traffic-ahead, scénario 3) --
        lead_actor: Optional[Any] = None
        lead_use_autopilot: bool = False
        if traffic_ahead:
            result = spawn_traffic_ahead(
                world, ego, env.client,
                distance_m=float(traffic_distance_m),
                speed_kmh=float(traffic_speed_kmh),
            )
            if result is not None:
                lead_actor, lead_use_autopilot = result
                env.actors.append(lead_actor)

        blackboard = Blackboard()

        # -- CarlaActorPerception (ground-truth distances, Safety Agent eyes) --
        # lane_half_width_m élargi à 8m quand traffic_ahead : la lead en autopilot
        # suit la route sur les courbes et peut avoir lat > 5m en projection Euclidienne
        # (virage serré à spawn28 ≈ 60°). 8m couvre 2 voies → acceptable pour démo
        # à 1 seul véhicule tiers. Ne pas utiliser pour détection multi-voies.
        _lane_hw = 8.0 if traffic_ahead else 1.5
        actor_perception = CarlaActorPerception(
            world=world,
            ego=ego,
            blackboard=blackboard,
            lane_half_width_m=_lane_hw,
            max_range_m=50.0,
        )

        # -- CarlaTrafficLightPerception (feux tricolores ground-truth) --
        tl_perception = CarlaTrafficLightPerception(
            world=world,
            ego=ego,
            blackboard=blackboard,
            horizon_m=80.0,
            lateral_max_m=20.0,
        )

        # -- CarlaStopSignPerception (panneaux STOP ground-truth, landmark type '206') --
        stop_sign_perception = CarlaStopSignPerception(
            world=world,
            ego=ego,
            blackboard=blackboard,
            horizon_m=60.0,
        )

        # -- Choisir la carte a passer au PlanningAgent --
        real_map = world.get_map()
        if inject_wrongway_at is not None:
            carla_map = _WrongWayInjectionMap(real_map, inject_at=inject_wrongway_at)
            print(
                f"\n  [INJECT] Mode injection active : snap wrong-way au tick {inject_wrongway_at}"
                f"\n  -> Surveille les ticks {inject_wrongway_at-2}..{inject_wrongway_at+5} "
                f"pour FALLBK + recuperation\n"
            )
        else:
            carla_map = real_map

        decision_agent = DecisionAgent(blackboard, cfg)
        planning_agent = PlanningAgent(blackboard, cfg, carla_map=carla_map,
                                       branch_policy=branch_policy)
        control_agent = ControlAgent(blackboard, cfg)
        safety_agent = SafetyAgent(blackboard, cfg)

        # -- GlobalPlannerAgent (--global-planner --route-to N) --
        # Calcule la route A->B UNE SEULE FOIS avant la boucle.
        # Si reussi : active RouteFollowingBranchPolicy sur planning_agent.
        # Si echoue : log + fallback branch_policy courant (non-regression garantie).
        if use_global_planner and route_to is not None:
            from backend.src.agents.global_planner import GlobalPlannerAgent
            spawn_pts = real_map.get_spawn_points()
            if route_to < len(spawn_pts):
                dest_loc = spawn_pts[route_to].location
                gp_agent = GlobalPlannerAgent(world, blackboard, cfg)
                route_ok = gp_agent.compute_route(ego.get_location(), dest_loc)
                if route_ok:
                    planning_agent.set_branch_policy("route_following")
                    print(
                        f"[ROUTE] spawn {spawn_index} -> spawn {route_to}  "
                        f"({len(blackboard.route.route)} wps)  "
                        f"branch_policy=route_following"
                    )
                else:
                    print(f"[ROUTE] WARN: route non calculee -> fallback '{branch_policy}'")
            else:
                print(f"[ROUTE] WARN: --route-to {route_to} hors plage "
                      f"({len(spawn_pts)} spawns) -> ignore")

        ki = control_agent._longitudinal._pid.ki
        integral_cap = control_agent.longitudinal_integral_cap

        spawn_mode = (
            f"inject-wrongway@t={inject_wrongway_at}" if inject_wrongway_at is not None
            else (f"spawn-road {spawn_road}" if spawn_road is not None
                  else (f"spawn-index {spawn_index}" if spawn_index is not None
                        else "spawn aleatoire"))
        )
        print()
        print("=" * 108)
        print(
            f"  DEMO ControlAgent + PlanningAgent -- {spawn_mode}  "
            f"({n_wps} wp, {spacing}m, {target_kmh} km/h)"
        )
        print(
            f"  ki={ki}  speed_ref={control_agent._longitudinal._speed_ref:.2f} m/s  "
            f"integral_cap={integral_cap:.1f}  "
            f"{'diag TOUS LES TICKS' if diag_all else 'diag zone 120-230 + FALLBK/NONE'}"
        )
        print("=" * 108)
        print(
            f"  {'Tick':>5}  {'v_real':>8}  {'v_cible':>8}  {'err_v':>7}  "
            f"{'steer':>7}  {'thr':>5}  {'brk':>5}  {'CTE':>8}  "
            f"{'integral':>10}  {'I_term':>8}"
        )
        print(
            f"  {'':->5}  {'':->8}  {'':->8}  {'':->7}  {'':->7}  "
            f"{'':->5}  {'':->5}  {'':->8}  {'':->10}  {'':->8}"
        )

        spectator = world.get_spectator()
        collision_count = 0
        prev_traj: list = []
        fallbk_count = 0
        none_count = 0

        # -- Etat freinage d'urgence (--brake-test-at) --
        braking_active = False
        brake_pos0: tuple = (0.0, 0.0)
        brake_pos_prev: tuple = (0.0, 0.0)
        brake_dist_total = 0.0
        brake_v0 = 0.0
        brake_tick0 = -1
        brake_stopped = False

        # -- Dashboard WebSocket (--dashboard, opt-in) --
        if dashboard_port is not None:
            _dashboard = DashboardServer(port=dashboard_port)
            if not _dashboard.start():
                _dashboard = None    # websockets absent → désactivé silencieusement
        _fps_t0 = time.perf_counter()   # pour mesure FPS avec/sans dashboard
        _fps_curr = 0.0                 # FPS mesuré sur les 50 derniers ticks

        # Envoie la route globale au dashboard UNE SEULE FOIS, avant la boucle.
        # Doit être APRÈS DashboardServer.start() (ci-dessus) pour que _route_msg
        # soit visible du thread asyncio quand le premier client se connecte.
        if _dashboard is not None and blackboard.route.active:
            _rpts = [
                [round(loc.x, 1), round(loc.y, 1)]
                for loc, _ in blackboard.route.route
            ]
            _dashboard.set_route(_rpts)

        for tick in range(n_ticks):
            speed_ms = ego_speed_ms(ego)
            heading = ego_heading_deg(ego)
            pos = ego_position(ego)

            # -- Mise a jour du tick dans le proxy (si injection active) --
            if inject_wrongway_at is not None:
                carla_map.set_tick(tick)

            # -- 0. CarlaActorPerception -- publie detections avant Decision --
            actor_perception.run(
                ego_x=pos[0],
                ego_y=pos[1],
                ego_heading_deg=heading,
                tick=tick,
            )

            # -- 0b. CarlaTrafficLightPerception -- ajoute TL aux champs perception --
            tl_perception.run(
                ego_x=pos[0],
                ego_y=pos[1],
                ego_heading_deg=heading,
                tick=tick,
            )

            # -- 0b2. CarlaStopSignPerception -- ajoute stop_sign_dist_m --
            stop_sign_perception.run(
                ego_x=pos[0],
                ego_y=pos[1],
                ego_heading_deg=heading,
                tick=tick,
            )

            # -- 0b3. WalkerControl piéton (scénario 2) --
            # WalkerControl n'est pas persistant : doit être appliqué à chaque tick.
            if ped_actor is not None and ped_walk_dir is not None and tick >= pedestrian_start_tick:
                try:
                    ped_ctrl = carla.WalkerControl()
                    ped_ctrl.direction = ped_walk_dir
                    ped_ctrl.speed = 0.5  # m/s — traversée lente pour spawn28 à 6 km/h
                    ped_actor.apply_control(ped_ctrl)
                except Exception:
                    pass  # piéton peut avoir été détruit

            # -- 0b3b. VehicleControl lead (scénario 3) --
            if lead_actor is not None:
                # Cas c : freinage forcé à partir du tick spécifié
                if traffic_brake_at_tick >= 0 and tick >= traffic_brake_at_tick:
                    if lead_use_autopilot and tick == traffic_brake_at_tick:
                        try:
                            lead_actor.set_autopilot(False)
                            print(f"[LEAD-BRAKE] t={tick}  autopilot desactive -> freinage force")
                        except Exception:
                            pass
                        lead_use_autopilot = False
                    if not lead_use_autopilot:
                        try:
                            lc = carla.VehicleControl()
                            lc.throttle = 0.0
                            lc.steer    = 0.0
                            lc.brake    = 1.0
                            lead_actor.apply_control(lc)
                        except Exception:
                            pass
                elif not lead_use_autopilot:
                    # Véhicule statique (speed=0) : frein actif chaque tick
                    try:
                        lc = carla.VehicleControl()
                        lc.throttle = 0.0
                        lc.steer    = 0.0
                        lc.brake    = 1.0
                        lead_actor.apply_control(lc)
                    except Exception:
                        pass

            # -- 0b4. Log [LEAD] car-following (scénario 3) — chaque tick si lead détecté --
            if traffic_ahead:
                _dets = blackboard.perception.detections if blackboard.perception else []
                if _dets:
                    _c = _dets[0]
                    _d_obs   = _c.get("forward_dist", float("inf"))
                    _v_lead  = _c.get("v_lead_kmh", 0.0)
                    _v_ego   = speed_ms * 3.6
                    _d_secu  = max(10.0, 2.0 * speed_ms)     # miroir du calcul FSM
                    _err_d   = _d_obs - _d_secu
                    _err_v   = _v_ego - _v_lead
                    _v_tgt   = _v_lead + 0.5 * _err_d - 0.5 * _err_v
                    _v_tgt   = max(0.0, min(_v_tgt, 20.0))
                    print(
                        f"[LEAD] t={tick:>4}"
                        f"  d_obs={_d_obs:>6.1f}m  d_secu={_d_secu:>5.1f}m"
                        f"  v_lead={_v_lead:>5.1f}  v_ego={_v_ego:>5.1f}"
                        f"  v_tgt={_v_tgt:>5.1f} km/h"
                        f"  err_d={_err_d:>+6.1f}  err_v={_err_v:>+5.1f}"
                    )

            # -- 0c. Forçage VERT (--force-red-until) --
            # set_state(Green) PUIS freeze(False) : CARLA reprend le cycle depuis Green.
            # (ancien comportement : freeze=True + set Green → feux bloques indefiniment)
            if force_red_until is not None and tick == force_red_until:
                try:
                    for tl_actor in world.get_actors().filter("traffic.traffic_light*"):
                        tl_actor.set_state(carla.TrafficLightState.Green)
                    world.freeze_all_traffic_lights(False)
                    print(f"\n[TL-FORCE] t={tick}  tous les feux -> GREEN + degel (cycle libre)\n")
                except Exception as exc:
                    print(f"[TL-FORCE] WARN forçage Green ({exc})")

            # -- a. DecisionAgent (FSM) -- AVANT PlanningAgent
            # Lit planning.junction_in_chain du tick precedent (lag 1 tick intentionnel).
            # Publie decision.branch_heading pour que PlanningAgent filtre nexts[].
            decision_agent.run(
                ego_speed_ms=speed_ms,
                ego_heading_deg=heading,
                ego_position=pos,
            )

            # -- b. PlanningAgent -- lit decision.branch_heading (vient d'etre publie)
            prev_traj = list(blackboard.planning.waypoints)
            planning_agent.run(pos[0], pos[1], heading)

            # -- Destination atteinte (navigation avec Global Route Planner) --
            if blackboard.route.active and blackboard.planning.route_complete:
                _dest = blackboard.route.destination
                _d = math.hypot(_dest.x - pos[0], _dest.y - pos[1]) if _dest else float("inf")
                print(
                    f"\n{'='*80}"
                    f"\n  [ROUTE] DESTINATION ATTEINTE  t={tick}"
                    f"  dist_dest={_d:.1f}m"
                    f"  route_idx={planning_agent._route_idx}/{len(blackboard.route.route)}"
                    f"  collisions={collision_count}"
                    f"\n{'='*80}"
                )
                break

            # -- c. ControlAgent --
            control_agent.run(
                current_speed_ms=speed_ms,
                current_heading_deg=heading,
                ego_position=pos,
            )

            # -- d. SafetyAgent (veto transversal, prioritaire) --
            safety_agent.run(ego_speed_ms=speed_ms, tick=tick)

            # -- e. Commandes CARLA (Safety veto > brake-test > normal) --
            ctrl = blackboard.control

            # Freinage d'urgence : override total des commandes (calibration)
            if brake_test_at is not None and tick == brake_test_at and not braking_active:
                braking_active = True
                brake_pos0 = pos
                brake_pos_prev = pos
                brake_v0 = speed_ms
                brake_tick0 = tick
                print(
                    f"\n[BRAKE-TEST] DEBUT t={tick}  "
                    f"v={speed_ms*3.6:.2f} km/h ({speed_ms:.2f} m/s)"
                    f"  pos=({pos[0]:.2f}, {pos[1]:.2f})"
                    f"  brake=0.7"
                )

            if braking_active and not brake_stopped:
                # Accumule la distance parcourue depuis T0
                step_dist = math.hypot(pos[0] - brake_pos_prev[0], pos[1] - brake_pos_prev[1])
                brake_dist_total += step_dist
                brake_pos_prev = pos
                # Log compact chaque tick pendant le freinage
                print(
                    f"[BRAKE-TEST] t={tick:>4}  "
                    f"v={speed_ms*3.6:>6.2f} km/h  "
                    f"dist={brake_dist_total:>5.2f}m"
                )
                if speed_ms < 0.1:
                    brake_stopped = True
                    ttc_emergency = brake_dist_total / max(brake_v0, 0.1) + 0.5
                    print(
                        f"\n[BRAKE-TEST] ARRET t={tick}"
                        f"  d_stop={brake_dist_total:.2f}m"
                        f"  v0={brake_v0:.2f} m/s"
                        f"  TTC_emergency = {brake_dist_total:.2f}/{brake_v0:.2f} + 0.5 "
                        f"= {ttc_emergency:.2f}s"
                    )
                # Override : freinage force, pas de gaz
                ego.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=0.7,
                ))
            elif blackboard.safety.override:
                # Veto Safety Agent (prioritaire sur commandes normales)
                ego.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    steer=float(ctrl.steer),   # cap maintenu
                    brake=float(safety_agent.brake_force),
                ))
            else:
                ego.apply_control(carla.VehicleControl(
                    throttle=float(ctrl.throttle),
                    steer=float(ctrl.steer),
                    brake=float(ctrl.brake),
                ))

            try:
                world.tick()
            except Exception as _tick_exc:
                # CARLA TM crash natif (0xC0000409) après ~290 ticks en mode synchrone.
                # On affiche le résumé partiel avant de propager l'exception.
                print(
                    f"\n[CARLA-CRASH] world.tick() a leve une exception au tick {tick}: {_tick_exc}"
                    f"\n  Résumé partiel : collisions={collision_count}"
                    f"  FALLBK={fallbk_count}  NONE={none_count}"
                    f"  Safety interventions={blackboard.safety.interventions_count}"
                )
                raise

            if spectator_mode:
                ego_tf = ego.get_transform()
                if spectator_mode == "top":
                    spec_loc = ego_tf.location + carla.Location(z=25.0)
                    spectator.set_transform(carla.Transform(
                        spec_loc,
                        carla.Rotation(pitch=-90.0),
                    ))
                else:  # chase
                    spec_loc = ego_tf.transform(carla.Location(x=-6.0, z=3.0))
                    spectator.set_transform(carla.Transform(
                        spec_loc,
                        ego_tf.rotation,
                    ))

            # -- Vanish obstacle (test sortie de veto) --
            if obs_actor is not None and obstacle_vanish_at is not None and tick == obstacle_vanish_at:
                try:
                    obs_actor.destroy()
                except Exception:
                    pass
                obs_actor = None
                print(
                    f"\n[OBSTACLE] Detruit au tick {tick}"
                    f"  -> veto doit se lever dans ~20 ticks (latch={safety_agent._latch_remaining})"
                )

            # -- d. Metriques --
            traj = blackboard.planning.waypoints
            target_speed = traj[0][2] if traj else target_kmh
            err_v = speed_ms * 3.6 - target_speed

            cte = 0.0
            if len(traj) >= 2:
                cte = cross_track_error(
                    pos[0], pos[1],
                    traj[0][0], traj[0][1],
                    traj[1][0], traj[1][1],
                )
            elif traj:
                cte = math.hypot(pos[0] - traj[0][0], pos[1] - traj[0][1])

            collision_count += 1 if env.collision else 0

            is_none_case = (
                ctrl.brake > 0.29
                and ctrl.throttle < 0.01
                and abs(ctrl.steer) < 0.01
            )

            diag = planning_agent._last_diag
            snap_ok = diag.get("snap_coherent", True)
            is_fallbk = not snap_ok
            if is_fallbk:
                fallbk_count += 1
            if is_none_case:
                none_count += 1

            if tick % 10 == 0:
                integral_val = control_agent.longitudinal_integral
                i_term = ki * integral_val
                curv_val = control_agent.last_curvature_radpm
                sp_val  = control_agent.last_steer_p
                sd_val  = control_agent.last_steer_d
                sff_val = control_agent.last_steer_ff
                print(
                    f"  {tick:>5}  {speed_ms*3.6:>7.2f}  {target_speed:>7.1f}  "
                    f"  {err_v:>+6.2f}  {ctrl.steer:>+6.3f}  "
                    f"{ctrl.throttle:>5.3f}  {ctrl.brake:>5.3f}  {cte:>+8.3f}  "
                    f"curv={curv_val:>+7.4f}  "
                    f"sp={sp_val:>+6.3f} sd={sd_val:>+6.3f} sff={sff_val:>+6.3f}  "
                    f"{integral_val:>10.4f}  {i_term:>8.4f}",
                )
                # Log route_idx toutes les 10 ticks (filter "route_idx") quand GRP actif.
                # Prouve avancement monotone continu, stable a l'arret, relance apres virage.
                if blackboard.route.active:
                    _dest = blackboard.route.destination
                    _dist_dest = (
                        math.hypot(_dest.x - pos[0], _dest.y - pos[1])
                        if _dest is not None else float("inf")
                    )
                    _ridx = planning_agent._route_idx
                    _rlen = len(blackboard.route.route)
                    print(
                        f"  [ROUTE] t={tick:>4}  route_idx={_ridx:>3}/{_rlen}"
                        f"  v={speed_ms*3.6:>5.1f}km/h"
                        f"  pos=({pos[0]:>7.1f},{pos[1]:>7.1f})"
                        f"  dist_dest={_dist_dest:>6.1f}m"
                        f"  fsm={blackboard.decision.fsm_state}"
                    )

            # -- Diagnostic yaw des waypoints aux instants clés de la courbe --
            # Loggue les coordonnées et headings bruts des 7 premiers waypoints
            # FF-RAMP diagnostic : profile de montée de steer_ff tick par tick.
            # Montre target (avant rate-limiter) vs ff (après) sur t=230-280.
            # Permet de voir si la montée est en rampe (rate-limiter actif) ou échelon.
            if diag_all and 230 <= tick <= 280:
                _ff_target = control_agent.last_steer_ff_target
                _ff_actual = control_agent.last_steer_ff
                _ff_prev   = control_agent._steer_ff_prev   # etat interne après update
                print(
                    f"  [FF-RAMP t={tick:>3}]  "
                    f"target={_ff_target:+.5f}  "
                    f"ff_actual={_ff_actual:+.5f}  "
                    f"prev_after={_ff_prev:+.5f}  "
                    f"curv={control_agent.last_curvature_radpm:+.5f}"
                )

            # pour identifier si curv se fige parce que les waypoints sont figés
            # (bug planning, toujours les mêmes) ou malgré des waypoints frais
            # (bug dans le calcul de Δheading).
            # Actif seulement si --diag-all ET tick in {250, 295, 350}.
            if diag_all and tick in {250, 295, 350}:
                _wps = list(blackboard.planning.waypoints)
                print(f"\n  {'='*72}")
                print(f"  DIAG-CURV tick={tick}  ego=({pos[0]:.2f},{pos[1]:.2f})"
                      f"  heading={heading:+.2f} deg  curv={control_agent.last_curvature_radpm:+.4f} rad/m")
                print(f"  Waypoints [0..6] utilises dans _compute_curvature_ff :")
                print(f"  {'idx':>4}  {'x':>9}  {'y':>9}  "
                      f"{'yaw_seg(deg)':>13}  {'dist_seg(m)':>11}")
                for _i in range(min(7, len(_wps))):
                    if _i + 1 < len(_wps):
                        _dx = _wps[_i+1][0] - _wps[_i][0]
                        _dy = _wps[_i+1][1] - _wps[_i][1]
                        _yaw = math.degrees(math.atan2(_dy, _dx))
                        _d   = math.hypot(_dx, _dy)
                        print(f"  {_i:>4}  {_wps[_i][0]:>9.3f}  {_wps[_i][1]:>9.3f}  "
                              f"{_yaw:>+13.4f}  {_d:>11.4f}")
                    else:
                        print(f"  {_i:>4}  {_wps[_i][0]:>9.3f}  {_wps[_i][1]:>9.3f}  (dernier)")
                # Δyaw entre segments consécutifs (ce que _compute_curvature_ff calcule)
                print(f"  delta_yaw entre segments (= dheading feed dans kappa) :")
                _headings = []
                for _i in range(min(7, len(_wps) - 1)):
                    _dx = _wps[_i+1][0] - _wps[_i][0]
                    _dy = _wps[_i+1][1] - _wps[_i][1]
                    _headings.append(math.atan2(_dy, _dx))
                for _i in range(len(_headings) - 1):
                    _dh = _headings[_i+1] - _headings[_i]
                    _dh = (_dh + math.pi) % (2 * math.pi) - math.pi
                    _d0 = math.hypot(
                        _wps[_i+1][0] - _wps[_i][0],
                        _wps[_i+1][1] - _wps[_i][1]
                    )
                    _kap = _dh / _d0 if _d0 > 0.1 else 0.0
                    print(f"    seg[{_i}->{ _i+1}]: dh={math.degrees(_dh):>+8.4f} deg  "
                          f"dist={_d0:.4f}m  kappa={_kap:>+.4f} rad/m")
                print(f"  {'='*72}\n")

            # -- Dashboard push (fire-and-forget, jamais bloquant) --
            if _dashboard is not None:
                # FPS measure toutes les 50 ticks
                if tick > 0 and tick % 50 == 0:
                    _fps_elapsed = time.perf_counter() - _fps_t0
                    _fps_curr = 50.0 / _fps_elapsed if _fps_elapsed > 0 else 0.0
                    print(f"[DASHBOARD] t={tick}  FPS={_fps_curr:.1f}  "
                          f"(50 ticks en {_fps_elapsed*1000:.0f} ms"
                          f"  clients={len(_dashboard._clients)})")
                    _fps_t0 = time.perf_counter()

                # Cap les valeurs inf/nan pour JSON valide.
                # NOTE: v < cap est False pour nan et +inf (→ retourne cap).
                #       Mais v < cap est True pour -inf (→ round(-inf) = -inf, BUG).
                #       math.isfinite() corrige les 3 cas : inf, -inf, nan.
                def _fc(v: float, cap: float = 999.0) -> float:
                    return round(min(v, cap), 2) if math.isfinite(v) else round(cap, 2)

                # Horizon planning = somme des distances entre waypoints consécutifs
                _wps = blackboard.planning.waypoints
                _horizon_m = 0.0
                if len(_wps) >= 2:
                    for _wi in range(len(_wps) - 1):
                        _horizon_m += math.hypot(
                            _wps[_wi + 1][0] - _wps[_wi][0],
                            _wps[_wi + 1][1] - _wps[_wi][1],
                        )

                # Obstacle le plus proche sur la voie
                _dets = blackboard.perception.detections
                _closest = _dets[0] if _dets else None

                _dashboard.push({
                    "tick": tick,
                    "fps": round(_fps_curr, 1),
                    "vehicle": {
                        "v_kmh":    round(speed_ms * 3.6, 2),
                        "x":        round(pos[0], 2),
                        "y":        round(pos[1], 2),
                        "yaw":      round(heading, 1),
                        "throttle": round(blackboard.control.throttle, 3),
                        "brake":    round(blackboard.control.brake, 3),
                        "steer":    round(blackboard.control.steer, 3),
                    },
                    "perception": {
                        "n_obstacles":   len(_dets),
                        "closest_dist":  _fc(_closest["forward_dist"]) if _closest else 999.0,
                        "closest_class": _closest["class_name"] if _closest else "",
                        "closest_v":     _closest.get("v_lead_kmh", 0.0) if _closest else 0.0,
                    },
                    "decision": {
                        "fsm_state":    blackboard.decision.fsm_state,
                        "target_speed": round(blackboard.decision.target_speed, 1),
                        "reason":       blackboard.decision.reason,
                    },
                    "planning": {
                        "n_waypoints": len(_wps),
                        "horizon_m":   round(_horizon_m, 1),
                        "branch":      blackboard.planning.branch_in_chain,
                        "cte":         round(cte, 3),
                        "wps_xy":      [[round(w[0], 1), round(w[1], 1)]
                                        for w in _wps],
                    },
                    "control": {
                        "steer":        round(blackboard.control.steer, 3),
                        "throttle":     round(blackboard.control.throttle, 3),
                        "brake":        round(blackboard.control.brake, 3),
                        "target_speed": round(target_speed, 1),
                    },
                    "safety": {
                        "override":      blackboard.safety.override,
                        "ttc":           _fc(blackboard.safety.ttc),
                        "interventions": blackboard.safety.interventions_count,
                        "sensor_fault":  blackboard.safety.sensor_fault,
                    },
                    "tl": {
                        "state":  blackboard.perception.traffic_light_state,
                        "dist_m": _fc(blackboard.perception.traffic_light_dist_m),
                    },
                    "stop": {
                        "detected": blackboard.perception.stop_sign_dist_m < 998.0,
                        "dist_m":   _fc(blackboard.perception.stop_sign_dist_m),
                    },
                })

                # Log échantillon JSON toutes les 100 ticks (preuve vraies valeurs)
                if tick % 100 == 50:
                    import json as _json
                    _sample = {
                        "tick": tick,
                        "v_kmh": round(speed_ms * 3.6, 2),
                        "fsm": blackboard.decision.fsm_state,
                        "cte": round(cte, 3),
                        "ttc": _fc(blackboard.safety.ttc),
                        "tl": blackboard.perception.traffic_light_state,
                    }
                    print(f"[DASHBOARD-SAMPLE] {_json.dumps(_sample)}")

            # -- Log diagnostic --
            # Toujours pour : FALLBK / NONE / zone configuree / ticks autour injection
            is_near_injection = (
                inject_wrongway_at is not None
                and abs(tick - inject_wrongway_at) <= 5
            )
            diag_zone = diag_all or (120 <= tick <= 230) or is_near_injection
            if diag_zone or is_none_case or is_fallbk:
                curr_traj = list(blackboard.planning.waypoints)

                # jump_l2f : DERNIER wp ancien -> PREMIER wp courant (~38m, artefact affichage)
                jump_l2f = 0.0
                if prev_traj and curr_traj:
                    jump_l2f = math.hypot(
                        curr_traj[0][0] - prev_traj[-1][0],
                        curr_traj[0][1] - prev_traj[-1][1],
                    )
                # jump_f2f : PREMIER wp ancien -> PREMIER wp courant (~0.22m, vraie continuite)
                jump_f2f = 0.0
                if prev_traj and curr_traj:
                    jump_f2f = math.hypot(
                        curr_traj[0][0] - prev_traj[0][0],
                        curr_traj[0][1] - prev_traj[0][1],
                    )

                start_yaw = diag.get("start_wp_yaw")
                delta_yaw_val = None
                if start_yaw is not None:
                    delta_yaw_val = ((heading - start_yaw + 180) % 360) - 180

                wp1_heading = None
                if curr_traj:
                    wp1_heading = math.degrees(math.atan2(
                        curr_traj[0][1] - pos[1],
                        curr_traj[0][0] - pos[0],
                    ))

                if is_none_case:
                    flag = "  [NONE!] "
                elif is_fallbk:
                    flag = "  [FALLBK]"
                elif is_near_injection and inject_wrongway_at is not None:
                    rel = tick - inject_wrongway_at
                    flag = f"  [pre={-rel}]" if rel < 0 else f"  [post=+{rel}]"
                else:
                    flag = "          "

                dec = blackboard.decision
                fsm_tag = dec.fsm_state[:5]

                # Signaux FSM (branch) — c'est ce qui compte maintenant
                raw_bic   = diag.get("branch_in_chain", False)
                raw_bdist = diag.get("branch_distance_m", float("inf"))
                raw_bc    = diag.get("branch_count", 0)
                bh_tag    = f"bh={dec.branch_heading:.0f}" if dec.branch_heading is not None else "bh=---"

                # Diagnostics junction (is_junction, informatif uniquement)
                raw_jic   = diag.get("junction_in_chain", False)
                raw_jdist = diag.get("junction_distance_m", float("inf"))
                raw_jwc   = diag.get("junction_wp_count", 0)

                bdist_s = f"{raw_bdist:.1f}m" if raw_bdist < 999 else "inf"
                jdist_s = f"{raw_jdist:.1f}m" if raw_jdist < 999 else "inf"

                # Ligne BRANCH (signal FSM) -- toujours visible si bc>0 ou SLOW_DOWN
                branch_line = (
                    f"bic={'T' if raw_bic else 'F'}  bdist={bdist_s:>6}  bc={raw_bc}  "
                    f"|  jic={'T' if raw_jic else 'F'}  jdist={jdist_s:>6}  jwc={raw_jwc:>2}/{n_wps}"
                )

                curv_diag = control_agent.last_curvature_radpm
                print(
                    f"  [DIAG t={tick:>3}]{flag} "
                    f"ego={heading:>+7.1f}  "
                    f"dYaw={delta_yaw_val if delta_yaw_val is not None else 'N/A':>+7.1f}  "
                    f"f2f={jump_f2f:>5.3f}m  "
                    f"snap={'OK' if snap_ok else 'FALLBK'}  "
                    f"fsm={fsm_tag}  {bh_tag}  "
                    f"curv={curv_diag:>+7.4f}  "
                    f"{branch_line}  "
                    f"v={speed_ms*3.6:>5.2f}  CTE={cte:>+6.3f}"
                )

            # GIL yield CONDITIONNEL : uniquement si --dashboard est actif.
            # Sans ca, les agents Python + prints tiennent le GIL ~10ms/tick (71 Hz)
            # → thread asyncio DashboardWS starve → browser ferme la connexion (1001).
            # Sans --dashboard : _dashboard is None → ce bloc ne s'execute PAS,
            # la boucle CARLA est identique a avant (benchmark / non-regression intacts).
            if _dashboard is not None:
                time.sleep(0.002)

        print()
        print("=" * 80)
        print(f"  Fin : {n_ticks} ticks | {n_ticks*0.05:.1f}s | "
              f"v_finale={ego_speed_ms(ego)*3.6:.2f} km/h")
        print(f"  Collisions : {collision_count}  "
              f"FALLBK total : {fallbk_count}  "
              f"NONE total : {none_count}")
        if inject_wrongway_at is not None:
            status = "OK" if fallbk_count > 0 else "PAS DECLENCHE (verifier)"
            print(f"  Injection t={inject_wrongway_at} : {status}")
        print("=" * 80)

    except Exception as exc:
        logger.error("Erreur : %s", exc, exc_info=True)
    finally:
        if _dashboard is not None:
            _dashboard.stop()
        env.close()
        logger.info("CARLA: acteurs nettoyes, mode synchrone desactive.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demo ControlAgent + PlanningAgent -- lane-following CARLA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Test A (un seul try, road 900)
  .\\venv37\\Scripts\\python.exe -m backend.src.scripts.demo_control_carla \\
      --spawn-road 900 --ticks 200 --diag-all

  # Test B (injection deterministe au tick 100)
  .\\venv37\\Scripts\\python.exe -m backend.src.scripts.demo_control_carla \\
      --inject-wrongway-at 100 --ticks 150 --diag-all

  # Liste des spawn points
  .\\venv37\\Scripts\\python.exe -m backend.src.scripts.demo_control_carla --list-spawns

  # Scan tous les spawns pour trouver ceux avec un vrai fork (bc>0) dans 40m
  .\\venv37\\Scripts\\python.exe -m backend.src.scripts.demo_control_carla --scan-forks

  # Test sur un spawn avec fork (index retourne par --scan-forks)
  .\\venv37\\Scripts\\python.exe -m backend.src.scripts.demo_control_carla \\
      --spawn-index N --ticks 400 --diag-all
        """,
    )
    p.add_argument("--ticks",               type=int,   default=600)
    p.add_argument("--target-kmh",          type=float, default=20.0)
    p.add_argument("--waypoints",           type=int,   default=None)
    p.add_argument("--spacing",             type=float, default=None)
    p.add_argument("--spawn-index",         type=int,   default=None)
    p.add_argument("--spawn-road",          type=int,   default=None)
    p.add_argument("--list-spawns",         action="store_true")
    p.add_argument("--scan-forks",          action="store_true",
                   help="Scan tous les spawns et identifie ceux avec un fork (len>1) dans l'horizon")
    p.add_argument("--diag-all",            action="store_true")
    p.add_argument("--inject-wrongway-at",  type=int,   default=None,
                   help="Injecte un snap cap+180 au tick N (test B deterministe)")
    p.add_argument("--obstacle-at",          type=float, default=None,
                   help="Spawn un obstacle statique a N metres devant l'ego (test Safety TTC)")
    p.add_argument("--stopped-vehicle-at",  type=float, default=None,
                   help="Spawn un vehicule IMMOBILE a N metres devant l'ego (scenario 1 : vehicule arrete). "
                        "Alias de --obstacle-at avec meme comportement. Ex: --stopped-vehicle-at 30")
    p.add_argument("--obstacle-vanish-at",  type=int,   default=None,
                   help="Detruit l'obstacle au tick T (teste la sortie de veto + CLEAR)")
    p.add_argument("--brake-test-at",       type=int,   default=None,
                   help="Force brake=0.7 au tick T, mesure d_stop et TTC_emergency (calibration Safety)")
    p.add_argument("--force-red-until",     type=int,   default=None,
                   help="Gele tous les feux en ROUGE de t=0 a t=T, puis relache (test RED_LIGHT deterministe)")
    p.add_argument("--tl-red-time",         type=float, default=5.0,
                   help="Durée de la phase ROUGE en secondes (defaut=5s). "
                        "Réduit l'attente démo vs 30s CARLA par defaut. Ex: --tl-red-time 8")
    p.add_argument("--tl-green-time",       type=float, default=15.0,
                   help="Durée de la phase VERTE en secondes (defaut=15s). "
                        "Mettre 99 pour navigation longue : TL reste vert tout le run. Ex: --tl-green-time 99")
    p.add_argument("--clear-npcs",          action="store_true",
                   help="Detruit tous les vehicules tiers deja presents dans CARLA avant de spawner l'ego")
    p.add_argument("--spectator",           type=str, choices=["top", "chase"], default=None,
                   help="Vue spectateur CARLA : 'top' (dessus, pitch=-90, z=25) ou 'chase' (3e personne, x=-6, z=3). "
                        "Sans ce flag : pas de mise a jour spectateur (benchmark plus rapide).")
    p.add_argument("--spawn-offset-back",   type=float, default=0.0,
                   help="Recule l'ego de N metres en arriere sur la route avant de demarrer "
                        "(allonge l'approche visible vers un feu / STOP). Ex: --spawn-offset-back 60")
    p.add_argument("--traffic-ahead",       action="store_true",
                   help="Spawn un vehicule lent en autopilot devant l'ego (scenario 3 car-following). "
                        "Combine avec --traffic-speed et --traffic-dist. "
                        "Affiche [LEAD] a chaque tick pour surveiller la convergence.")
    p.add_argument("--traffic-speed",       type=float, default=10.0,
                   help="Vitesse cible du vehicule lead en km/h (defaut=10). "
                        "Cas b : 10 km/h constant. Cas c : commencer a 10, le vehicule s'arrete seul.")
    p.add_argument("--traffic-dist",        type=float, default=20.0,
                   help="Distance de spawn du vehicule lead en metres devant l'ego (defaut=20m).")
    p.add_argument("--pedestrian-cross-at", type=float, default=None,
                   help="Spawn un pieton qui traverse a N metres devant l'ego (scenario 2). "
                        "Place le pieton cote droit, il marche vers la gauche a 1.4 m/s. "
                        "Utiliser sur portion DROITE de la route (limite projection euclidienne). "
                        "Ex: --pedestrian-cross-at 25  (defaut: start-tick=0, speed=0.5m/s)")
    p.add_argument("--pedestrian-start-tick", type=int, default=0,
                   help="Tick auquel le pieton commence a traverser (defaut=0 : immediat). "
                        "Avec crossing=25m et vitesse pieton=0.5m/s : entre dans voie a t~80, "
                        "SLOW_DOWN a t~120, STOP a t~180, reprise a t~200. "
                        "Ex: 0 (defaut, fonctionne avec spawn28 a 6km/h). "
                        "Augmenter si l'ego arrive trop tot (route directe, 20 km/h).")
    p.add_argument("--traffic-brake-at",    type=int,   default=-1,
                   help="Tick auquel le vehicule lead freine a fond jusqu'a l'arret (cas c). "
                        "Desactive l'autopilot du lead et applique brake=1.0 chaque tick. "
                        "Ex: --traffic-brake-at 150  (apres convergence cas b)")
    p.add_argument("--branch-policy",       type=str,
                   choices=["straight", "tl_seeking", "route_following"],
                   default="straight",
                   help="Politique de branchement aux carrefours : 'straight' (GoStraightPolicy), "
                        "'tl_seeking' (branche menant a un TL same_road), "
                        "'route_following' (suit GlobalRoutePlanner -- activer avec --global-planner). "
                        "Ex: --branch-policy tl_seeking")
    p.add_argument("--route-to",            type=int,  default=None,
                   help="Index spawn destination pour navigation A->B (necessite --global-planner). "
                        "Ex: --spawn-index 28 --route-to 34 --global-planner")
    p.add_argument("--global-planner",      action="store_true",
                   help="Active le GlobalRoutePlanner CARLA pour navigation A->B. "
                        "Necessite --route-to N. Calcule la route au demarrage et "
                        "active RouteFollowingBranchPolicy. Sans ce flag : comportement identique a avant.")
    p.add_argument("--dashboard",           action="store_true",
                   help="Active le dashboard WebSocket temps réel (opt-in, aucun overhead sans ce flag). "
                        "Le serveur écoute sur ws://localhost:PORT. Necessite: pip install websockets>=11")
    p.add_argument("--dashboard-port",      type=int, default=8765,
                   help="Port WebSocket du dashboard (defaut=8765). Ex: --dashboard-port 9000")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # --stopped-vehicle-at est un alias de --obstacle-at (même comportement, nom plus descriptif)
    _obstacle_at = args.stopped_vehicle_at if args.stopped_vehicle_at is not None else args.obstacle_at
    run_demo(
        n_ticks=args.ticks,
        target_kmh=args.target_kmh,
        n_waypoints_override=args.waypoints,
        spacing_override=args.spacing,
        spawn_index=args.spawn_index,
        spawn_road=args.spawn_road,
        do_list_spawns=args.list_spawns,
        do_scan_forks=args.scan_forks,
        diag_all=args.diag_all,
        inject_wrongway_at=args.inject_wrongway_at,
        obstacle_at=_obstacle_at,
        obstacle_vanish_at=args.obstacle_vanish_at,
        brake_test_at=args.brake_test_at,
        force_red_until=args.force_red_until,
        tl_red_time=args.tl_red_time,
        tl_green_time=args.tl_green_time,
        clear_npcs=args.clear_npcs,
        spectator_mode=args.spectator,
        spawn_offset_back=args.spawn_offset_back,
        branch_policy=args.branch_policy,
        route_to=args.route_to,
        use_global_planner=args.global_planner,
        pedestrian_cross_at=args.pedestrian_cross_at,
        pedestrian_start_tick=args.pedestrian_start_tick,
        traffic_ahead=args.traffic_ahead,
        traffic_speed_kmh=args.traffic_speed,
        traffic_distance_m=args.traffic_dist,
        traffic_brake_at_tick=args.traffic_brake_at,
        dashboard_port=args.dashboard_port if args.dashboard else None,
    )

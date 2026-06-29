"""
Script d'inférence — BaselineAgent (TransFuser) dans CARLA 0.9.15
Usage :
    1. Lancer CARLA : C:\pfa\WindowsNoEditor\CarlaUE4.exe
    2. Activer venv37 : .\venv37\Scripts\Activate.ps1
    3. Lancer : python run_baseline.py
"""

import sys
import time
import logging
import numpy as np
import yaml
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ── Charger config ────────────────────────────────────────────────────────────
with open("backend/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CARLA_HOST = config["carla"]["host"]
CARLA_PORT = config["carla"]["port"]
WEIGHTS    = config["baseline"]["model_path"]
EPISODES   = 3
MAX_STEPS  = 200

# ── Import CARLA ──────────────────────────────────────────────────────────────
try:
    import carla
    log.info("carla importé OK (version %s)", carla.__version__ if hasattr(carla, "__version__") else "0.9.15")
except ImportError as e:
    log.error("Impossible d'importer carla : %s", e)
    sys.exit(1)

# ── Charger les poids TransFuser directement ──────────────────────────────────
log.info("Chargement des poids depuis %s ...", WEIGHTS)
state_dict = torch.load(WEIGHTS, map_location="cpu")
log.info("Poids chargés — %d paramètres dans le state_dict", len(state_dict))

# Afficher les premières clés pour diagnostic
sample_keys = list(state_dict.keys())[:5]
log.info("Exemple de clés : %s", sample_keys)

# ── Connexion à CARLA ─────────────────────────────────────────────────────────
log.info("Connexion à CARLA %s:%d ...", CARLA_HOST, CARLA_PORT)
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(30.0)

try:
    world = client.get_world()
    log.info("Connecté — carte : %s", world.get_map().name)
except Exception as e:
    log.error("Connexion échouée : %s", e)
    log.error("Vérifie que CarlaUE4.exe est lancé !")
    sys.exit(1)

# ── Mode synchrone ────────────────────────────────────────────────────────────
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

blueprint_lib = world.get_blueprint_library()
vehicle_bp    = blueprint_lib.find("vehicle.tesla.model3")
spawn_points  = world.get_map().get_spawn_points()

actors = []

def cleanup():
    log.info("Nettoyage des acteurs ...")
    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass
    settings.synchronous_mode = False
    world.apply_settings(settings)

# ── Caméra RGB ────────────────────────────────────────────────────────────────
camera_data = {"image": None}

def camera_callback(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]   # BGRA → RGB
    camera_data["image"] = arr

try:
    for ep in range(EPISODES):
        log.info("=== Épisode %d/%d ===", ep + 1, EPISODES)

        # Spawner le véhicule
        sp = spawn_points[ep % len(spawn_points)]
        vehicle = world.spawn_actor(vehicle_bp, sp)
        actors.append(vehicle)

        # Attacher caméra RGB
        cam_bp = blueprint_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "84")
        cam_bp.set_attribute("image_size_y", "84")
        cam_bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        actors.append(camera)
        camera.listen(camera_callback)

        # Attendre la première image
        for _ in range(10):
            world.tick()
            if camera_data["image"] is not None:
                break

        # Déplacer le spectateur CARLA sur le véhicule dès le départ
        spectator = world.get_spectator()

        rewards = []
        for step in range(MAX_STEPS):
            world.tick()

            # ── Spectateur suit le véhicule (vue arrière surélevée) ──────────
            v_tf = vehicle.get_transform()
            spec_tf = carla.Transform(
                v_tf.transform(carla.Location(x=-6.0, z=3.5)),
                carla.Rotation(pitch=-15, yaw=v_tf.rotation.yaw)
            )
            spectator.set_transform(spec_tf)

            # ── Observation ──────────────────────────────────────────────────
            frame = camera_data["image"]
            if frame is None:
                frame = np.zeros((84, 84, 3), dtype=np.uint8)

            vel = vehicle.get_velocity()
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5  # m/s

            # ── Contrôle du véhicule ─────────────────────────────────────────
            if speed < 5.0:
                throttle, steer, brake = 0.6, 0.0, 0.0
            else:
                throttle, steer, brake = 0.3, 0.0, 0.0

            control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
            )
            vehicle.apply_control(control)

            reward = speed - 0.1 * abs(steer)
            rewards.append(reward)

            if step % 50 == 0:
                log.info("  Step %3d | speed=%.1f m/s | reward=%.2f", step, speed, reward)

        log.info("  Épisode %d terminé — reward moyen = %.2f", ep + 1, np.mean(rewards))

        # Détruire acteurs de l'épisode
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        actors.clear()
        time.sleep(1.0)

    log.info("=== TERMINÉ ===")

except KeyboardInterrupt:
    log.info("Interrompu par l'utilisateur.")
finally:
    cleanup()

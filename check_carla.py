"""
Verification CARLA avant entrainement/inference.
Teste : connexion, spawn vehicule, spectateur, capteurs.
Usage : python check_carla.py
"""
import sys
import time
import math
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

try:
    import carla
except ImportError:
    log.error("carla non trouve. Active venv37 : .\\venv37\\Scripts\\Activate.ps1")
    sys.exit(1)

HOST, PORT = "localhost", 2000

def check():
    ok = True

    # ── 1. Connexion ─────────────────────────────────────────────────────────
    log.info("[1/5] Connexion a CARLA %s:%d ...", HOST, PORT)
    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)
        world  = client.get_world()
        log.info("      OK — carte : %s | version : %s",
                 world.get_map().name, client.get_server_version())
    except Exception as e:
        log.error("      ECHEC — %s", e)
        log.error("      Verifier que CarlaUE4.exe est lance !")
        return False

    # ── 2. Mode synchrone ────────────────────────────────────────────────────
    log.info("[2/5] Activation mode synchrone ...")
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.tick()
    log.info("      OK — delta=0.05s")

    # ── 3. Spawn vehicule ────────────────────────────────────────────────────
    log.info("[3/5] Spawn vehicule test ...")
    bplib  = world.get_blueprint_library()
    bp     = bplib.find("vehicle.tesla.model3")
    sps    = world.get_map().get_spawn_points()
    vehicle = None
    try:
        vehicle = world.spawn_actor(bp, sps[0])
        world.tick()
        loc = vehicle.get_location()
        log.info("      OK — vehicule a (%.1f, %.1f, %.1f)", loc.x, loc.y, loc.z)
    except Exception as e:
        log.error("      ECHEC spawn : %s", e)
        ok = False

    if not vehicle:
        _restore(world, settings)
        return False

    # ── 4. Spectateur suit le vehicule ───────────────────────────────────────
    log.info("[4/5] Test spectateur (verifiez la fenetre CARLA) ...")
    spectator = world.get_spectator()
    sp_tf     = sps[0]

    # Teleporter immediatement sur le spawn
    spectator.set_transform(carla.Transform(
        sp_tf.transform(carla.Location(x=-10.0, z=6.0)),
        carla.Rotation(pitch=-20, yaw=sp_tf.rotation.yaw)
    ))
    world.tick()
    time.sleep(0.5)

    # Avancer le vehicule et suivre
    for i in range(30):
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
        v_tf = vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            v_tf.transform(carla.Location(x=-8.0, z=5.0)),
            carla.Rotation(pitch=-20, yaw=v_tf.rotation.yaw)
        ))
        world.tick()

    vel   = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
    if speed > 1.0:
        log.info("      OK — vehicule roule a %.1f km/h (spectateur suit)", speed)
    else:
        log.warning("      ATTENTION — vehicule immobile (%.1f km/h)", speed)
        ok = False

    # ── 5. Camera RGB ────────────────────────────────────────────────────────
    log.info("[5/5] Test capteur camera RGB ...")
    cam_data = {"ok": False}

    def cam_cb(img):
        cam_data["ok"] = True

    cam_bp = bplib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "256")
    cam_bp.set_attribute("image_size_y", "256")
    cam    = world.spawn_actor(cam_bp,
                 carla.Transform(carla.Location(x=1.5, z=2.4)),
                 attach_to=vehicle)
    cam.listen(cam_cb)

    for _ in range(10):
        world.tick()
        if cam_data["ok"]:
            break

    cam.stop(); cam.destroy()

    if cam_data["ok"]:
        log.info("      OK — images RGB recues")
    else:
        log.error("      ECHEC — aucune image camera recue")
        ok = False

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    vehicle.destroy()
    _restore(world, settings)

    # ── Rapport ──────────────────────────────────────────────────────────────
    print()
    if ok:
        log.info("=" * 50)
        log.info("CARLA OK — pret pour l'entrainement / inference !")
        log.info("=" * 50)
    else:
        log.error("=" * 50)
        log.error("PROBLEMES DETECTES — voir les erreurs ci-dessus")
        log.error("=" * 50)
    return ok


def _restore(world, settings):
    try:
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except Exception:
        pass


if __name__ == "__main__":
    success = check()
    sys.exit(0 if success else 1)

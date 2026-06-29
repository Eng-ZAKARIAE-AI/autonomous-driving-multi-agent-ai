"""Test WS isolé — serveur dashboard SANS CARLA.

Usage :
  python backend/src/scripts/test_ws_standalone.py

Ouvre ensuite http://localhost:8765 n'importe comment, ou ouvre step2.html.
Le serveur envoie un payload fixe (avec quelques champs qui varient) toutes les 50ms.
Observe dans la console : la connexion tient-elle >30s sans deconnexion ?

Ce test tranche entre :
  - Problème GIL/thread CARLA qui étouffe asyncio   → tient ici, tombe avec CARLA
  - Bug intrinsèque du serveur WS ou du client HTML  → tombe aussi ici
"""

import sys
import time
import math

# Assure que le repo racine est dans le path.
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.src.dashboard.ws_server import DashboardServer

def main() -> None:
    srv = DashboardServer(host="localhost", port=8765)
    ok = srv.start()
    if not ok:
        print("ERREUR : websockets non installé. pip install websockets>=11")
        sys.exit(1)

    print("[TEST] Serveur démarré sur ws://localhost:8765")
    print("[TEST] Ouvre step2.html dans le navigateur.")
    print("[TEST] Durée du test : 120 s. Ctrl+C pour arrêter.")
    print("[TEST] Si la connexion tient 120s → le problème est le GIL/CARLA.")
    print("[TEST] Si elle tombe → bug intrinsèque serveur WS ou HTML.")
    print()

    # Simule une route A→B (trajectoire en L pour tester auto-scale)
    route_pts = []
    for i in range(60):   # segment horizontal 120 m
        route_pts.append([round(i * 2.0, 1), 0.0])
    for i in range(1, 40):  # virage + segment vertical 80 m
        route_pts.append([118.0, round(i * 2.0, 1)])
    srv.set_route(route_pts)
    print(f"[TEST] Route simulee envoyee : {len(route_pts)} pts")

    t0 = time.perf_counter()
    tick = 0

    try:
        while True:
            t = time.perf_counter() - t0
            # Simule un payload complet (valeurs qui varient pour que les graphiques bougent)
            v_kmh = 16.0 + 4.0 * math.sin(t * 0.5)
            cte   = 0.15 * math.sin(t * 1.3)
            states = ["FOLLOW_LANE", "SLOW_DOWN", "RED_LIGHT"]
            fsm = states[int(t / 10) % 3]

            srv.push({
                "tick": tick,
                "fps": 71.5,
                "vehicle": {
                    "v_kmh":    round(v_kmh, 2),
                    "x":        round(10.0 + t * 4.0, 2),
                    "y":        -20.0,
                    "yaw":      90.0,
                    "throttle": 0.4,
                    "brake":    0.0,
                    "steer":    round(cte * 0.5, 3),
                },
                "perception": {
                    "n_obstacles":   0,
                    "closest_dist":  999.0,
                    "closest_class": "",
                    "closest_v":     0.0,
                },
                "decision": {
                    "fsm_state":    fsm,
                    "target_speed": 16.0,
                    "reason":       "test isolant",
                },
                "planning": {
                    "n_waypoints": 10,
                    "horizon_m":   20.0,
                    "branch":      False,
                    "cte":         round(cte, 3),
                    "wps_xy":      [
                        [round(10.0 + t * 4.0 + i * 2.0, 1), round(cte * 0.5, 1)]
                        for i in range(10)
                    ],
                },
                "control": {
                    "steer":        round(cte * 0.5, 3),
                    "throttle":     0.4,
                    "brake":        0.0,
                    "target_speed": 16.0,
                },
                "safety": {
                    "override":      False,
                    "ttc":           None,   # équivalent de inf sanitisé
                    "interventions": 0,
                    "sensor_fault":  False,
                },
                "tl": {
                    "state":  "Green" if fsm == "FOLLOW_LANE" else "Red",
                    "dist_m": round(max(0.0, 40.0 - t % 40.0), 1),
                },
                "stop": {
                    "detected": False,
                    "dist_m":   999.0,
                },
            })

            tick += 1

            # Log toutes les 10s pour montrer que le serveur est vivant
            if tick % 200 == 0:
                elapsed = time.perf_counter() - t0
                print(f"[TEST] t={elapsed:.0f}s  tick={tick}  clients={len(srv._clients)}")

            if time.perf_counter() - t0 > 120:
                print("[TEST] 120s écoulées — test terminé.")
                break

            time.sleep(0.05)  # 20 Hz, même fréquence que la simu CARLA

    except KeyboardInterrupt:
        print("\n[TEST] Arrêt manuel.")
    finally:
        srv.stop()
        print("[TEST] Serveur arrêté.")
        elapsed = time.perf_counter() - t0
        print(f"[TEST] Durée totale : {elapsed:.1f}s  ticks pushés : {tick}")

if __name__ == "__main__":
    main()

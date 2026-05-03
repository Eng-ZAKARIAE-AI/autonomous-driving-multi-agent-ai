import argparse
import os

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Démarrage du projet CARLA pour la détection de chute")
    parser.add_argument("--host", default=os.getenv("CARLA_HOST", "localhost"), help="Adresse du serveur CARLA")
    parser.add_argument("--port", type=int, default=int(os.getenv("CARLA_PORT", 2000)), help="Port du serveur CARLA")
    parser.add_argument("--map", default="Town03", help="Carte CARLA à charger")
    parser.add_argument("--frames", type=int, default=300, help="Nombre de frames de simulation")
    parser.add_argument("--throttle", type=float, default=0.4, help="Commande d'accélération du véhicule ego")
    parser.add_argument("--steer", type=float, default=0.05, help="Commande de direction du véhicule ego")
    parser.add_argument("--visualize", action="store_true", help="Afficher la caméra CARLA pendant la simulation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from envs.carla_simulation import CarlaSimulation, SimpleFallDetector

    simulation = CarlaSimulation(host=args.host, port=args.port, map_name=args.map)
    detector = SimpleFallDetector()

    if args.visualize and cv2 is None:
        raise RuntimeError(
            "OpenCV n'est pas installé. Installez opencv-python pour activer la visualisation avec --visualize."
        )

    try:
        state = simulation.reset()
        print(f"✅ Connecté à CARLA : {args.host}:{args.port} sur {args.map}")

        for frame in range(args.frames):
            if frame == int(args.frames * 0.4):
                print("⚠️  Simulation d'un événement : chute du piéton")
                simulation.simulate_fall_event()

            simulation.apply_ego_control(throttle=args.throttle, steer=args.steer)
            simulation.tick()
            state = simulation.get_state()
            event = detector.predict(state)

            if args.visualize and state["camera"] is not None:
                frame_image = state["camera"].copy()
                label = f"Etat: {event}"
                cv2.putText(
                    frame_image,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 200, 0) if event == "normal" else (0, 0, 255),
                    2,
                )
                cv2.imshow("CARLA Camera", frame_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if event != "normal":
                print(f"[FRAME {frame}] événement détecté -> {event}")

        print("🎯 Simulation terminée")
    finally:
        simulation.close()
        if args.visualize and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

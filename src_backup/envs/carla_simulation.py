import random
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import carla
import cv2
import numpy as np
import torch


class CarlaSimulation:
    """Gestion complète d'une simulation CARLA pour un scénario de chute/accident."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        map_name: str = "Town03",
        synchronous: bool = True,
        fixed_delta_seconds: float = 0.05,
    ):
        self.host = host
        self.port = port
        self.map_name = map_name
        self.synchronous = synchronous
        self.fixed_delta_seconds = fixed_delta_seconds

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(60.0)
        self._wait_for_server_ready()

        self.world: Optional[carla.World] = None
        self.blueprint_library: Optional[carla.BlueprintLibrary] = None
        self.ego_vehicle: Optional[carla.Actor] = None
        self.pedestrian: Optional[carla.Actor] = None
        self.sensor_actors: list[carla.Actor] = []
        self.actors: list[carla.Actor] = []

        self.state: Dict[str, Any] = {
            "camera": None,
            "collision": False,
            "collision_history": deque(maxlen=20),
            "ego_location": None,
            "pedestrian_location": None,
            "pedestrian_speed": 0.0,
            "pedestrian_height": None,
            "timestamp": 0.0,
        }

        self._previous_ped_position: Optional[carla.Location] = None
        self._fall_simulated = False

        self._load_world()
        self._configure_world()

    def _load_world(self) -> None:
        for attempt in range(1, 6):
            try:
                if self.map_name:
                    self.world = self.client.load_world(self.map_name)
                else:
                    self.world = self.client.get_world()
                self.blueprint_library = self.world.get_blueprint_library()
                return
            except Exception as exc:
                if attempt == 5:
                    raise RuntimeError(
                        f"time-out while loading CARLA world '{self.map_name}' after {attempt} attempts. "
                        f"Make sure CARLA is ready and connected to {self.host}:{self.port}"
                    ) from exc
                time.sleep(2.0)

    def _wait_for_server_ready(self) -> None:
        for attempt in range(1, 7):
            try:
                self.client.get_server_version()
                return
            except Exception:
                if attempt == 6:
                    raise RuntimeError(
                        f"Unable to reach CARLA server at {self.host}:{self.port}. "
                        "Start the simulator and wait until it is fully initialized."
                    )
                time.sleep(2.0)

    def _configure_world(self) -> None:
        assert self.world is not None
        settings = self.world.get_settings()
        self._original_settings = settings
        if self.synchronous:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta_seconds
        else:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def reset(self) -> Dict[str, Any]:
        self._destroy_all_actors()
        self._load_world()
        self._configure_world()
        self._spawn_ego_vehicle()
        self._spawn_pedestrian()
        self._spawn_sensors()
        self._previous_ped_position = self.state["pedestrian_location"]
        self._fall_simulated = False
        self.world.tick()
        return self.get_state()

    def _spawn_ego_vehicle(self) -> None:
        assert self.world is not None and self.blueprint_library is not None
        vehicle_bp = random.choice(self.blueprint_library.filter("vehicle.tesla.model3"))
        spawn_points = self.world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)
        self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        if self.ego_vehicle is None:
            raise RuntimeError("Impossible de créer le véhicule ego")
        self.actors.append(self.ego_vehicle)
        self.ego_vehicle.set_autopilot(False)
        self.state["ego_location"] = self.ego_vehicle.get_location()

    def _spawn_pedestrian(self) -> None:
        assert self.world is not None and self.blueprint_library is not None
        walker_bp = random.choice(self.blueprint_library.filter("walker.pedestrian.*"))
        walker_bp.set_attribute("is_invincible", "false")
        walker_bp.set_attribute("speed", "1.0")

        # Try multiple spawn locations until one works
        for attempt in range(10):
            pedestrian_spawn = self._find_pedestrian_spawn()
            self.pedestrian = self.world.try_spawn_actor(walker_bp, pedestrian_spawn)
            if self.pedestrian is not None:
                break
        else:
            raise RuntimeError("Impossible de créer le piéton après 10 tentatives")

        self.actors.append(self.pedestrian)
        self.state["pedestrian_location"] = self.pedestrian.get_location()
        self.state["pedestrian_height"] = self.state["pedestrian_location"].z
        self._set_walker_control()

    def _find_pedestrian_spawn(self) -> carla.Transform:
        assert self.world is not None
        spawn_points = self.world.get_map().get_spawn_points()

        # Try to find a good pedestrian spawn location
        for _ in range(20):  # Try up to 20 different locations
            base_spawn = random.choice(spawn_points)
            # Create a pedestrian spawn near the vehicle spawn but with some variation
            offset_x = random.uniform(-5.0, 5.0)
            offset_y = random.uniform(5.0, 15.0)  # Keep some distance from vehicle

            pedestrian_location = carla.Location(
                x=base_spawn.location.x + offset_x,
                y=base_spawn.location.y + offset_y,
                z=base_spawn.location.z + 1.0  # Slightly above ground to avoid spawning underground
            )

            # Check if the location is valid by casting a ray downward
            world_snapshot = self.world.get_snapshot()
            ray_start = pedestrian_location
            ray_end = carla.Location(
                x=pedestrian_location.x,
                y=pedestrian_location.y,
                z=pedestrian_location.z - 10.0  # Cast ray 10 meters down
            )

            # If we can find ground, adjust z to be on ground level
            hits = self.world.cast_ray(ray_start, ray_end)
            if hits:
                # Place pedestrian on the ground
                ground_z = hits[0].location.z
                pedestrian_location.z = ground_z + 0.1  # Slightly above ground

            return carla.Transform(pedestrian_location, base_spawn.rotation)

        # Fallback: use a simple offset from a random spawn point
        position = random.choice(spawn_points)
        position.location.y += 10
        return position

    def _set_walker_control(self) -> None:
        assert self.pedestrian is not None
        control = carla.WalkerControl()
        control.speed = 1.2
        control.direction = carla.Vector3D(x=1.0, y=0.0, z=0.0)
        self.pedestrian.apply_control(control)

    def _spawn_sensors(self) -> None:
        assert self.world is not None and self.ego_vehicle is not None and self.blueprint_library is not None
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        camera.listen(self._camera_callback)
        self.sensor_actors.append(camera)

        collision_bp = self.blueprint_library.find("sensor.other.collision")
        collision = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        collision.listen(self._collision_callback)
        self.sensor_actors.append(collision)

        gnss_bp = self.blueprint_library.find("sensor.other.gnss")
        gnss = self.world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.ego_vehicle)
        gnss.listen(self._gnss_callback)
        self.sensor_actors.append(gnss)

    def _camera_callback(self, image: carla.Image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        self.state["camera"] = array
        self.state["timestamp"] = image.timestamp

    def _collision_callback(self, event: carla.CollisionEvent) -> None:
        self.state["collision"] = True
        self.state["collision_history"].append(event)

    def _gnss_callback(self, event: carla.GnssMeasurement) -> None:
        self.state["ego_location"] = carla.Location(event.latitude, event.longitude, event.altitude)

    def _update_pedestrian_state(self) -> None:
        if self.pedestrian is None:
            return
        location = self.pedestrian.get_location()
        self.state["pedestrian_location"] = location
        self.state["pedestrian_height"] = location.z
        if self._previous_ped_position is not None:
            distance = location.distance(self._previous_ped_position)
            self.state["pedestrian_speed"] = distance / self.fixed_delta_seconds
        self._previous_ped_position = location

    def tick(self) -> None:
        assert self.world is not None
        if self.synchronous:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        self._update_pedestrian_state()
        self.state["collision"] = False

    def get_state(self) -> Dict[str, Any]:
        return dict(self.state)

    def apply_ego_control(self, throttle: float = 0.5, steer: float = 0.0) -> None:
        if self.ego_vehicle is None:
            return
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)
        self.ego_vehicle.apply_control(control)

    def simulate_fall_event(self) -> None:
        if self._fall_simulated or self.pedestrian is None:
            return
        location = self.pedestrian.get_location()
        fallen_location = carla.Location(location.x, location.y, location.z - 0.8)
        current_rotation = self.pedestrian.get_transform().rotation
        fallen_rotation = carla.Rotation(pitch=90.0, yaw=current_rotation.yaw, roll=0.0)
        self.pedestrian.set_transform(carla.Transform(fallen_location, fallen_rotation))
        control = carla.WalkerControl()
        control.speed = 0.0
        control.direction = carla.Vector3D(0.0, 0.0, 0.0)
        self.pedestrian.apply_control(control)
        self._fall_simulated = True

    def _destroy_all_actors(self) -> None:
        for actor in self.sensor_actors + self.actors:
            if actor is not None:
                actor.destroy()
        self.sensor_actors = []
        self.actors = []
        self.ego_vehicle = None
        self.pedestrian = None

    def close(self) -> None:
        if self.world is not None:
            self.world.apply_settings(self._original_settings)
        self._destroy_all_actors()


class SimpleFallDetector:
    """Détecteur simple construit pour illustrer l'intégration d'un modèle IA."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_dummy_model().to(self.device)

    def _build_dummy_model(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 224 * 224, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.resize(image, (224, 224))
        tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor

    def predict(self, sensor_data: Dict[str, Any]) -> str:
        if sensor_data["collision"]:
            return "collision_detected"

        if sensor_data["pedestrian_height"] is not None and sensor_data["pedestrian_speed"] < 0.4:
            if sensor_data["pedestrian_height"] < 1.0:
                return "fall_detected"

        image = sensor_data["camera"]
        if image is not None:
            tensor = self.preprocess(image)
            with torch.no_grad():
                logits = self.model(tensor)
                scores = torch.softmax(logits, dim=1).cpu().numpy()[0]
                if scores[1] > 0.85:
                    return "possible_fall_from_model"
        return "normal"


def run_demo_loop() -> None:
    simulation = CarlaSimulation()
    detector = SimpleFallDetector()

    try:
        state = simulation.reset()
        print("✅ Simulation CARLA initialisée")

        for frame in range(300):
            if frame == 120:
                print("⚠️  Simulation d'une chute de piéton")
                simulation.simulate_fall_event()

            simulation.apply_ego_control(throttle=0.4, steer=0.05)
            simulation.tick()
            state = simulation.get_state()
            event = detector.predict(state)

            if state["camera"] is not None:
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

        print("🎯 Fin du scénario de démonstration")
    finally:
        simulation.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo_loop()

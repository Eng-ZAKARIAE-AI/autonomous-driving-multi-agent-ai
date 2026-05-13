"""CARLA environment wrapper with a Gym-like interface."""

import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import carla
import numpy as np

from backend.src.config import config
from backend.src.reward.reward_function import RewardFunction
from backend.src.models.state_builder import StateBuilder


def _vector_length(vector: Tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(vector, dtype=np.float32)))


class CarlaGymEnv:
    """Gym-like CARLA environment for RL training."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.traffic_manager: Optional[carla.TrafficManager] = None
        self.ego: Optional[carla.Actor] = None
        self.collision_sensor: Optional[carla.Actor] = None
        self.camera_sensor: Optional[carla.Actor] = None
        self.camera_image: Optional[np.ndarray] = None
        self.collision = False
        self.goal_location: Optional[carla.Location] = None
        self.step_count = 0
        self.state_builder = StateBuilder(config)
        self.reward_fn = RewardFunction(config)
        self.last_observation: Dict[str, Any] = {}

    def initialize(self) -> bool:
        return self.connect()

    def connect(self) -> bool:
        host = self.config['carla']['host']
        port = int(self.config['carla']['port'])
        timeout = float(self.config['carla']['timeout'])
        retries = int(self.config['carla'].get('connect_retries', 10))
        delay = float(self.config['carla'].get('retry_delay', 1.0))

        auto_launch_enabled = self.config['carla'].get('auto_launch', False)
        launch_command = self.config['carla'].get('launch_command')
        carla_root = self.config['carla'].get('carla_root') or os.getenv('CARLA_ROOT')
        if auto_launch_enabled or launch_command or carla_root:
            self._auto_launch_carla()

        for attempt in range(1, retries + 1):
            try:
                self.client = carla.Client(host, port)
                self.client.set_timeout(timeout)
                self.world = self.client.load_world(self.config['carla']['map'])

                if self.config['carla']['synchronous']:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = float(self.config['carla']['fixed_delta_seconds'])
                    self.world.apply_settings(settings)

                self.traffic_manager = self.client.get_trafficmanager()
                self.traffic_manager.set_synchronous_mode(self.config['carla']['synchronous'])
                return True
            except Exception as exc:
                print(f"❌ CARLA initialization failed (attempt {attempt}/{retries}): {exc}")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    print("🚨 CARLA could not be reached after retries."
                          " Start the simulator manually or set carla.auto_launch=true"
                          " with carla.carla_root or carla.launch_command in config/config.yaml.")
                    return False

    def reset(self) -> Dict[str, Any]:
        self._destroy_actors()
        self.collision = False
        self.camera_image = None
        self.step_count = 0

        self._spawn_ego()
        self._spawn_traffic()
        self._setup_collision_sensor()
        self._setup_camera_sensor()
        self.goal_location = self._build_goal_point()
        self.reward_fn.reset()
        self._update_spectator()

        if self.config['carla']['synchronous']:
            self.world.tick()
        else:
            time.sleep(float(self.config['carla']['fixed_delta_seconds']))

        observation = self._get_observation()
        self.last_observation = observation
        return self.state_builder.build_state(observation)

    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.ego is None:
            raise RuntimeError('Ego vehicle not spawned.')

        self.step_count += 1
        self.collision = False

        steer = float(np.clip(action.get('steer', 0.0), -1.0, 1.0))
        throttle = float(np.clip(action.get('throttle', 0.0), 0.0, 1.0))
        brake = float(np.clip(action.get('brake', 0.0), 0.0, 1.0))

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego.apply_control(control)

        if self.config['carla']['synchronous']:
            self.world.tick()
        else:
            time.sleep(float(self.config['carla']['fixed_delta_seconds']))

        self._update_spectator()
        observation = self._get_observation()
        state = self.state_builder.build_state(observation)
        reward = self.reward_fn.compute(observation, action)
        done = self._is_done(observation)
        info = {
            'collision': self.collision,
            'speed': observation['speed'],
            'lane_offset': observation['lane_offset'],
            'goal_distance': observation['goal_distance']
        }
        self.last_observation = observation
        return state, reward, done, info

    def close(self) -> None:
        self._destroy_actors()
        if self.world and self.config['carla']['synchronous']:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

    def _auto_launch_carla(self) -> None:
        carla_root = self.config['carla'].get('carla_root') or os.getenv('CARLA_ROOT')
        launch_command = self.config['carla'].get('launch_command')
        if launch_command:
            try:
                subprocess.Popen(launch_command, shell=True)
            except Exception as exc:
                print(f"⚠️ Failed to auto-launch CARLA with command: {exc}")
        elif carla_root:
            executable = os.path.join(carla_root, 'CarlaUE4.exe')
            if os.path.exists(executable):
                try:
                    subprocess.Popen(f'"{executable}" -opengl4', shell=True)
                except Exception as exc:
                    print(f"⚠️ Failed to auto-launch CARLA executable: {exc}")
            else:
                print(f"⚠️ CARLA executable not found at {executable}")
        else:
            print("⚠️ CARLA auto-launch requested but no launch_command or carla_root provided.")

    def _spawn_ego(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        ego_bp = random.choice(blueprint_library.filter(self.config['carla']['ego_filter']))
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            raise RuntimeError('No spawn points available.')

        ego_spawn = random.choice(spawn_points)
        self.ego = self.world.try_spawn_actor(ego_bp, ego_spawn)
        if self.ego is None:
            raise RuntimeError('Failed to spawn ego vehicle.')

        self.ego.set_autopilot(False)
        self.actors = [self.ego]

    def _spawn_traffic(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        num_vehicles = min(int(self.config['carla']['traffic_vehicles']), max(0, len(spawn_points) - 1))

        for index in range(num_vehicles):
            spawn_point = spawn_points[(index + 1) % len(spawn_points)]
            npc_bp = random.choice(vehicle_blueprints)
            npc = self.world.try_spawn_actor(npc_bp, spawn_point)
            if npc is not None:
                npc.set_autopilot(True, self.traffic_manager.get_port())
                self.actors.append(npc)

    def _setup_collision_sensor(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        sensor_bp = blueprint_library.find('sensor.other.collision')
        transform = carla.Transform(carla.Location(x=0.0, z=0.0))
        self.collision_sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=self.ego)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.actors.append(self.collision_sensor)

    def _setup_camera_sensor(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.config['camera']['width']))
        camera_bp.set_attribute('image_size_y', str(self.config['camera']['height']))
        camera_bp.set_attribute('fov', str(self.config['camera']['fov']))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego)
        self.camera_sensor.listen(lambda image: self._on_camera_image(image))
        self.actors.append(self.camera_sensor)

    def _on_camera_image(self, image: carla.Image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.camera_image = array[:, :, :3]

    def _build_goal_point(self) -> carla.Location:
        ego_transform = self.ego.get_transform()
        ego_waypoint = self.world.get_map().get_waypoint(ego_transform.location, project_to_road=True)
        future_waypoints = ego_waypoint.next(50)
        if future_waypoints:
            return future_waypoints[0].transform.location
        return ego_transform.location + carla.Location(x=100.0, y=0.0, z=0.0)

    def _update_spectator(self) -> None:
        spectator = self.world.get_spectator()
        transform = self.ego.get_transform()
        camera_location = transform.location + carla.Location(x=-12.0, z=8.0)
        spectator_transform = carla.Transform(camera_location, transform.rotation)
        spectator.set_transform(spectator_transform)

    def _get_observation(self) -> Dict[str, Any]:
        transform = self.ego.get_transform()
        velocity = self.ego.get_velocity()
        speed = _vector_length((velocity.x, velocity.y, velocity.z))

        waypoint = self.world.get_map().get_waypoint(transform.location, project_to_road=True)
        lane_offset = 0.0
        if waypoint is not None:
            dx = transform.location.x - waypoint.transform.location.x
            dy = transform.location.y - waypoint.transform.location.y
            lane_offset = float(np.dot(np.array([dx, dy], dtype=np.float32),
                                       np.array([np.cos(np.radians(transform.rotation.yaw + 90.0)),
                                                 np.sin(np.radians(transform.rotation.yaw + 90.0))], dtype=np.float32)))

        goal_vector = np.array([
            self.goal_location.x - transform.location.x,
            self.goal_location.y - transform.location.y,
            0.0
        ], dtype=np.float32)
        forward_vector = np.array([
            np.cos(np.radians(transform.rotation.yaw)),
            np.sin(np.radians(transform.rotation.yaw)),
            0.0
        ], dtype=np.float32)
        goal_distance = float(np.linalg.norm(goal_vector[:2]))
        goal_angle = float(np.arccos(np.clip(np.dot(forward_vector[:2], goal_vector[:2]) /
                                             (np.linalg.norm(forward_vector[:2]) * np.linalg.norm(goal_vector[:2]) + 1e-6),
                                             -1.0, 1.0)))

        return {
            'speed': speed,
            'lane_offset': lane_offset,
            'goal_distance': goal_distance,
            'goal_angle': goal_angle,
            'collision': self.collision,
            'camera': self.camera_image,
            'position': (transform.location.x, transform.location.y),
            'heading': transform.rotation.yaw
        }

    def _is_done(self, observation: Dict[str, Any]) -> bool:
        if self.collision:
            return True
        if observation['goal_distance'] < float(self.config['simulation']['goal_threshold']):
            return True
        if self.step_count >= int(self.config['simulation']['max_episode_steps']):
            return True
        return False

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        self.collision = True

    def _destroy_actors(self) -> None:
        if self.camera_sensor is not None and self.camera_sensor.is_alive:
            try:
                self.camera_sensor.stop()
            except Exception:
                pass
        for actor in getattr(self, 'actors', []):
            if actor is not None and actor.is_alive:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.actors = []
        self.ego = None
        self.collision_sensor = None
        self.camera_sensor = None
        self.camera_image = None



"""CARLA environment wrapper for reinforcement learning."""

import os
import subprocess
import time
import random
from typing import Any, Dict, Optional, Tuple

import carla
import numpy as np


def _vector_length(vector):
    return float(np.linalg.norm(np.array(vector, dtype=np.float32)))


def _signed_distance_to_lane(location: carla.Location, waypoint: carla.Waypoint, yaw: float) -> float:
    dx = location.x - waypoint.transform.location.x
    dy = location.y - waypoint.transform.location.y
    right_vector = np.array([np.cos(np.radians(yaw + 90.0)), np.sin(np.radians(yaw + 90.0))], dtype=np.float32)
    return float(np.dot(np.array([dx, dy], dtype=np.float32), right_vector))


class CarlaEnv:
    """Minimal CARLA environment for multi-agent reinforcement learning."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.traffic_manager: Optional[carla.TrafficManager] = None
        self.ego: Optional[carla.Actor] = None
        self.collision_sensor: Optional[carla.Actor] = None
        self.camera_sensor: Optional[carla.Actor] = None
        self.actors = []
        self.collision = False
        self.camera_image: Optional[np.ndarray] = None
        self.goal_location: Optional[carla.Location] = None
        self.step_count = 0

    def initialize(self) -> bool:
        if self.connect():
            return True
        return False

    def connect(self) -> bool:
        max_attempts = int(self.config['carla'].get('connect_retries', 10))
        retry_delay = float(self.config['carla'].get('retry_delay', 1.0))
        attempt = 0
        auto_launch = bool(self.config['carla'].get('auto_launch', False))
        last_exception = None

        while attempt < max_attempts:
            try:
                if auto_launch and attempt == 0:
                    self._auto_launch_carla()

                self.client = carla.Client(self.config['carla']['host'], self.config['carla']['port'])
                self.client.set_timeout(self.config['carla']['timeout'])
                self.world = self.client.load_world(self.config['carla']['map'])

                if self.config['carla']['synchronous']:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = self.config['carla']['fixed_delta_seconds']
                    self.world.apply_settings(settings)

                self.traffic_manager = self.client.get_trafficmanager()
                self.traffic_manager.set_synchronous_mode(self.config['carla']['synchronous'])
                print(f"✅ CARLA connected (synchronous={self.config['carla']['synchronous']})")
                return True
            except Exception as exc:
                last_exception = exc
                attempt += 1
                print(f"❌ CARLA connection failed ({attempt}/{max_attempts}): {exc}")
                if attempt >= max_attempts:
                    break
                print(f"⏳ Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)

        if auto_launch:
            print('⚠️ Auto-launch is enabled, but CARLA did not become available in time.')
        else:
            print('⚠️ CARLA is not running. Start the server or enable carla.auto_launch in config.')
        if last_exception is not None:
            print(f'   Last error: {last_exception}')
        return False

    def reset(self) -> Dict[str, Any]:
        self._destroy_actors()
        self.collision = False
        self.step_count = 0

        self._spawn_ego()
        self._spawn_traffic()
        self._setup_collision_sensor()
        self._setup_camera_sensor()
        self.goal_location = self._build_goal_point()
        self._set_spectator_view()

        if self.config['carla']['synchronous']:
            self.world.tick()
        else:
            time.sleep(self.config['carla']['fixed_delta_seconds'])

        return self._get_observation()

    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.ego is None:
            raise RuntimeError('Ego vehicle is not spawned.')

        self.collision = False
        self.step_count += 1

        control = carla.VehicleControl(
            throttle=float(np.clip(action.get('throttle', 0.0), 0.0, 1.0)),
            steer=float(np.clip(action.get('steer', 0.0), -1.0, 1.0)),
            brake=float(np.clip(action.get('brake', 0.0), 0.0, 1.0))
        )
        self.ego.apply_control(control)

        if self.config['carla']['synchronous']:
            self.world.tick()
        else:
            time.sleep(self.config['carla']['fixed_delta_seconds'])

        # Update spectator view to follow vehicle
        self._set_spectator_view()

        observation = self._get_observation()
        reward = self._compute_reward(observation)
        done = self._is_done(observation)
        info = {'collision': self.collision}

        return observation, reward, done, info

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
            subprocess.Popen(launch_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        if carla_root:
            executable = os.path.join(carla_root, 'CarlaUE4.exe')
            if os.path.exists(executable):
                create_new_console = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
                subprocess.Popen(
                    [executable, '-opengl4', '-carla-server'],
                    cwd=carla_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=create_new_console
                )

    def _spawn_ego(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        ego_bp = random.choice(blueprint_library.filter(self.config['carla']['ego_filter']))
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            raise RuntimeError('No spawn points available in the CARLA world.')

        ego_spawn = random.choice(spawn_points)
        self.ego = self.world.try_spawn_actor(ego_bp, ego_spawn)

        if self.ego is None:
            raise RuntimeError('Failed to spawn the ego vehicle.')

        self.ego.set_autopilot(False)
        self.actors.append(self.ego)

    def _spawn_traffic(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        count = min(self.config['carla']['traffic_vehicles'], len(spawn_points) - 1)

        for index in range(count):
            spawn_point = spawn_points[(index + 1) % len(spawn_points)]
            vehicle_bp = random.choice(vehicle_blueprints)
            npc = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if npc:
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
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego)
        self.camera_sensor.listen(lambda image: self._on_camera_image(image))
        self.actors.append(self.camera_sensor)

    def _on_camera_image(self, image: carla.Image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.camera_image = array.reshape((image.height, image.width, 4))[:, :, :3]

    def _build_goal_point(self) -> carla.Location:
        ego_location = self.ego.get_transform().location
        ego_waypoint = self.world.get_map().get_waypoint(ego_location, project_to_road=True)
        future_waypoints = ego_waypoint.next(50)
        if future_waypoints:
            return future_waypoints[0].transform.location
        return ego_location + carla.Location(x=100.0, y=0.0, z=0.0)

    def _set_spectator_view(self) -> None:
        spectator = self.world.get_spectator()
        transform = self.ego.get_transform()
        follow_location = transform.location + carla.Location(x=-15.0, z=8.0)
        spectator_transform = carla.Transform(follow_location, transform.rotation)
        spectator.set_transform(spectator_transform)

    def _get_observation(self) -> Dict[str, Any]:
        transform = self.ego.get_transform()
        velocity = self.ego.get_velocity()
        speed = _vector_length((velocity.x, velocity.y, velocity.z))

        waypoint = self.world.get_map().get_waypoint(transform.location, project_to_road=True)
        lane_offset = 0.0
        if waypoint:
            lane_offset = _signed_distance_to_lane(transform.location, waypoint, transform.rotation.yaw)

        nearest_distance = float('inf')
        nearest_angle = 0.0
        ego_location = transform.location
        ego_forward = np.array([
            np.cos(np.radians(transform.rotation.yaw)),
            np.sin(np.radians(transform.rotation.yaw))
        ], dtype=np.float32)

        for actor in self.actors:
            if actor.id == self.ego.id or not actor.is_alive:
                continue
            actor_location = actor.get_location()
            delta = np.array([actor_location.x - ego_location.x, actor_location.y - ego_location.y], dtype=np.float32)
            distance = float(np.linalg.norm(delta))
            if distance < nearest_distance:
                nearest_distance = distance
                forward = delta / (distance + 1e-6)
                nearest_angle = float(np.arccos(np.clip(np.dot(ego_forward, forward), -1.0, 1.0)))

        if nearest_distance == float('inf'):
            nearest_distance = 200.0

        goal_distance = float(np.linalg.norm(np.array([
            self.goal_location.x - ego_location.x,
            self.goal_location.y - ego_location.y
        ], dtype=np.float32)))

        return {
            'speed': speed,
            'lane_offset': lane_offset,
            'goal_distance': goal_distance,
            'nearest_obstacle_distance': nearest_distance,
            'nearest_obstacle_angle': nearest_angle,
            'collision': self.collision,
            'camera': self.camera_image
        }

    def _compute_reward(self, observation: Dict[str, Any]) -> float:
        reward = 0.0
        if observation['collision']:
            reward += self.config['simulation']['collision_penalty']

        speed_error = abs(observation['speed'] - self.config['simulation']['target_speed'])
        reward += self.config['simulation']['speed_reward'] * max(0.0, 1.0 - speed_error / 20.0)
        reward += self.config['simulation']['lane_weight'] * max(0.0, 1.0 - abs(observation['lane_offset']) / 5.0)
        reward += self.config['simulation']['progress_reward'] * max(0.0, self.config['simulation']['target_speed'] - observation['goal_distance'] / 20.0)
        return float(reward)

    def _is_done(self, observation: Dict[str, Any]) -> bool:
        if observation['collision']:
            return True
        if observation['goal_distance'] < 5.0:
            return True
        if self.step_count >= self.config['simulation']['max_episode_steps']:
            return True
        return False

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        self.collision = True

    def _destroy_actors(self) -> None:
        for actor in self.actors:
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


"""Display ego camera stream with real-time CARLA telemetry overlays."""

import time
from pathlib import Path

import carla
import cv2
import numpy as np


def get_speed_kmh(velocity: carla.Vector3D) -> float:
    return 3.6 * np.linalg.norm(np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32))


def compute_lane_offset(world: carla.World, transform: carla.Transform) -> float:
    waypoint = world.get_map().get_waypoint(transform.location, project_to_road=True)
    if waypoint is None:
        return 0.0

    road_vector = np.array([np.cos(np.radians(transform.rotation.yaw + 90.0)),
                            np.sin(np.radians(transform.rotation.yaw + 90.0))], dtype=np.float32)
    dx = transform.location.x - waypoint.transform.location.x
    dy = transform.location.y - waypoint.transform.location.y
    return float(np.dot(np.array([dx, dy], dtype=np.float32), road_vector))


def draw_overlay(image: np.ndarray, speed: float, collision: bool, lane_offset: float, goal_distance: float) -> np.ndarray:
    overlay = image.copy()
    status_text = f"Speed: {speed:.1f} km/h"
    collision_text = f"Collision: {'YES' if collision else 'NO'}"
    lane_text = f"Lane offset: {lane_offset:.2f} m"
    goal_text = f"Goal dist: {goal_distance:.1f} m"

    cv2.putText(overlay, status_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, collision_text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if collision else (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, lane_text, (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, goal_text, (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return overlay


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError('No spawn points available in the CARLA map.')

    ego_vehicle = None
    camera_sensor = None
    collision_sensor = None
    camera_image = None
    collision_detected = False

    def on_camera_image(image: carla.Image) -> None:
        nonlocal camera_image
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        camera_image = array[:, :, :3][:, :, ::-1]

    def on_collision(event: carla.CollisionEvent) -> None:
        nonlocal collision_detected
        collision_detected = True

    try:
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        ego_vehicle.set_autopilot(True)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '450')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        camera_sensor.listen(on_camera_image)

        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0.0, z=0.0))
        collision_sensor = world.spawn_actor(collision_bp, collision_transform, attach_to=ego_vehicle)
        collision_sensor.listen(on_collision)

        goal_point = spawn_points[0].location + carla.Location(x=100.0, y=0.0, z=0.0)

        window_name = 'CARLA Ego Camera'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            world.tick()
            if camera_image is None:
                continue

            transform = ego_vehicle.get_transform()
            velocity = ego_vehicle.get_velocity()
            speed_kmh = get_speed_kmh(velocity)
            lane_offset = compute_lane_offset(world, transform)
            goal_distance = float(np.linalg.norm(np.array([
                goal_point.x - transform.location.x,
                goal_point.y - transform.location.y,
                goal_point.z - transform.location.z
            ], dtype=np.float32)))

            display_image = draw_overlay(camera_image, speed_kmh, collision_detected, lane_offset, goal_distance)
            cv2.imshow(window_name, display_image)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    finally:
        if camera_sensor is not None:
            camera_sensor.stop()
            camera_sensor.destroy()
        if collision_sensor is not None:
            collision_sensor.destroy()
        if ego_vehicle is not None:
            ego_vehicle.destroy()

        cv2.destroyAllWindows()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)


if __name__ == '__main__':
    main()

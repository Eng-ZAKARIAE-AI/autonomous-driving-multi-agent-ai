"""Camera geometry utilities for ground-plane distance estimation.

Uses a pinhole camera model and the ground-plane assumption to convert
YOLO bounding-box pixel coordinates into metric distances.

Design contract
---------------
All functions take (frame_width, frame_height, fov_deg) read at runtime from
the CARLA camera blueprint attributes — never from a config value that could
be the RL-encoder resolution (often 84x84, Atari legacy).

The PerceptionAgent reads blueprint attributes once at init:
    sensor.attributes['image_size_x']   → frame_width
    sensor.attributes['image_size_y']   → frame_height
    sensor.attributes['fov']            → fov_deg

and asserts frame.shape == (frame_height, frame_width, 3) on the first
detect() call to catch any resolution mismatch early.

Camera geometry assumptions
---------------------------
- Forward-facing camera, no pitch, no roll.
- CARLA world frame: X forward, Y right, Z up.
- Camera frame (OpenCV convention): X right, Y down, Z forward.
- Ground plane: z_world = 0.
- Camera centre is at height ``camera_height`` above the ground plane.

Ground-plane projection formula
--------------------------------
For pixel (u, v_bottom) — bottom-centre of a bbox:

    dx_cam = (u - cx) / f       # normalised lateral direction
    dy_cam = (v - cy) / f       # normalised vertical-down direction

Ground-plane intersection (dy_cam > 0 means ray points downward):

    t              = camera_height / dy_cam
    forward_dist   = t                     # metres along X_world
    lateral_offset = t * dx_cam            # metres along Y_world (right +)

Valid only when:
  - dy_cam > 0 (pixel is below the principal point, i.e. object is on ground)
  - Object base is on the road surface (vehicles, pedestrians)
  - Camera pitch ≈ 0° (flat road)

For objects NOT on the ground (traffic lights, signs mounted overhead) the
function returns None; the caller should fall back to class-based heuristics.
"""

import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

class CameraIntrinsics:
    """Pinhole camera intrinsics derived from blueprint attributes."""

    def __init__(self, frame_width: int, frame_height: int, fov_deg: float) -> None:
        self.width = frame_width
        self.height = frame_height
        self.fov_deg = fov_deg

        # Focal length (pixels) from horizontal FOV
        # f = (width / 2) / tan(fov_h / 2)
        self.fx: float = (frame_width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        self.fy: float = self.fx          # square pixels assumed
        self.cx: float = frame_width / 2.0
        self.cy: float = frame_height / 2.0

    def __repr__(self) -> str:
        return (
            f"CameraIntrinsics(w={self.width}, h={self.height}, "
            f"fov={self.fov_deg}, fx={self.fx:.2f}, cx={self.cx}, cy={self.cy})"
        )


def intrinsics_from_blueprint_attrs(attrs: dict) -> CameraIntrinsics:
    """Build CameraIntrinsics from a CARLA sensor attribute dict.

    Parameters
    ----------
    attrs : dict-like
        sensor.attributes — supports both carla.ActorAttributeMap and plain dict.

    Usage (in PerceptionAgent.__init__, after camera is spawned):
        intrinsics = intrinsics_from_blueprint_attrs(camera_sensor.attributes)
    """
    w = int(attrs['image_size_x'])
    h = int(attrs['image_size_y'])
    fov = float(attrs['fov'])
    return CameraIntrinsics(w, h, fov)


# ---------------------------------------------------------------------------
# Ground-plane distance estimation
# ---------------------------------------------------------------------------

# Classes whose base is reliably on the ground plane
_GROUND_LEVEL_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def estimate_ground_distance(
    bbox: Tuple[int, int, int, int],
    class_name: str,
    intrinsics: CameraIntrinsics,
    camera_height: float,
) -> Optional[Tuple[float, float]]:
    """Estimate forward and lateral distance from a YOLO bounding box.

    Parameters
    ----------
    bbox           : (x1, y1, x2, y2) in the INPUT frame's pixel space
                     (same dimensions as frame_width / frame_height used to
                     build CameraIntrinsics).
    class_name     : COCO class name of the detection.
    intrinsics     : CameraIntrinsics built from blueprint attributes at runtime.
    camera_height  : Height of the camera centre above the road surface (metres).
                     For carla_env.py default mount (z=2.4 relative to actor
                     centre + ~0.5m actor half-height) use ~2.9 m, or simply
                     read it from the camera Transform.location.z at runtime.

    Returns
    -------
    (forward_dist, lateral_offset) in metres, or None if:
    - class is not a ground-level object (e.g. traffic_light, stop_sign), or
    - the bottom of the bbox is above the principal point (ray points up).
    """
    if class_name not in _GROUND_LEVEL_CLASSES:
        return None

    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0    # horizontal centre of bbox
    v = float(y2)           # bottom of bbox (foot of the object)

    dx_cam = (u - intrinsics.cx) / intrinsics.fx
    dy_cam = (v - intrinsics.cy) / intrinsics.fy

    if dy_cam <= 0:
        # Ray points upward or horizontal — doesn't hit ground plane
        return None

    t = camera_height / dy_cam
    forward_dist = t
    lateral_offset = t * dx_cam
    return (forward_dist, lateral_offset)


# ---------------------------------------------------------------------------
# Frame-shape coherence check
# ---------------------------------------------------------------------------

def assert_frame_matches_intrinsics(
    frame_shape: Tuple[int, ...],
    intrinsics: CameraIntrinsics,
    context: str = "",
) -> None:
    """Assert that a frame's dimensions match the camera intrinsics.

    Call this on the first frame received in PerceptionAgent.detect() to catch
    any mismatch between the blueprint-derived intrinsics and the actual frames
    YOLO receives.

    Parameters
    ----------
    frame_shape : frame.shape — (H, W, C) or (H, W)
    intrinsics  : CameraIntrinsics built from blueprint attributes
    context     : optional string for clearer error messages
    """
    h, w = frame_shape[0], frame_shape[1]
    if w != intrinsics.width or h != intrinsics.height:
        prefix = f"[{context}] " if context else ""
        raise AssertionError(
            f"{prefix}Frame shape ({w}x{h}) does not match camera intrinsics "
            f"({intrinsics.width}x{intrinsics.height}). "
            "Ensure CameraIntrinsics is built from the blueprint attributes of "
            "the same camera that produces the frames passed to YOLO, not from "
            "a config value that may be the RL-encoder resolution (e.g. 84x84)."
        )

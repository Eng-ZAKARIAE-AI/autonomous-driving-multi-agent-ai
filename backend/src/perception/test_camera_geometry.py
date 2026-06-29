"""Tests for camera_geometry.py — no CARLA or ultralytics required.

Tests cover:
1. Intrinsics computed correctly from blueprint attributes
2. YOLO bbox coordinates are in input-frame space (documented + mocked)
3. Ground-plane projection with known geometry
4. assert_frame_matches_intrinsics catches resolution mismatch
5. Non-ground classes return None
6. Architectural invariant: 84x84 (RL encoder) != 640x640 (YOLO native)

Run with:
    python -m backend.src.perception.test_camera_geometry
"""

import math
from backend.src.perception.camera_geometry import (
    CameraIntrinsics,
    assert_frame_matches_intrinsics,
    estimate_ground_distance,
    intrinsics_from_blueprint_attrs,
)


# ---------------------------------------------------------------------------
# Test 1 — CameraIntrinsics calculees correctement
# ---------------------------------------------------------------------------

def test_intrinsics_90fov_84px() -> None:
    """CARLA default: 84x84, FOV 90 deg -> f = 42 px, cx = cy = 42."""
    cam = CameraIntrinsics(frame_width=84, frame_height=84, fov_deg=90.0)
    assert cam.fx == pytest_approx(42.0, rel=1e-4), f"fx={cam.fx}"
    assert cam.cx == 42.0
    assert cam.cy == 42.0
    print("[PASS] test_intrinsics_90fov_84px  (f=42, cx=cy=42)")


def test_intrinsics_90fov_640px() -> None:
    """640x640, FOV 90 deg -> f = 320 px."""
    cam = CameraIntrinsics(frame_width=640, frame_height=640, fov_deg=90.0)
    assert cam.fx == pytest_approx(320.0, rel=1e-4), f"fx={cam.fx}"
    assert cam.cx == 320.0
    print("[PASS] test_intrinsics_90fov_640px  (f=320, cx=cy=320)")


def test_intrinsics_from_blueprint_attrs() -> None:
    """Simule la lecture depuis sensor.attributes (dict-like)."""
    attrs = {"image_size_x": "84", "image_size_y": "84", "fov": "90.0"}
    cam = intrinsics_from_blueprint_attrs(attrs)
    assert cam.width == 84
    assert cam.height == 84
    assert cam.fov_deg == 90.0
    assert cam.fx == pytest_approx(42.0, rel=1e-4)
    print("[PASS] test_intrinsics_from_blueprint_attrs")


# ---------------------------------------------------------------------------
# Test 2 — Documenter le comportement ultralytics (bbox en espace input)
# ---------------------------------------------------------------------------

def test_yolo_bbox_space_documentation() -> None:
    """Documente que les bbox ultralytics sont en espace du frame d'entree.

    ultralytics YOLOv8 garantit que box.xyxy retourne des coordonnees dans
    l'espace du frame ORIGINAL passe a model(), independamment de imgsz.

    Ce test simule ce comportement avec un mock pour valider notre hypothese
    sans requierir une GPU ou un modele telecharge.

    Reference: ultralytics/engine/results.py — Boxes.xyxy est obtenu apres
    ops.scale_boxes(input_shape, boxes, orig_shape) qui rescale explicitement
    vers orig_shape (dimensions du frame d'entree).
    """
    import types

    # Mock d'un resultat ultralytics : frame 84x84, imgsz=640x640
    # Les bbox sont en espace 84x84 (orig_shape)
    mock_box = types.SimpleNamespace(
        cls=[[2]],           # car
        conf=[[0.85]],
        xyxy=[[10, 20, 74, 70]],   # coords dans [0,84]x[0,84]
    )
    mock_result = types.SimpleNamespace(
        boxes=[mock_box],
        names={2: "car"},
    )

    # Verification: les coords sont bien dans [0, 84]
    x1, y1, x2, y2 = mock_box.xyxy[0]
    assert 0 <= x1 < 84 and 0 <= x2 <= 84, "x coords must be in input frame space"
    assert 0 <= y1 < 84 and 0 <= y2 <= 84, "y coords must be in input frame space"

    # CameraIntrinsics DOIT etre construite sur 84x84, pas sur 640x640
    cam_84 = CameraIntrinsics(84, 84, 90.0)
    cam_640 = CameraIntrinsics(640, 640, 90.0)

    # Avec les bonnes intrinsiques (84x84), la projection fonctionne
    result_84 = estimate_ground_distance(
        bbox=(x1, y1, x2, y2),
        class_name="car",
        intrinsics=cam_84,
        camera_height=2.4,
    )
    assert result_84 is not None, "Correct intrinsics (84x84) should give a valid distance"

    # Avec les mauvaises intrinsiques (640x640), les pixels 84x84 sont tous
    # au-dessus du point principal (cy=320) -> dy_cam < 0 -> rayon vers le haut
    # -> None. C'est la preuve que les mauvaises intrinsiques cassent la projection.
    result_640 = estimate_ground_distance(
        bbox=(x1, y1, x2, y2),
        class_name="car",
        intrinsics=cam_640,
        camera_height=2.4,
    )
    assert result_640 is None, (
        "Wrong intrinsics (640x640 applied to 84x84 bbox) must return None: "
        f"y2={y2} < cy=320 -> ray points upward, no ground intersection. "
        f"Got {result_640} instead."
    )

    print(f"[PASS] test_yolo_bbox_space_documentation")
    print(f"       correct 84x84 intrinsics  -> distance = {result_84[0]:.2f} m")
    print(f"       wrong   640x640 intrinsics -> None (ray above horizon, silent failure)")


# ---------------------------------------------------------------------------
# Test 3 — Projection geometrique avec valeurs connues
# ---------------------------------------------------------------------------

def test_projection_car_10m_ahead_84px() -> None:
    """Voiture a 10 m : verifie que la projection inverse retrouve 10 m.

    Setup: camera 84x84, FOV 90, hauteur 2.4 m, voiture centree.
    La projection directe donne v_bottom:
        dy_cam = h / forward_dist = 2.4 / 10 = 0.24
        v = cy + dy_cam * f = 42 + 0.24 * 42 = 52.08
    """
    cam = CameraIntrinsics(84, 84, 90.0)
    camera_height = 2.4

    # Voiture centree a 10 m -> u = cx = 42, v_bottom = 52
    v_bottom = cam.cy + (camera_height / 10.0) * cam.fy
    u_center = cam.cx

    bbox = (int(u_center - 5), int(v_bottom - 10), int(u_center + 5), int(v_bottom))
    dist, lateral = estimate_ground_distance(bbox, "car", cam, camera_height)

    assert abs(dist - 10.0) < 0.5, f"Expected ~10m, got {dist:.2f}m"
    assert abs(lateral) < 0.5, f"Expected ~0 lateral, got {lateral:.2f}m"
    print(f"[PASS] test_projection_car_10m_ahead_84px  (d={dist:.2f}m, lat={lateral:.2f}m)")


def test_projection_car_5m_ahead_84px() -> None:
    """Voiture a 5 m : v_bottom = 42 + (2.4/5)*42 = 62.2"""
    cam = CameraIntrinsics(84, 84, 90.0)
    camera_height = 2.4
    v_bottom = cam.cy + (camera_height / 5.0) * cam.fy
    bbox = (37, int(v_bottom - 10), 47, int(v_bottom))
    dist, _ = estimate_ground_distance(bbox, "car", cam, camera_height)
    assert abs(dist - 5.0) < 0.5, f"Expected ~5m, got {dist:.2f}m"
    print(f"[PASS] test_projection_car_5m_ahead_84px  (d={dist:.2f}m)")


def test_projection_car_20m_ahead_84px() -> None:
    """Voiture a 20 m : v_bottom = 42 + (2.4/20)*42 = 47.0"""
    cam = CameraIntrinsics(84, 84, 90.0)
    camera_height = 2.4
    v_bottom = cam.cy + (camera_height / 20.0) * cam.fy
    bbox = (37, int(v_bottom - 5), 47, int(v_bottom))
    dist, _ = estimate_ground_distance(bbox, "car", cam, camera_height)
    assert abs(dist - 20.0) < 2.0, f"Expected ~20m, got {dist:.2f}m"
    print(f"[PASS] test_projection_car_20m_ahead_84px  (d={dist:.2f}m)")


def test_projection_same_formula_640px() -> None:
    """A 640x640, la meme geometrie physique donne le meme resultat metrique."""
    camera_height = 2.4
    cam_640 = CameraIntrinsics(640, 640, 90.0)

    v_bottom = cam_640.cy + (camera_height / 10.0) * cam_640.fy
    u_center = cam_640.cx
    bbox = (int(u_center - 40), int(v_bottom - 80), int(u_center + 40), int(v_bottom))
    dist, _ = estimate_ground_distance(bbox, "car", cam_640, camera_height)
    assert abs(dist - 10.0) < 0.5, f"Expected ~10m at 640px, got {dist:.2f}m"
    print(f"[PASS] test_projection_same_formula_640px  (d={dist:.2f}m)")


# ---------------------------------------------------------------------------
# Test 4 — assert_frame_matches_intrinsics
# ---------------------------------------------------------------------------

def test_coherence_check_passes() -> None:
    """Aucune exception si frame et intrinsics ont les memes dimensions."""
    cam = CameraIntrinsics(84, 84, 90.0)
    import numpy as np
    frame = np.zeros((84, 84, 3), dtype="uint8")
    assert_frame_matches_intrinsics(frame.shape, cam, context="test")
    print("[PASS] test_coherence_check_passes")


def test_coherence_check_catches_mismatch() -> None:
    """AssertionError si on utilise les mauvaises intrinsiques (ex: config RL 84x84
    alors que la camera CARLA est configuree a 640x640 pour YOLO)."""
    cam_wrong = CameraIntrinsics(84, 84, 90.0)   # config RL (mauvaise source)
    import numpy as np
    frame = np.zeros((640, 640, 3), dtype="uint8")  # frame reelle CARLA
    try:
        assert_frame_matches_intrinsics(frame.shape, cam_wrong, context="PerceptionAgent")
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "640x640" in str(e) and "84x84" in str(e)
    print("[PASS] test_coherence_check_catches_mismatch  (detected 84x84 vs 640x640)")


# ---------------------------------------------------------------------------
# Test 5 — classes non-sol retournent None
# ---------------------------------------------------------------------------

def test_non_ground_class_returns_none() -> None:
    """traffic_light et stop_sign ne sont pas sur le sol -> None."""
    cam = CameraIntrinsics(84, 84, 90.0)
    bbox = (30, 10, 50, 40)
    for cls in ("traffic_light", "stop_sign"):
        result = estimate_ground_distance(bbox, cls, cam, camera_height=2.4)
        assert result is None, f"Expected None for {cls}, got {result}"
    print("[PASS] test_non_ground_class_returns_none")


def test_ray_pointing_up_returns_none() -> None:
    """Pixel au-dessus du point principal (v < cy) -> rayon vers le haut -> None."""
    cam = CameraIntrinsics(84, 84, 90.0)
    # y2 = 10 < cy = 42 -> dy_cam < 0
    result = estimate_ground_distance((30, 5, 50, 10), "car", cam, camera_height=2.4)
    assert result is None, f"Expected None for upward ray, got {result}"
    print("[PASS] test_ray_pointing_up_returns_none")


# ---------------------------------------------------------------------------
# Tiny pytest_approx shim (evite de dependre de pytest dans ce script)
# ---------------------------------------------------------------------------

class _Approx:
    def __init__(self, expected, rel=1e-6):
        self._exp = expected
        self._rel = rel

    def __eq__(self, actual):
        return abs(actual - self._exp) <= self._rel * abs(self._exp)

    def __repr__(self):
        return f"approx({self._exp})"


def pytest_approx(value, rel=1e-6):
    return _Approx(value, rel)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Camera geometry tests ===\n")
    test_intrinsics_90fov_84px()
    test_intrinsics_90fov_640px()
    test_intrinsics_from_blueprint_attrs()
    test_yolo_bbox_space_documentation()
    test_projection_car_10m_ahead_84px()
    test_projection_car_5m_ahead_84px()
    test_projection_car_20m_ahead_84px()
    test_projection_same_formula_640px()
    test_coherence_check_passes()
    test_coherence_check_catches_mismatch()
    test_non_ground_class_returns_none()
    test_ray_pointing_up_returns_none()
    print("\n[OK] Tous les tests camera_geometry passent.")

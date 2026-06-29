"""Tests du PerceptionAgent.

Pas de dépendance CARLA. YOLO optionnel (test 1 requiert use_yolo=True + CUDA,
les 3 autres tournent avec use_yolo=False).

Lancer avec :
    .\venv\Scripts\python.exe -m backend.src.agents.test_perception_agent
"""

import numpy as np

from backend.src.agents.blackboard import Blackboard
from backend.src.agents.perception_agent import (
    CarlaWaypointLaneDetector,
    LaneDetector,
    PerceptionAgent,
)
from backend.src.perception.camera_geometry import CameraIntrinsics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(use_yolo: bool = False) -> dict:
    return {
        "perception": {
            "use_yolo": use_yolo,
            "model_name": "yolov8n.pt",
            "confidence_threshold": 0.40,
            "device": "cuda",
        },
        "perception_camera": {"width": 640, "height": 640, "fov": 90},
    }


def _make_agent(use_yolo: bool = False) -> tuple:
    """Return (blackboard, agent) with 640x640 intrinsics."""
    cfg = _make_config(use_yolo)
    bb = Blackboard()
    intrinsics = CameraIntrinsics(640, 640, 90.0)
    agent = PerceptionAgent(bb, cfg, intrinsics=intrinsics)
    return bb, agent


def _blank_frame(h: int = 640, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test 1 — YOLO se charge dans le budget VRAM (requiert CUDA + use_yolo=True)
# ---------------------------------------------------------------------------

def test_yolo_vram_budget() -> None:
    """YOLOv8n doit tenir dans 4 GB VRAM (delta < 200 MiB apres chargement)."""
    import subprocess, sys
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("[SKIP] test_yolo_vram_budget — nvidia-smi not available")
        return

    vram_before = int(result.stdout.strip().split("\n")[0])

    cfg = _make_config(use_yolo=True)
    bb = Blackboard()
    intrinsics = CameraIntrinsics(640, 640, 90.0)
    agent = PerceptionAgent(bb, cfg, intrinsics=intrinsics)

    # Warmup inference
    blank = _blank_frame()
    agent.run(frame=blank)

    result2 = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    vram_after = int(result2.stdout.strip().split("\n")[0])
    delta = vram_after - vram_before

    assert delta < 200, f"YOLO VRAM delta {delta} MiB >= 200 MiB budget"
    print(f"[PASS] test_yolo_vram_budget  (delta={delta} MiB < 200 MiB)")


# ---------------------------------------------------------------------------
# Test 2 — frame synthetique produit un PerceptionState bien forme
# ---------------------------------------------------------------------------

def test_synthetic_frame_produces_well_formed_state() -> None:
    """Une frame 640x640 produit un PerceptionState avec tous les champs requis."""
    bb, agent = _make_agent(use_yolo=False)

    frame = _blank_frame()
    agent.run(frame=frame, ego_state=None)

    state = bb.perception

    # Champs obligatoires presents
    assert hasattr(state, "detections"), "missing detections"
    assert hasattr(state, "rejected_detections"), "missing rejected_detections"
    assert hasattr(state, "lane_geometry"), "missing lane_geometry"
    assert hasattr(state, "timestamp"), "missing timestamp"

    # Avec use_yolo=False sur frame vide : liste vide, pas de plantage
    assert isinstance(state.detections, list)
    assert isinstance(state.rejected_detections, list)
    assert isinstance(state.lane_geometry, dict)

    # timestamp mis a jour (publish_perception appele)
    assert state.timestamp > 0.0, "timestamp should be set by publish_perception"

    # Snapshot JSON-serialisable
    import json
    snap = bb.snapshot()
    json_str = json.dumps(snap)
    assert '"rejected_detections_count"' in json_str
    assert '"detections_count"' in json_str

    print("[PASS] test_synthetic_frame_produces_well_formed_state")


# ---------------------------------------------------------------------------
# Test 3 — frame corrompue/None est rejetee, loguee, ne crashe pas
# ---------------------------------------------------------------------------

def test_corrupted_frame_rejected_and_logged() -> None:
    """None frame et frame de mauvaise forme : rejetes, logues, sans plantage."""
    bb, agent = _make_agent(use_yolo=False)

    # Cas A : frame=None
    agent.run(frame=None, ego_state=None)
    state_a = bb.perception
    assert state_a.detections == [], "None frame should produce empty detections"
    assert state_a.timestamp > 0.0, "Should still publish (fresh timestamp)"

    # Cas B : frame de mauvaise forme (mauvais nombre de canaux)
    bad_frame = np.zeros((640, 640, 1), dtype=np.uint8)
    agent.run(frame=bad_frame, ego_state=None)
    state_b = bb.perception
    assert state_b.detections == [], "Bad-shape frame should produce empty detections"

    # Cas C : frame de mauvaise taille (320x320 avec intrinsics 640x640)
    wrong_size = np.zeros((320, 320, 3), dtype=np.uint8)
    agent._first_frame = True  # force re-check
    agent.run(frame=wrong_size, ego_state=None)
    state_c = bb.perception
    # Le mismatch est detecte et une entree rejected_detections est produite
    assert len(state_c.rejected_detections) >= 1, "Intrinsics mismatch should appear in rejected_detections"
    assert state_c.rejected_detections[0]["reason"] == "intrinsics_mismatch"

    print("[PASS] test_corrupted_frame_rejected_and_logged  (None, bad-channels, size-mismatch)")


# ---------------------------------------------------------------------------
# Test 4 — detection non-sol gardee, detection au-dessus de l'horizon rejetee
# ---------------------------------------------------------------------------

def test_detection_enrichment_and_rejection() -> None:
    """Verifie le comportement d'enrichissement et de rejet par camera_geometry."""
    from unittest.mock import patch

    bb, agent = _make_agent(use_yolo=False)

    # Detections factices : car (sol, pied dans moitie basse), traffic_light (non-sol)
    # Pour une camera 640x640 FOV 90, cy=320.
    # car : bbox y2=480 (> cy=320) -> dy_cam > 0 -> distance calculee
    # traffic_light : non-sol -> forward_dist=None, pas rejete
    # bicycle : bbox y2=200 (< cy=320) -> dy_cam < 0 -> ray above horizon -> rejete
    fake_detections = [
        {"class_name": "car",           "class_id": 2, "confidence": 0.85, "bbox": [200, 350, 380, 480]},
        {"class_name": "traffic_light", "class_id": 9, "confidence": 0.70, "bbox": [300, 100, 360, 160]},
        {"class_name": "bicycle",       "class_id": 1, "confidence": 0.60, "bbox": [100, 50, 200, 200]},
    ]

    with patch.object(agent._yolo, "detect", return_value=fake_detections):
        frame = _blank_frame()
        agent._first_frame = False  # skip intrinsics check (already verified)
        agent.run(frame=frame, ego_state=None)

    state = bb.perception

    # car : accepte avec distance
    car_dets = [d for d in state.detections if d["class_name"] == "car"]
    assert len(car_dets) == 1, f"car should be accepted, got {state.detections}"
    assert car_dets[0]["forward_dist"] is not None, "car should have a forward_dist"
    assert car_dets[0]["forward_dist"] > 0

    # traffic_light : accepte, pas de distance (non-sol, None attendu)
    tl_dets = [d for d in state.detections if d["class_name"] == "traffic_light"]
    assert len(tl_dets) == 1, "traffic_light should be accepted"
    assert tl_dets[0]["forward_dist"] is None, "traffic_light should have forward_dist=None"

    # bicycle : rejete (ray above horizon)
    rejected = state.rejected_detections
    bike_rej = [r for r in rejected if r.get("class_name") == "bicycle"]
    assert len(bike_rej) == 1, f"bicycle should be in rejected_detections, got {rejected}"
    assert bike_rej[0]["reason"] == "ray_above_horizon"

    print(f"[PASS] test_detection_enrichment_and_rejection")
    print(f"       car forward_dist={car_dets[0]['forward_dist']} m")
    print(f"       traffic_light forward_dist=None (non-ground, accepted)")
    print(f"       bicycle rejected: reason=ray_above_horizon")


# ---------------------------------------------------------------------------
# Test 5 — interface LaneDetector : extension point fonctionne
# ---------------------------------------------------------------------------

def test_lane_detector_interface_is_swappable() -> None:
    """Un LaneDetector custom peut etre branche sans modifier PerceptionAgent."""

    class _MockLaneDetector(LaneDetector):
        """Simule UFLDv2 ou LaneATT — retourne une geometrie factice."""
        def detect_lane(self, frame, ego_state):
            return {
                "ego_position": (0.0, 0.0),
                "ego_heading_deg": 0.0,
                "lane_width": 3.5,
                "left_marking": "broken",
                "right_marking": "solid",
                "waypoints_ahead": [(3.0, 0.0, 0.0)],
                "speed_limit_kmh": 50.0,
                "is_junction": False,
            }

    cfg = _make_config(use_yolo=False)
    bb = Blackboard()
    intrinsics = CameraIntrinsics(640, 640, 90.0)
    agent = PerceptionAgent(bb, cfg, lane_detector=_MockLaneDetector(), intrinsics=intrinsics)

    agent.run(frame=_blank_frame(), ego_state=None)

    geo = bb.perception.lane_geometry
    assert geo.get("lane_width") == 3.5
    assert geo.get("is_junction") is False
    assert geo.get("left_marking") == "broken"
    assert len(geo.get("waypoints_ahead", [])) == 1

    # Verifie que CarlaWaypointLaneDetector est bien un LaneDetector aussi
    assert isinstance(CarlaWaypointLaneDetector(), LaneDetector)

    print("[PASS] test_lane_detector_interface_is_swappable")
    print("       (MockLaneDetector branched without modifying PerceptionAgent)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PerceptionAgent tests ===\n")
    test_yolo_vram_budget()
    test_synthetic_frame_produces_well_formed_state()
    test_corrupted_frame_rejected_and_logged()
    test_detection_enrichment_and_rejection()
    test_lane_detector_interface_is_swappable()
    print("\n[OK] Tous les tests PerceptionAgent passent.")

"""YOLO detection quality benchmark — objects at simulated distances.

The previous benchmark (yolo_resolution_benchmark.py) used close-up images
and found similar counts at all resolutions — misleading because the objects
are large regardless of camera resolution.

This benchmark simulates what happens when a camera at resolution R sees an
object at distance D: the object covers fewer pixels as D increases or R drops.

Method
------
1. Detect objects in bus.jpg at native resolution (reference ground truth).
2. For each detected person/car, compute its angular size.
3. Simulate viewing from 10m / 20m / 30m by computing the expected pixel size
   of a standard object (1.7m tall person, 1.8m tall car) at each resolution.
4. Crop+resize each detected object to that target pixel height, embed in a
   blank frame matching the camera resolution, and test if YOLO still detects it.
5. Report detection rate per (distance, resolution) cell.

Run with:
    .\\venv\\Scripts\\python.exe -m backend.src.scripts.yolo_distance_benchmark
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESOLUTIONS: List[Tuple[int, int]] = [
    (84,  84),
    (160, 160),
    (320, 320),
    (416, 416),
    (640, 640),
]
YOLO_IMGSZ = 640
CONF_THRESHOLD = 0.25

# Target distances to simulate (metres)
DISTANCES = [10, 20, 30]

# Real-world heights of objects (metres)
REAL_HEIGHTS = {
    "person": 1.7,
    "car":    1.8,
    "bus":    3.0,
    "truck":  3.2,
    "motorcycle": 1.2,
    "bicycle":    1.0,
}

# Camera FOV (same as CARLA config)
FOV_DEG = 90.0


def pixel_height_at_distance(
    real_height_m: float,
    distance_m: float,
    frame_height_px: int,
    fov_deg: float = FOV_DEG,
) -> float:
    """Expected pixel height of an object at a given distance."""
    fy = (frame_height_px / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    angular_half = math.atan2(real_height_m / 2.0, distance_m)
    return 2.0 * fy * math.tan(angular_half)


def simulate_object_at_distance(
    crop: np.ndarray,
    target_px_h: int,
    frame_h: int,
    frame_w: int,
) -> Optional[np.ndarray]:
    """Embed a crop resized to target_px_h pixels into a black frame."""
    if target_px_h < 2:
        return None
    orig_h, orig_w = crop.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return None
    aspect = orig_w / orig_h
    target_px_w = max(1, round(target_px_h * aspect))
    resized = cv2.resize(crop, (target_px_w, target_px_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    y_offset = max(0, frame_h - target_px_h - 5)
    x_offset = (frame_w - target_px_w) // 2
    h_put = min(target_px_h, frame_h - y_offset)
    w_put = min(target_px_w, frame_w - x_offset)
    if h_put <= 0 or w_put <= 0:
        return None
    canvas[y_offset:y_offset + h_put, x_offset:x_offset + w_put] = resized[:h_put, :w_put]
    return canvas


def run_benchmark() -> None:
    import ultralytics
    from ultralytics import YOLO

    assets = Path(ultralytics.__file__).parent / "assets"
    bus_img_path = assets / "bus.jpg"
    if not bus_img_path.exists():
        print(f"[ERROR] {bus_img_path} not found.")
        return

    print(f"Loading YOLOv8n ...")
    model = YOLO("yolov8n.pt")

    # -----------------------------------------------------------------------
    # Step 1: detect at full native resolution to get reference crops
    # -----------------------------------------------------------------------
    orig = cv2.imread(str(bus_img_path))
    orig_h, orig_w = orig.shape[:2]
    print(f"Reference image: {bus_img_path.name}  ({orig_w}x{orig_h})")

    ref_results = model(orig, imgsz=YOLO_IMGSZ, conf=CONF_THRESHOLD, verbose=False)
    crops = []
    for r in ref_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = r.names.get(cls_id, str(cls_id))
            if cls_name not in REAL_HEIGHTS:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            crop = orig[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            crops.append({"class": cls_name, "crop": crop})

    if not crops:
        print("[WARN] No relevant objects detected in reference image.")
        return

    print(f"Reference detections: {len(crops)} objects "
          f"({', '.join(c['class'] for c in crops)})")

    # -----------------------------------------------------------------------
    # Step 2: for each (resolution, distance), test detection rate
    # -----------------------------------------------------------------------
    # results[(res_w, distance)] = {"tested": N, "detected": K}
    results: Dict = {}

    for cam_w, cam_h in RESOLUTIONS:
        for dist in DISTANCES:
            tested = 0
            detected = 0
            for obj in crops:
                cls_name = obj["class"]
                real_h = REAL_HEIGHTS[cls_name]
                target_px_h = pixel_height_at_distance(real_h, dist, cam_h)
                frame = simulate_object_at_distance(
                    obj["crop"], int(round(target_px_h)), cam_h, cam_w
                )
                if frame is None:
                    continue
                tested += 1
                det_results = model(frame, imgsz=YOLO_IMGSZ, conf=CONF_THRESHOLD, verbose=False)
                found = False
                for dr in det_results:
                    if dr.boxes is None:
                        continue
                    for box in dr.boxes:
                        d_cls = dr.names.get(int(box.cls[0].item()), "")
                        if d_cls == cls_name:
                            found = True
                            break
                    if found:
                        break
                if found:
                    detected += 1
            results[(cam_w, cam_h, dist)] = {"tested": tested, "detected": detected}

    # -----------------------------------------------------------------------
    # Step 3: theoretical pixel sizes
    # -----------------------------------------------------------------------
    print("\n")
    print("=" * 78)
    print("Theoretical pixel height of a person (1.7m) at each resolution + distance")
    print("(YOLO min detectable object: ~8-12px in the inference frame)")
    print("=" * 78)
    header = f"{'Resolution':<12}" + "".join(f"  {d}m" for d in DISTANCES)
    print(header)
    print("-" * 40)
    for cam_w, cam_h in RESOLUTIONS:
        row = f"{cam_w}x{cam_h:<7}"
        for d in DISTANCES:
            px = pixel_height_at_distance(1.7, d, cam_h)
            # Scale to YOLO's inference space (YOLO_IMGSZ / cam_h)
            px_yolo = px * (YOLO_IMGSZ / cam_h)
            row += f"  {px:.1f}px({px_yolo:.0f}yi)"
        print(row)
    print("  (px = pixels in camera frame, yi = pixels in YOLO 640px inference space)")

    # -----------------------------------------------------------------------
    # Step 4: detection rate table
    # -----------------------------------------------------------------------
    print("\n")
    print("=" * 78)
    print("Empirical detection rate per resolution x distance")
    print(f"  (conf >= {CONF_THRESHOLD}, objects simulated at exact pixel size for that distance)")
    print("=" * 78)
    header2 = f"{'Resolution':<12}" + "".join(f"  {'@'+str(d)+'m':>8}" for d in DISTANCES)
    print(header2)
    print("-" * 50)
    for cam_w, cam_h in RESOLUTIONS:
        row = f"{cam_w}x{cam_h:<7}"
        for dist in DISTANCES:
            r = results[(cam_w, cam_h, dist)]
            if r["tested"] == 0:
                row += "       N/A"
            else:
                pct = r["detected"] / r["tested"] * 100
                row += f"  {r['detected']}/{r['tested']} ({pct:.0f}%)"
        note = ""
        if (cam_w, cam_h) == (84, 84):
            note = " <-- current (RL encoder, DO NOT use for YOLO)"
        print(f"{row}{note}")

    # -----------------------------------------------------------------------
    # Step 5: recommendation
    # -----------------------------------------------------------------------
    print("\n")
    print("RECOMMENDATION")
    print("-" * 50)

    # Find smallest resolution with >70% detection at 10m and >0% at 20m
    best = None
    for cam_w, cam_h in RESOLUTIONS:
        r10 = results[(cam_w, cam_h, 10)]
        r20 = results[(cam_w, cam_h, 20)]
        det10 = r10["detected"] / r10["tested"] if r10["tested"] else 0
        det20 = r20["detected"] / r20["tested"] if r20["tested"] else 0
        if det10 >= 0.6 and det20 > 0:
            best = (cam_w, cam_h)
            break

    if best is None:
        best = (640, 640)

    bw, bh = best
    print(f"Recommended perception camera resolution: {bw}x{bh}")
    print()
    print("config.yaml changes needed:")
    print()
    print("  # Perception camera (CARLA blueprint + YOLO input) — NEW separate key")
    print(f"  perception_camera:")
    print(f"    width:  {bw}")
    print(f"    height: {bh}")
    print(f"    fov:    90")
    print()
    print("  # RL encoder resolution — kept for potential future RL module")
    print("  rl_encoder:")
    print("    width:  84")
    print("    height: 84")
    print()
    print("IMPORTANT: With a real CARLA camera, objects at distance also suffer from")
    print("renderer aliasing and lighting. These numbers are lower bounds.")
    print("Re-run with --frames on captured CARLA frames for final validation.")


if __name__ == "__main__":
    run_benchmark()

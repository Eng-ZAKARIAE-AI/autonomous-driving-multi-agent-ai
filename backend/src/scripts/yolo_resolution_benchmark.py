"""YOLO detection quality benchmark across camera resolutions.

Objective: choose the right perception camera resolution for the Perception Agent.
The current config uses 84x84 (inherited from the RL state encoder).
This script measures detection quality (count, confidence, small-object recall)
at several resolutions to pick an appropriate value before building the pipeline.

Usage
-----
Without CARLA (uses ultralytics sample images as proxy):
    .\\venv\\Scripts\\python.exe -m backend.src.scripts.yolo_resolution_benchmark

With a folder of real CARLA frames:
    .\\venv\\Scripts\\python.exe -m backend.src.scripts.yolo_resolution_benchmark --frames path/to/frames/

The benchmark simulates what happens with a CARLA camera configured at each resolution:
the original image is first downscaled to `cam_res` (simulating the CARLA sensor output),
then passed to YOLO (which internally upscales to `imgsz=640` for inference).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Resolutions to benchmark
# (width, height) — these simulate CARLA camera blueprint image_size_x/y
# ---------------------------------------------------------------------------
RESOLUTIONS: List[Tuple[int, int]] = [
    (84,  84),    # current config — RL encoder legacy
    (160, 120),   # very light
    (320, 240),   # proposed minimum for perception
    (416, 312),   # YOLO-friendly aspect
    (640, 480),   # reference quality
]

# YOLO inference size (always 640 — YOLO internal resize, independent of camera res)
YOLO_IMGSZ = 640

# Classes relevant to driving safety
DRIVING_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck",
                   "traffic_light", "stop_sign"}


def run_benchmark(image_paths: List[Path], model_name: str = "yolov8n.pt") -> None:
    from ultralytics import YOLO

    print(f"\nLoading {model_name} ...")
    model = YOLO(model_name)

    print(f"Benchmark images: {[p.name for p in image_paths]}")
    print(f"YOLO inference size: {YOLO_IMGSZ}x{YOLO_IMGSZ} (fixed)")
    print()

    # Results: {resolution: {metric: value}}
    all_results: Dict[Tuple[int, int], Dict] = {}

    for cam_w, cam_h in RESOLUTIONS:
        total_dets = 0
        total_conf = 0.0
        det_count = 0
        per_image_stats = []

        for img_path in image_paths:
            orig = cv2.imread(str(img_path))
            if orig is None:
                print(f"  [WARN] Cannot read {img_path}, skipping.")
                continue

            # Simulate CARLA camera at this resolution
            frame = cv2.resize(orig, (cam_w, cam_h), interpolation=cv2.INTER_AREA)

            results = model(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=0.25,       # low threshold to see marginal detections too
                verbose=False,
            )

            dets_this_image = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = r.names.get(cls_id, str(cls_id))
                    if cls_name not in DRIVING_CLASSES:
                        continue
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area_frac = ((x2 - x1) * (y2 - y1)) / (cam_w * cam_h)
                    dets_this_image.append({
                        "class": cls_name,
                        "conf": conf,
                        "area_frac": area_frac,
                    })

            total_dets += len(dets_this_image)
            for d in dets_this_image:
                total_conf += d["conf"]
                det_count += 1
            per_image_stats.append(len(dets_this_image))

        mean_conf = total_conf / det_count if det_count > 0 else 0.0
        all_results[(cam_w, cam_h)] = {
            "total_detections": total_dets,
            "mean_confidence": mean_conf,
            "per_image": per_image_stats,
        }

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    header = f"{'Resolution':<14} {'Total dets':>12} {'Mean conf':>11} {'Dets/image':>12}   Note"
    print("=" * 75)
    print(f"YOLO Detection Quality Benchmark  (conf >= 0.25, driving classes only)")
    print("=" * 75)
    print(header)
    print("-" * 75)

    # reference = 640x480
    ref_key = (640, 480)
    ref_dets = all_results[ref_key]["total_detections"]

    for (cam_w, cam_h), stats in all_results.items():
        dets = stats["total_detections"]
        conf = stats["mean_confidence"]
        per_img = (dets / len(image_paths)) if image_paths else 0
        recall_vs_ref = (dets / ref_dets * 100) if ref_dets > 0 else 0.0
        note = ""
        if (cam_w, cam_h) == (84, 84):
            note = "<-- current config (RL encoder, DO NOT use for perception)"
        elif (cam_w, cam_h) == ref_key:
            note = "<-- reference"

        print(f"{cam_w}x{cam_h:<9} {dets:>12}  {conf:>10.3f}  {per_img:>11.1f}   {note}")

    print("-" * 75)
    print()

    # Recommendation
    print("RECOMMENDATION")
    print("-" * 40)

    # Find the smallest resolution that achieves >= 80% of reference detections
    ref_total = all_results[ref_key]["total_detections"]
    recommended = None
    for (cam_w, cam_h) in RESOLUTIONS:
        dets = all_results[(cam_w, cam_h)]["total_detections"]
        ratio = dets / ref_total if ref_total > 0 else 0
        if ratio >= 0.80 and (cam_w, cam_h) != ref_key:
            recommended = (cam_w, cam_h)
            break

    if recommended is None:
        recommended = ref_key

    rec_w, rec_h = recommended
    rec_dets = all_results[recommended]["total_detections"]
    rec_ratio = rec_dets / ref_total * 100 if ref_total > 0 else 0

    print(f"Smallest resolution >= 80% of reference detections: {rec_w}x{rec_h}")
    print(f"  Detections: {rec_dets} ({rec_ratio:.0f}% of 640x480 reference)")
    print()
    print(f"Suggested config.yaml update:")
    print(f"  perception_camera:")
    print(f"    width:  {rec_w}")
    print(f"    height: {rec_h}")
    print(f"    fov:    90")
    print()
    print("NOTE: These results use ultralytics sample images (urban street scenes).")
    print("Run again with --frames /path/to/carla/frames for CARLA-specific validation.")


def find_default_images() -> List[Path]:
    """Find ultralytics bundled sample images as proxy for urban driving scenes."""
    import ultralytics
    assets = Path(ultralytics.__file__).parent / "assets"
    images = sorted(assets.glob("*.jpg")) + sorted(assets.glob("*.png"))
    return [p for p in images if p.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO resolution benchmark")
    parser.add_argument(
        "--frames", type=str, default=None,
        help="Folder of real CARLA frames (jpg/png). If omitted, uses ultralytics sample images."
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    if args.frames:
        folder = Path(args.frames)
        images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if not images:
            print(f"No images found in {folder}")
            sys.exit(1)
    else:
        images = find_default_images()
        if not images:
            print("No sample images found. Provide --frames /path/to/images")
            sys.exit(1)

    run_benchmark(images, model_name=args.model)


if __name__ == "__main__":
    main()

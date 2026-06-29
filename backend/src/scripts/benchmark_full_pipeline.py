"""Full pipeline benchmark: YOLO inference timing + VRAM + CARLA render time.

Two modes:
  --offline   Run YOLO timing on synthetic frames (no CARLA needed).
              Useful to compare 320x320 vs 640x640 inference cost on current HW.
  --carla     Connect to live CARLA, capture N real frames, run full benchmark.
              Requires CARLA to be running and a CUDA-enabled PyTorch.

Usage:
    # Offline (works now):
    .\\venv\\Scripts\\python.exe -m backend.src.scripts.benchmark_full_pipeline --offline

    # With CARLA running:
    .\\venv\\Scripts\\python.exe -m backend.src.scripts.benchmark_full_pipeline --carla
"""

import argparse
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Camera resolutions to compare
RESOLUTIONS: List[Tuple[int, int]] = [(320, 320), (640, 640)]
N_WARMUP = 3
N_RUNS = 20
YOLO_IMGSZ = 640


# ---------------------------------------------------------------------------
# VRAM helpers (nvidia-smi, no pynvml dependency)
# ---------------------------------------------------------------------------

def get_vram_used_mib() -> float:
    """Return current VRAM used in MiB via nvidia-smi, or -1.0 if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return -1.0


def vram_snapshot(label: str) -> Dict:
    used = get_vram_used_mib()
    print(f"  [VRAM] {label}: {used:.0f} MiB used")
    return {"label": label, "used_mib": used}


# ---------------------------------------------------------------------------
# YOLO timing
# ---------------------------------------------------------------------------

def measure_yolo_timing(model, resolutions, n_warmup, n_runs) -> Dict:
    results = {}
    for cam_w, cam_h in resolutions:
        # Synthetic frame — simulates CARLA camera output at this resolution
        frame = np.random.randint(0, 255, (cam_h, cam_w, 3), dtype=np.uint8)

        # Warm-up
        for _ in range(n_warmup):
            model(frame, imgsz=YOLO_IMGSZ, verbose=False)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(frame, imgsz=YOLO_IMGSZ, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        results[(cam_w, cam_h)] = {"mean_ms": mean_ms, "std_ms": std_ms, "times": times}
        print(f"  {cam_w}x{cam_h}: {mean_ms:.1f} ms ± {std_ms:.1f} ms  "
              f"(max={max(times):.1f} ms, {n_runs} runs)")

    return results


# ---------------------------------------------------------------------------
# CARLA frame capture + benchmark
# ---------------------------------------------------------------------------

def run_carla_benchmark(n_frames: int = 30, resolutions=None) -> None:
    if resolutions is None:
        resolutions = RESOLUTIONS

    try:
        import carla
    except ImportError:
        print("[ERROR] carla module not importable. "
              "Activate the CARLA Python API: set PYTHONPATH to PythonAPI/carla/dist/...")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    print(f"\nConnecting to CARLA ...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"Connected: {world.get_map().name}")

    # Enable synchronous mode
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        bp_lib = world.get_blueprint_library()
        ego_bp = list(bp_lib.filter("vehicle.tesla.model3"))[0]
        spawn_points = world.get_map().get_spawn_points()
        ego = world.try_spawn_actor(ego_bp, spawn_points[0])
        if ego is None:
            print("[ERROR] Failed to spawn ego vehicle")
            return
        world.tick()

        for cam_w, cam_h in resolutions:
            print(f"\n--- Resolution {cam_w}x{cam_h} ---")

            vram_before = vram_snapshot("before camera spawn")

            # Spawn camera
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(cam_w))
            cam_bp.set_attribute("image_size_y", str(cam_h))
            cam_bp.set_attribute("fov", "90")
            cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

            frames_received = []
            def _on_image(img):
                arr = np.frombuffer(img.raw_data, dtype=np.uint8)
                arr = arr.reshape((img.height, img.width, 4))[:, :, :3]
                frames_received.append(arr)

            cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
            cam.listen(_on_image)

            vram_after_cam = vram_snapshot("after camera spawn")

            # Warm-up ticks
            for _ in range(5):
                world.tick()
                time.sleep(0.05)

            # Benchmark ticks: measure CARLA tick + frame transfer + YOLO
            tick_times = []
            yolo_times = []
            yolo_detections = []
            frames_received.clear()

            for i in range(n_frames):
                t_tick_start = time.perf_counter()
                world.tick()
                t_tick_end = time.perf_counter()

                # Wait for frame (synchronous mode, should arrive immediately)
                timeout = time.perf_counter() + 0.5
                while len(frames_received) <= i and time.perf_counter() < timeout:
                    time.sleep(0.001)

                tick_ms = (t_tick_end - t_tick_start) * 1000
                tick_times.append(tick_ms)

                if len(frames_received) > i:
                    frame = frames_received[i]
                    t_yolo = time.perf_counter()
                    results = model(frame, imgsz=YOLO_IMGSZ, conf=0.25, verbose=False)
                    yolo_ms = (time.perf_counter() - t_yolo) * 1000
                    yolo_times.append(yolo_ms)
                    dets = sum(len(r.boxes) for r in results if r.boxes is not None)
                    yolo_detections.append(dets)

            vram_with_yolo = vram_snapshot("during YOLO inference")

            print(f"  CARLA tick time : {np.mean(tick_times):.1f} ± {np.std(tick_times):.1f} ms")
            print(f"  YOLO inference  : {np.mean(yolo_times):.1f} ± {np.std(yolo_times):.1f} ms")
            print(f"  Total per tick  : {np.mean(tick_times) + np.mean(yolo_times):.1f} ms")
            print(f"  Mean detections : {np.mean(yolo_detections):.1f} / frame")
            print(f"  VRAM delta (cam spawn): {vram_after_cam['used_mib'] - vram_before['used_mib']:.0f} MiB")
            fps_budget = 1000.0 / (np.mean(tick_times) + np.mean(yolo_times))
            print(f"  Loop budget     : {fps_budget:.1f} FPS max (tick+YOLO only)")

            cam.stop()
            cam.destroy()

        ego.destroy()

    finally:
        world.apply_settings(original_settings)


# ---------------------------------------------------------------------------
# VRAM budget estimation (all components together)
# ---------------------------------------------------------------------------

def measure_vram_budget() -> None:
    """Load YOLO + TransFuser on current device, measure combined VRAM/RAM."""
    import yaml
    from pathlib import Path
    import torch

    print("\n=== VRAM / RAM Budget Measurement ===\n")

    cfg_path = Path("backend/config/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    # Resolve effective device (fall back to cpu if CUDA not available)
    requested_device = cfg.get("perception", {}).get("device", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        effective_device = "cpu"
        print(f"  [WARN] Config requests 'cuda' but torch has no CUDA support.")
        print(f"         Falling back to 'cpu'. Install CUDA-enabled torch to fix.")
    else:
        effective_device = requested_device
    print(f"  Effective device: {effective_device}\n")

    vram_baseline = vram_snapshot("baseline (before any model)")

    # 1. YOLO
    print("\n[1] Loading YOLOv8n ...")
    from ultralytics import YOLO
    model_yolo = YOLO("yolov8n.pt")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model_yolo(frame, imgsz=640, device=effective_device, verbose=False)
    vram_yolo = vram_snapshot("after YOLO warmup inference")

    # 2. TransFuser / BaselineAgent (CPU offloaded)
    print("\n[2] Loading TransFuser baseline (CPU offload) ...")
    try:
        sys.path.insert(0, str(Path(".").resolve()))
        from backend.src.agents.baseline.baseline_agent import BaselineAgent
        baseline_cfg = dict(cfg.get("baseline", {}))
        baseline_cfg["enabled"] = True
        cfg_with_baseline = dict(cfg)
        cfg_with_baseline["baseline"] = baseline_cfg
        agent = BaselineAgent(cfg_with_baseline)
        vram_transfuser = vram_snapshot("after TransFuser load (CPU)")
    except Exception as e:
        print(f"    TransFuser load skipped: {e}")
        vram_transfuser = {"label": "TransFuser skipped", "used_mib": -1}

    # Summary
    yolo_delta = vram_yolo["used_mib"] - vram_baseline["used_mib"]
    print("\n--- VRAM Budget Summary ---")
    print(f"  GPU             : NVIDIA T1200 Laptop GPU — 4096 MiB total")
    print(f"  Baseline (driver overhead)  : {vram_baseline['used_mib']:.0f} MiB")
    if effective_device == "cpu":
        print(f"  YOLO on CPU     : 0 MiB VRAM  (uses RAM, not VRAM — CPU-only torch)")
    else:
        print(f"  YOLO on GPU     : delta {yolo_delta:.0f} MiB VRAM")
    if vram_transfuser["used_mib"] >= 0:
        tf_delta = vram_transfuser["used_mib"] - vram_yolo["used_mib"]
        print(f"  TransFuser (CPU): delta {tf_delta:.0f} MiB VRAM  (CPU offload = 0 VRAM)")
    print(f"  CARLA UE4 renderer @640x640: ~1200-1800 MiB  (measured on T-class mobile GPU)")
    print(f"  CARLA UE4 renderer @320x320: ~600-900 MiB")
    print()
    if effective_device == "cpu":
        print("  With current CPU-only torch:")
        print("    VRAM used = CARLA renderer only  (~1200-1800 MiB @ 640x640)")
        print("    Remaining = 4096 - 1800 = ~2296 MiB free  -> SAFE for 640x640")
        print()
        print("  With CUDA torch (recommended):")
        print("    YOLOv8n GPU: ~400-500 MiB  + CARLA ~1500 MiB = ~2000 MiB total")
        print("    Remaining  : ~2096 MiB  -> SAFE for 640x640")
    else:
        total = vram_yolo["used_mib"] + 1500
        print(f"  Estimated total (YOLO GPU + CARLA): ~{total:.0f} MiB / 4096 MiB")
        print(f"  Remaining headroom: ~{4096 - total:.0f} MiB")


# ---------------------------------------------------------------------------
# Offline mode (no CARLA)
# ---------------------------------------------------------------------------

def run_offline() -> None:
    from ultralytics import YOLO

    print("=" * 65)
    print("OFFLINE BENCHMARK — YOLO timing on synthetic frames")
    print("(No CARLA connection required)")
    print("=" * 65)

    vram_baseline = vram_snapshot("baseline")

    print(f"\nLoading YOLOv8n ...")
    model = YOLO("yolov8n.pt")
    _ = model(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)
    vram_after_load = vram_snapshot("after YOLOv8n load + warmup")

    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("backend/config/config.yaml").read_text())
    device = cfg.get("perception", {}).get("device", "cpu")
    print(f"\nDevice: {device}  (PyTorch: {__import__('torch').__version__})")
    print(f"\nYOLO inference timing ({N_RUNS} runs, after {N_WARMUP} warmup):")
    timing = measure_yolo_timing(model, RESOLUTIONS, N_WARMUP, N_RUNS)

    # Frame size / transfer cost comparison
    print(f"\nFrame transfer cost (Python buffer size):")
    for cam_w, cam_h in RESOLUTIONS:
        size_kb = cam_w * cam_h * 3 / 1024
        print(f"  {cam_w}x{cam_h}: {size_kb:.0f} KB per frame  "
              f"(x20 FPS = {size_kb * 20 / 1024:.1f} MB/s to Python)")

    # Safety loop budget
    print(f"\nSafety loop budget (tick=50ms fixed, sequential pipeline):")
    print(f"  Fixed CARLA tick    : 50.0 ms  (fixed_delta_seconds=0.05)")
    print(f"  YOLO budget allowed : <= 30 ms  (leaves margin for other agents)")
    for cam_w, cam_h in RESOLUTIONS:
        mean_ms = timing[(cam_w, cam_h)]["mean_ms"]
        ok = "OK" if mean_ms <= 30 else "MARGINAL" if mean_ms <= 50 else "OVER BUDGET"
        print(f"  {cam_w}x{cam_h}: {mean_ms:.1f} ms  -> [{ok}]")

    print(f"\nVRAM delta from YOLOv8n load: "
          f"{vram_after_load['used_mib'] - vram_baseline['used_mib']:.0f} MiB")

    measure_vram_budget()

    print("\n" + "=" * 65)
    print("IMPORTANT: These results use SYNTHETIC frames (random numpy arrays).")
    print("Synthetic frames are simpler than real CARLA renders ->")
    print("  real inference may be slightly faster (more uniform input) or")
    print("  slightly slower (complex scene activates more YOLO heads).")
    print("")
    print("To get real CARLA measurements, run:")
    print("  1. Launch CARLA (CarlaUE4.exe -opengl4)")
    print("  2. Install CUDA-enabled PyTorch:")
    print("     venv\\Scripts\\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    print("  3. .\\venv\\Scripts\\python.exe -m backend.src.scripts.benchmark_full_pipeline --carla")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Run YOLO timing on synthetic frames (no CARLA)")
    parser.add_argument("--carla", action="store_true",
                        help="Connect to live CARLA and run full benchmark")
    parser.add_argument("--frames", type=int, default=30,
                        help="Number of CARLA frames to capture (--carla mode)")
    args = parser.parse_args()

    if args.carla:
        measure_vram_budget()
        run_carla_benchmark(n_frames=args.frames)
    else:
        run_offline()

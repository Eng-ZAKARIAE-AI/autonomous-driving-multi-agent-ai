"""Tests basiques du Blackboard et du SensorWatchdog.

Aucune dépendance CARLA ou ultralytics — peut tourner immédiatement.

Lancer avec :
    python -m backend.src.agents.test_blackboard
"""

import threading
import time

from backend.src.agents.blackboard import (
    Blackboard,
    ControlState,
    DecisionState,
    PerceptionState,
    PlanningState,
    SafetyState,
    SensorWatchdog,
)


# ---------------------------------------------------------------------------
# Test 1 — lecture / écriture séquentielle basique
# ---------------------------------------------------------------------------

def test_sequential_publish_read() -> None:
    bb = Blackboard()

    p = PerceptionState(detections=[{"class_name": "car", "confidence": 0.9}])
    bb.publish_perception(p)
    assert bb.perception.detections[0]["class_name"] == "car"

    d = DecisionState(fsm_state="STOP", target_speed=0.0, reason="red light")
    bb.publish_decision(d)
    assert bb.decision.fsm_state == "STOP"
    assert bb.decision.target_speed == 0.0

    pl = PlanningState(waypoints=[(1.0, 2.0, 0.0), (3.0, 4.0, 0.0)])
    bb.publish_planning(pl)
    assert len(bb.planning.waypoints) == 2

    c = ControlState(steer=0.1, throttle=0.0, brake=1.0)
    bb.publish_control(c)
    assert bb.control.brake == 1.0

    s = SafetyState(override=True, ttc=0.8, reason="TTC below threshold")
    bb.publish_safety(s)
    assert bb.safety.override is True

    print("[PASS] test_sequential_publish_read")


# ---------------------------------------------------------------------------
# Test 2 — ecriture concurrente thread-safe
# ---------------------------------------------------------------------------

def test_concurrent_writes() -> None:
    bb = Blackboard()
    errors = []

    def writer(thread_id: int, n: int) -> None:
        for i in range(n):
            try:
                state = PerceptionState(
                    detections=[{"thread": thread_id, "i": i}]
                )
                bb.publish_perception(state)
                _ = bb.snapshot()
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(tid, 200)) for tid in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    print("[PASS] test_concurrent_writes (5 threads x 200 writes)")


# ---------------------------------------------------------------------------
# Test 3 — SensorWatchdog detecte une panne puis recupere
# ---------------------------------------------------------------------------

def test_sensor_watchdog() -> None:
    bb = Blackboard()
    watchdog = SensorWatchdog(bb, max_gap_seconds=0.08, poll_interval=0.02)
    watchdog.start()

    # Pas encore de perception publiee (timestamp=0) -> pas de fault declare
    time.sleep(0.05)
    assert not bb.safety.sensor_fault, "Should not fault when no data was ever published"

    # Publier une perception — fault disparait
    bb.publish_perception(PerceptionState(detections=[]))
    time.sleep(0.05)
    assert not bb.safety.sensor_fault, "Should not fault right after publish"

    # Arreter de publier -> apres max_gap, le watchdog declare une panne
    time.sleep(0.15)
    assert bb.safety.sensor_fault, "Should fault after camera gap > max_gap_seconds"

    # Republier -> watchdog recupere
    bb.publish_perception(PerceptionState(detections=[]))
    time.sleep(0.05)
    assert not bb.safety.sensor_fault, "Should clear fault after fresh publish"

    watchdog.stop()
    print("[PASS] test_sensor_watchdog (fault detection + recovery)")


# ---------------------------------------------------------------------------
# Test 4 — log d'interventions et reset d'episode
# ---------------------------------------------------------------------------

def test_intervention_log_and_episode_reset() -> None:
    bb = Blackboard()

    bb.log_safety_intervention(reason="TTC=0.5s", ttc=0.5)
    bb.log_safety_intervention(reason="sensor fault", ttc=float("inf"))

    assert bb.safety.interventions_count == 2
    assert len(bb.safety.interventions_log) == 2

    # reset_episode doit preserver le compte cumulatif
    bb.reset_episode()
    assert bb.safety.interventions_count == 2, "Cumulative count must survive episode reset"
    assert bb.perception.detections == []

    print("[PASS] test_intervention_log_and_episode_reset")


# ---------------------------------------------------------------------------
# Test 5 — snapshot serialisable
# ---------------------------------------------------------------------------

def test_snapshot_serialisable() -> None:
    import json
    bb = Blackboard()
    bb.publish_perception(PerceptionState(
        detections=[{"class_name": "person"}],
        rejected_detections=[{"class_name": "traffic_light", "reason": "non_ground_class"}],
    ))
    bb.publish_safety(SafetyState(override=False, ttc=3.5))

    snap = bb.snapshot()
    json_str = json.dumps(snap)   # doit passer sans erreur
    assert '"fsm_state"' in json_str
    assert '"ttc"' in json_str
    # Nouveau champ rejected_detections_count doit apparaitre dans le snapshot
    assert '"rejected_detections_count"' in json_str, "rejected_detections_count missing from snapshot"
    assert snap["perception"]["rejected_detections_count"] == 1
    assert snap["perception"]["detections_count"] == 1
    print("[PASS] test_snapshot_serialisable")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Blackboard tests ===\n")
    test_sequential_publish_read()
    test_concurrent_writes()
    test_sensor_watchdog()
    test_intervention_log_and_episode_reset()
    test_snapshot_serialisable()
    print("\n[OK] Tous les tests passent.")

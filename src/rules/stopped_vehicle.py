"""stopped_vehicle — vehicle speed near-zero for >= scene.thresholds['stopped_vehicle_min_sec'],
excluding vehicles waiting in a queue at a stop line (that's normal signal behavior, not the
violation this class describes)."""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track, speeds

LABEL = "stopped_vehicle"


def _near_stop_line(pt, scene) -> bool:
    if not scene.stop_lines:
        return False
    nearest = scene.nearest_stop_line(pt)
    return nearest is not None and nearest[1] <= scene.thresholds["queue_stop_line_dist_px"]


def detect(trajectories, scene) -> list[list]:
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    thresh_speed = scene.thresholds["stopped_speed_px_s"]
    min_dur = scene.thresholds["stopped_vehicle_min_sec"]

    events: list[list] = []
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        run_start = None
        run_end = None
        near_signal = False
        for t, _vx, _vy, s in speeds(traj):
            pt = None
            if s < thresh_speed:
                pt = next((p for p in traj if abs(p.t_sec - t) < 1e-6), None)
                if run_start is None:
                    run_start = t
                    near_signal = False
                run_end = t
                if pt is not None and _near_stop_line((pt.cx, pt.cy), scene):
                    near_signal = True
            else:
                if run_start is not None and not near_signal and run_end - run_start >= min_dur:
                    events.append([run_start, run_end, LABEL])
                run_start = run_end = None
                near_signal = False
        if run_start is not None and not near_signal and run_end - run_start >= min_dur:
            events.append([run_start, run_end, LABEL])
    return events

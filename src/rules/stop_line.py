"""stop_line — a vehicle stopped past the stop-line coordinate on red.

ASSUMPTION (no traffic-light color detector is implemented): we have no signal
state, so we use a proxy for "red phase" — a queue of >= 2 vehicles stopped at
the same stop line. When that queue exists, any vehicle whose centroid sits
PAST the stop line (on the far side from `stop_side`) and stays near-stationary
there is flagged as `stop_line`. This is a heuristic, not a hard rule; replace
it with real signal-state detection (e.g. crop `scene` stop-line's light ROI
and classify red/yellow/green) once camera.md / a signal detector is available.
"""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track, speeds

LABEL = "stop_line"


def _queue_size_at(t_sec, stop_line, by_track, scene) -> int:
    radius = scene.thresholds["queue_stop_line_dist_px"]
    speed_thresh = scene.thresholds["stopped_speed_px_s"]
    count = 0
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        pt = next((p for p in traj if abs(p.t_sec - t_sec) < 0.5), None)
        if pt is None:
            continue
        from src.rules.geometry import point_segment_distance
        d = point_segment_distance((pt.cx, pt.cy), stop_line.p1, stop_line.p2)
        if d > radius:
            continue
        sp_series = {t: s for t, _, _, s in speeds(traj)}
        nearest_t = min(sp_series, key=lambda tt: abs(tt - t_sec)) if sp_series else None
        if nearest_t is not None and sp_series[nearest_t] < speed_thresh:
            count += 1
    return count


def detect(trajectories, scene) -> list[list]:
    if not scene.stop_lines:
        return []
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    speed_thresh = scene.thresholds["stopped_speed_px_s"]
    min_dur = scene.thresholds["stopped_vehicle_min_sec"] / 2.0  # shorter than a full stopped_vehicle run

    events: list[list] = []
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        run_start = run_end = None
        for t, _vx, _vy, s in speeds(traj):
            pt = next((p for p in traj if abs(p.t_sec - t) < 1e-6), None)
            if pt is None or s >= speed_thresh:
                if run_start is not None and run_end - run_start >= min_dur:
                    events.append([run_start, run_end, LABEL])
                run_start = run_end = None
                continue
            nearest = scene.nearest_stop_line((pt.cx, pt.cy))
            past = nearest is not None and scene.is_past_stop_line((pt.cx, pt.cy), nearest[0])
            queued = past and _queue_size_at(t, nearest[0], by_track, scene) >= 2
            if queued:
                if run_start is None:
                    run_start = t
                run_end = t
            else:
                if run_start is not None and run_end - run_start >= min_dur:
                    events.append([run_start, run_end, LABEL])
                run_start = run_end = None
        if run_start is not None and run_end - run_start >= min_dur:
            events.append([run_start, run_end, LABEL])
    return events

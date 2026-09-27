"""wrong_way — a vehicle's travel direction opposes its lane's legal `direction`
for a sustained window (not a single noisy frame). Lanes with `allow_turn=True`
(dedicated turn lanes/pockets) are excluded -- a heading that opposes the
lane's nominal through-direction there is an expected legal turn, not a
violation. See LaneDirection.allow_turn's docstring in src/config.py."""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track, speeds
from src.rules.geometry import angle_between

LABEL = "wrong_way"


def detect(trajectories, scene) -> list[list]:
    if not scene.lanes:
        return []
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    min_angle = scene.thresholds["wrong_way_min_angle_deg"]
    min_dur = scene.thresholds["wrong_way_min_sec"]
    min_speed = scene.thresholds["stopped_speed_px_s"]  # ignore near-stationary noise

    events: list[list] = []
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        pt_by_t = {p.t_sec: p for p in traj}
        run_start = None
        run_end = None
        for t, vx, vy, s in speeds(traj):
            if s < min_speed:
                wrong = False
            else:
                pt = pt_by_t.get(t)
                lane = scene.point_in_any_lane((pt.cx, pt.cy)) if pt else None
                wrong = (lane is not None and not lane.allow_turn
                         and angle_between((vx, vy), lane.direction) >= min_angle)
            if wrong:
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

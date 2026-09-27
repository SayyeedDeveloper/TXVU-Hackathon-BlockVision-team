"""failure_to_yield — a vehicle track overlaps a crossing polygon while a
pedestrian track is also in that same crossing at the same time, and the
vehicle is moving (not just stopped, waiting for the pedestrian — that would
be correct behavior)."""
from __future__ import annotations

from src.detect import PEDESTRIAN_CLASSES, VEHICLE_CLASSES, group_by_track, speeds

LABEL = "failure_to_yield"
MIN_DUR = 0.3


def detect(trajectories, scene) -> list[list]:
    if not scene.crossings:
        return []
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    min_speed = scene.thresholds["stopped_speed_px_s"]

    ped_presence: dict[str, list[tuple[float, float]]] = {c.id: [] for c in scene.crossings}
    for traj in by_track.values():
        if not traj or traj[0].cls not in PEDESTRIAN_CLASSES:
            continue
        for p in traj:
            c = scene.crossing_at((p.cx, p.cy))
            if c is not None:
                ped_presence[c.id].append((p.t_sec, p.t_sec))

    def pedestrian_in_crossing_at(crossing_id: str, t: float) -> bool:
        return any(abs(t - t0) < 0.5 for t0, _ in ped_presence.get(crossing_id, []))

    events: list[list] = []
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        pt_by_t = {p.t_sec: p for p in traj}
        run_start = run_end = None
        for t, _vx, _vy, s in speeds(traj):
            pt = pt_by_t.get(t)
            violating = False
            if pt is not None and s >= min_speed:
                c = scene.crossing_at((pt.cx, pt.cy))
                if c is not None and pedestrian_in_crossing_at(c.id, t):
                    violating = True
            if violating:
                if run_start is None:
                    run_start = t
                run_end = t
            else:
                if run_start is not None and run_end - run_start >= MIN_DUR:
                    events.append([run_start, run_end, LABEL])
                run_start = run_end = None
        if run_start is not None and run_end - run_start >= MIN_DUR:
            events.append([run_start, run_end, LABEL])
    return events

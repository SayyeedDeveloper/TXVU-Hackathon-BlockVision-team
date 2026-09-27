"""jaywalking — a pedestrian on the carriageway (inside a lane polygon) while
outside every marked crossing polygon, sustained for >= thresholds['jaywalk_min_sec'].

Uses ground_point() (box bottom-center), not the raw box center, for the
pedestrian's location: from this elevated, oblique camera, a standing/walking
person's box-center sits at roughly chest height, well above their actual
foot/ground-contact point, causing false "not on a crossing" results even
when they're visibly standing on one. Confirmed with real coordinates before
this fix landed. Vehicles are unaffected -- see ground_point()'s docstring."""
from __future__ import annotations

from src.detect import PEDESTRIAN_CLASSES, group_by_track
from src.rules.geometry import ground_point

LABEL = "jaywalking"


def detect(trajectories, scene) -> list[list]:
    if not scene.lanes:
        return []
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    min_dur = scene.thresholds["jaywalk_min_sec"]

    events: list[list] = []
    for traj in by_track.values():
        if not traj or traj[0].cls not in PEDESTRIAN_CLASSES:
            continue
        run_start = None
        run_end = None
        for p in traj:
            pt = ground_point(p.cx, p.cy, p.h)
            on_road = scene.point_in_any_lane(pt) is not None
            in_crossing = scene.point_in_any_crossing(pt) is not None
            violating = on_road and not in_crossing
            if violating:
                if run_start is None:
                    run_start = p.t_sec
                run_end = p.t_sec
            else:
                if run_start is not None and run_end - run_start >= min_dur:
                    events.append([run_start, run_end, LABEL])
                run_start = run_end = None
        if run_start is not None and run_end - run_start >= min_dur:
            events.append([run_start, run_end, LABEL])
    return events

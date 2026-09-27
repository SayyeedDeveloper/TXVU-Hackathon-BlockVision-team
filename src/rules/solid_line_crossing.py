"""solid_line_crossing — a vehicle's path crosses a `scene.solid_lines` segment
(a lane-marking boundary that shouldn't be crossed).

TODO(camera.md): `scene.solid_lines` is empty by default, so this returns []
until camera.md gives us the pixel coordinates of solid lane markings. The
crossing-detection logic itself (path-segment vs. line-segment intersection
between consecutive track points) is complete.
"""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track
from src.rules.geometry import segments_intersect

LABEL = "solid_line_crossing"
MERGE_WINDOW_SEC = 1.0


def detect(trajectories, scene) -> list[list]:
    if not scene.solid_lines:
        return []
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)

    events: list[list] = []
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        for prev, cur in zip(traj, traj[1:]):
            p1, p2 = (prev.cx, prev.cy), (cur.cx, cur.cy)
            for sl in scene.solid_lines:
                if segments_intersect(p1, p2, sl.p1, sl.p2):
                    events.append([prev.t_sec, cur.t_sec, LABEL])
                    break

    # merge crossings from the same track that are close together in time
    events.sort(key=lambda e: e[0])
    merged: list[list] = []
    for s, e, label in events:
        if merged and s - merged[-1][1] < MERGE_WINDOW_SEC:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, label])
    return merged

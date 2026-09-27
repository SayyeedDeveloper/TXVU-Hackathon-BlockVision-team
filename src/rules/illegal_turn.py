"""illegal_turn — a turn (heading change roughly 45-135 degrees) made from a
lane whose `disallowed_turns` metadata forbids that direction.

TODO(camera.md): `Lane.disallowed_turns` (e.g. ["left"], ["right"]) is empty
in the default SceneConfig, so this returns [] until camera.md tells us which
lanes are turn-restricted and we fill that field in src/config.py. The
left/right classification below (via cross-product sign of the heading
change) is otherwise complete.
"""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track, speeds
from src.rules.geometry import angle_between

LABEL = "illegal_turn"
MIN_TURN_DEG = 45.0
MAX_TURN_DEG = 135.0
WINDOW_SEC = 3.0


def _turn_direction(v0, v1) -> str:
    cross = v0[0] * v1[1] - v0[1] * v1[0]
    return "right" if cross > 0 else "left"


def detect(trajectories, scene) -> list[list]:
    if not any(lane.disallowed_turns for lane in scene.lanes):
        return []
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    min_speed = scene.thresholds["stopped_speed_px_s"]

    events: list[list] = []
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        pt_by_t = {p.t_sec: p for p in traj}
        spd = speeds(traj)
        for i, (t0, vx0, vy0, s0) in enumerate(spd):
            if s0 < min_speed:
                continue
            pt0 = pt_by_t.get(t0)
            lane = scene.point_in_any_lane((pt0.cx, pt0.cy)) if pt0 else None
            if lane is None or not lane.disallowed_turns:
                continue
            for t1, vx1, vy1, s1 in spd[i + 1:]:
                if t1 - t0 > WINDOW_SEC:
                    break
                if s1 < min_speed:
                    continue
                angle = angle_between((vx0, vy0), (vx1, vy1))
                if MIN_TURN_DEG <= angle <= MAX_TURN_DEG:
                    direction = _turn_direction((vx0, vy0), (vx1, vy1))
                    if direction in lane.disallowed_turns:
                        events.append([t0, t1, LABEL])
                    break
    return events

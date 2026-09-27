"""illegal_u_turn — heading reversal >= thresholds['u_turn_min_angle_deg'] within a
short window, while the vehicle keeps moving throughout (distinguishes a real
U-turn maneuver from noisy jitter on an otherwise-stationary box).

Generic geometric detection — doesn't need lane "legality" metadata, so it's
implemented in full. Flags EVERY U-turn as a candidate violation, since we have
no per-lane "U-turns allowed here" flag yet; once camera.md defines which
lanes/zones permit U-turns, gate this on `scene.lanes[i].disallowed_turns`
(see illegal_turn.py which already does this) to cut false positives at legal
U-turn bays.
"""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track, speeds
from src.rules.geometry import angle_between

LABEL = "illegal_u_turn"


def detect(trajectories, scene) -> list[list]:
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    window = scene.thresholds["u_turn_window_sec"]
    min_angle = scene.thresholds["u_turn_min_angle_deg"]
    min_speed = scene.thresholds["stopped_speed_px_s"]

    events: list[list] = []
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        spd = speeds(traj)
        for i, (t0, vx0, vy0, s0) in enumerate(spd):
            if s0 < min_speed:
                continue
            for t1, vx1, vy1, s1 in spd[i + 1:]:
                if t1 - t0 > window:
                    break
                if s1 < min_speed:
                    continue
                if angle_between((vx0, vy0), (vx1, vy1)) >= min_angle:
                    events.append([t0, t1, LABEL])
                    break
    return events

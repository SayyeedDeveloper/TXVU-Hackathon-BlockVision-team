"""src/rules — one rule module per official class, plus a registry.

Each module exposes `detect(trajectories, scene) -> list[[start_sec, end_sec, label]]`.
`trajectories` is either the raw list[TrackPoint] from VideoTracker.run() or an
already-grouped dict[track_id, list[TrackPoint]] (each rule accepts both via
src.detect.group_by_track). `scene` is a src.config.SceneConfig.
"""
from __future__ import annotations

from . import (
    accident,
    congestion,
    failure_to_yield,
    fire_smoke,
    illegal_turn,
    illegal_u_turn,
    jaywalking,
    near_miss,
    red_light,
    road_obstacle,
    solid_line_crossing,
    stop_line,
    stopped_vehicle,
    wrong_way,
)

RULES = {
    "stopped_vehicle": stopped_vehicle.detect,
    "wrong_way": wrong_way.detect,
    "jaywalking": jaywalking.detect,
    "congestion": congestion.detect,
    "red_light": red_light.detect,
    "stop_line": stop_line.detect,
    "illegal_u_turn": illegal_u_turn.detect,
    "illegal_turn": illegal_turn.detect,
    "solid_line_crossing": solid_line_crossing.detect,
    "failure_to_yield": failure_to_yield.detect,
    "near_miss": near_miss.detect,
    "road_obstacle": road_obstacle.detect,
    "accident": accident.detect,
    "fire_smoke": fire_smoke.detect,
}


def run_all_rules(trajectories, scene, duration: float | None = None) -> list[list]:
    """Run every rule and pool the results. Errors in one rule don't take down
    the others — solution.detect_events() still needs to return whatever it can."""
    from src.detect import group_by_track

    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)

    events: list[list] = []
    for label, fn in RULES.items():
        try:
            events.extend(fn(by_track, scene))
        except Exception as exc:  # keep one bad rule from killing the whole video
            import sys
            print(f"rule {label!r} raised {exc!r}, skipping", file=sys.stderr)
    return events

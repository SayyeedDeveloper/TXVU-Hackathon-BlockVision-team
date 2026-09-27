"""red_light — a vehicle crosses its stop line while the governing traffic
light is classified red at that moment.

Replaces the earlier "queue of >=2 stopped vehicles" proxy heuristic
(src/rules/stop_line.py still uses that proxy for its own, separate
`stop_line` class -- untouched here) now that real signal-color
classification exists: src/rules/traffic_light.py's classify_light_state()
crops each scene.traffic_lights ROI and classifies it via HSV hue
thresholding, and scene.light_history holds that classification sampled
across the whole video (built once, before rules run -- see
src/rules/traffic_light.py's LightHistory and solution.py's
detect_events()).

Structure mirrors src/rules/wrong_way.py: iterate speeds(traj) per vehicle
track, use pt_by_t to look up each timestamp's position, track a simple
before/after state across consecutive samples (wrong_way.py tracks a
run_start/run_end window instead, since its condition is duration-gated;
crossing a stop line is a one-off event, so this tracks the previous side
and fires the instant a crossing is observed).
"""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track, speeds
from src.rules.geometry import side_of_line

LABEL = "red_light"


def _light_state_at(t_sec: float, roi_name: str, scene) -> str:
    """Nearest-in-time sample from scene.light_history[roi_name] to `t_sec`,
    same nearest-timestamp-lookup pattern src/rules/stop_line.py's
    _queue_size_at() already uses for speed lookups."""
    samples = scene.light_history.get(roi_name)
    if not samples:
        return "unknown"
    nearest = min(samples, key=lambda ts: abs(ts[0] - t_sec))
    return nearest[1]


def _crossing_within_segment(pt, p1, p2, margin: float = 0.15) -> bool:
    """True only if `pt` projects onto the stop-line SEGMENT p1->p2 (projection
    parameter t in [-margin, 1+margin]) -- NOT merely somewhere on its infinite
    extension. side_of_line() (used for the sign-flip crossing test below) is
    an UNBOUNDED line test, so without this guard a vehicle on a DIFFERENT road
    that happens to cross the infinite continuation of this stop line (e.g.
    right-side traffic near turn_right_line vs the left-road
    stop_line_main_road1) would be treated as crossing it. Such a vehicle
    projects to t far outside [0,1] and is correctly excluded here: a road with
    no stop line in its actual path is out of scope for red_light entirely, not
    just "no violation". Legitimate crossers of the real painted line project
    within [0,1], so this does NOT change how an in-segment stop line is
    evaluated."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    seg2 = dx * dx + dy * dy
    if seg2 <= 0:
        return False
    t = ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / seg2
    return -margin <= t <= 1.0 + margin


def detect(trajectories, scene) -> list[list]:
    if not scene.stop_lines or not scene.traffic_lights:
        return []
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)

    events: list[list] = []
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        pt_by_t = {p.t_sec: p for p in traj}

        prev_side: dict[str, float | None] = {sl.id: None for sl in scene.stop_lines}
        for t, _vx, _vy, _s in speeds(traj):
            pt = pt_by_t.get(t)
            if pt is None:
                continue
            for sl in scene.stop_lines:
                side_now = side_of_line((pt.cx, pt.cy), sl.p1, sl.p2)
                prev = prev_side[sl.id]
                crossed = (prev is not None and prev != 0 and side_now != 0
                           and (prev > 0) != (side_now > 0)
                           # scope to the actual painted segment, not the infinite line --
                           # excludes vehicles on other roads crossing the line's extension
                           and _crossing_within_segment((pt.cx, pt.cy), sl.p1, sl.p2))
                if crossed:
                    light = scene.nearest_traffic_light(sl)
                    if light is not None and _light_state_at(t, light.id, scene) == "red":
                        events.append([max(0.0, t - 0.5), t + 0.5, LABEL])
                prev_side[sl.id] = side_now
    return events

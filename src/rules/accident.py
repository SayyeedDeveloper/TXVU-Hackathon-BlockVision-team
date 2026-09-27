"""accident — two tracks converge at high closing speed, come into contact
(min distance < thresholds['accident_contact_px']), and both velocities
collapse to ~0 and STAY there for >= thresholds['accident_stall_min_sec'].

Heuristic, not a learned model: this catches the clean "high-speed collision
then both stop" signature but will miss low-speed contact or partial
occlusion during the collision (tracker often drops/re-IDs boxes on impact).
Per the README tips, the recommended next step is to re-score candidate
windows with a small learned clip classifier once labeled clips exist —
this function only produces candidates from trajectory geometry.
"""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track, speeds
from src.rules.geometry import distance

LABEL = "accident"


def detect(trajectories, scene) -> list[list]:
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    closing_thresh = scene.thresholds["accident_closing_speed_px_s"]
    contact_thresh = scene.thresholds["accident_contact_px"]
    stall_min_sec = scene.thresholds["accident_stall_min_sec"]
    stopped_speed = scene.thresholds["stopped_speed_px_s"]

    track_ids = [tid for tid, traj in by_track.items() if traj and traj[0].cls in VEHICLE_CLASSES]
    events: list[list] = []

    for i, tid_a in enumerate(track_ids):
        traj_a = by_track[tid_a]
        pts_a = {p.t_sec: p for p in traj_a}
        spd_a = {t: s for t, _, _, s in speeds(traj_a)}
        for tid_b in track_ids[i + 1:]:
            traj_b = by_track[tid_b]
            pts_b = {p.t_sec: p for p in traj_b}
            spd_b = {t: s for t, _, _, s in speeds(traj_b)}
            common_t = sorted(set(pts_a) & set(pts_b))
            if len(common_t) < 3:
                continue

            dists = [(t, distance((pts_a[t].cx, pts_a[t].cy), (pts_b[t].cx, pts_b[t].cy))) for t in common_t]

            for k in range(1, len(dists)):
                t0, d0 = dists[k - 1]
                t1, d1 = dists[k]
                dt = t1 - t0
                if dt <= 0:
                    continue
                closing_speed = (d0 - d1) / dt
                if d1 >= contact_thresh or closing_speed < closing_thresh:
                    continue
                # contact at high closing speed: check both stall out afterwards
                stall_end = t1
                for t, _d in dists[k:]:
                    sa = spd_a.get(t, 0.0)
                    sb = spd_b.get(t, 0.0)
                    if sa < stopped_speed and sb < stopped_speed:
                        stall_end = t
                    else:
                        break
                if stall_end - t1 >= stall_min_sec:
                    events.append([round(max(0.0, t0 - 0.5), 3), round(stall_end, 3), LABEL])
                    break

    events.sort(key=lambda e: e[0])
    return events

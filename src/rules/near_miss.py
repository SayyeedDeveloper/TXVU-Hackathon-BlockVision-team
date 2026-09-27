"""near_miss — two tracks (vehicle-vehicle or vehicle-pedestrian) whose paths
were converging (closing distance shrinking at >= thresholds['near_miss_converge_px_s'])
followed by a sudden deceleration or heading change from at least one of them,
without the two ever coming into contact (min distance stays above
thresholds['near_miss_contact_px'] — if it drops below that, it's scored as a
possible `accident`, not a near_miss; see accident.py).

Generic pairwise-trajectory geometry — no scene metadata required, so this is
fully implemented. O(n^2) over tracks, fine for a handful of tracks per clip;
if a video has heavy traffic, consider spatial bucketing before scaling up.
"""
from __future__ import annotations

from src.detect import PEDESTRIAN_CLASSES, VEHICLE_CLASSES, group_by_track, speeds
from src.rules.geometry import distance

LABEL = "near_miss"
DECEL_WINDOW_SEC = 1.0


def _relevant(cls: str) -> bool:
    return cls in VEHICLE_CLASSES or cls in PEDESTRIAN_CLASSES


def detect(trajectories, scene) -> list[list]:
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    converge_thresh = scene.thresholds["near_miss_converge_px_s"]
    contact_thresh = scene.thresholds["near_miss_contact_px"]
    decel_thresh = scene.thresholds["near_miss_decel_px_s2"] * DECEL_WINDOW_SEC

    track_ids = [tid for tid, traj in by_track.items() if traj and _relevant(traj[0].cls)]
    events: list[list] = []

    for i, tid_a in enumerate(track_ids):
        traj_a = by_track[tid_a]
        pts_a = {p.t_sec: p for p in traj_a}
        for tid_b in track_ids[i + 1:]:
            traj_b = by_track[tid_b]
            pts_b = {p.t_sec: p for p in traj_b}
            common_t = sorted(set(pts_a) & set(pts_b))
            if len(common_t) < 3:
                continue

            dists = [(t, distance((pts_a[t].cx, pts_a[t].cy), (pts_b[t].cx, pts_b[t].cy))) for t in common_t]
            min_dist = min(d for _, d in dists)
            if min_dist < contact_thresh:
                continue  # treated as contact -> candidate accident, not near_miss

            for (t0, d0), (t1, d1) in zip(dists, dists[1:]):
                dt = t1 - t0
                if dt <= 0:
                    continue
                closing_speed = (d0 - d1) / dt
                if closing_speed >= converge_thresh:
                    # look for a sharp deceleration on either track shortly after t1
                    sharp = False
                    for spd_series in (speeds(traj_a), speeds(traj_b)):
                        for j in range(len(spd_series) - 1):
                            ta, _, _, sa = spd_series[j]
                            if not (t1 - 0.5 <= ta <= t1 + DECEL_WINDOW_SEC):
                                continue
                            for tb, _, _, sb in spd_series[j + 1:]:
                                if tb - ta > DECEL_WINDOW_SEC:
                                    break
                                if sa - sb >= decel_thresh:
                                    sharp = True
                                    break
                            if sharp:
                                break
                        if sharp:
                            break
                    if sharp:
                        events.append([round(max(0.0, t0 - 0.5), 3), round(t1 + 0.5, 3), LABEL])
                        break  # one near_miss event per pair is enough

    events.sort(key=lambda e: e[0])
    return events

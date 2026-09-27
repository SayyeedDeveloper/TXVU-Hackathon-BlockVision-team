"""congestion — standstill/crawling traffic: at least `congestion_min_vehicles`
vehicles simultaneously below `congestion_speed_px_s`, sustained for
`congestion_min_sec`. Scene-wide (not split by lane/direction) since we don't
require per-lane direction grouping metadata for this one."""
from __future__ import annotations

from collections import defaultdict

from src.detect import VEHICLE_CLASSES, group_by_track, speeds

LABEL = "congestion"
BUCKET_SEC = 1.0


def detect(trajectories, scene) -> list[list]:
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    speed_thresh = scene.thresholds["congestion_speed_px_s"]
    min_vehicles = scene.thresholds["congestion_min_vehicles"]
    min_dur = scene.thresholds["congestion_min_sec"]

    slow_count_by_bucket: dict[int, int] = defaultdict(int)
    max_bucket = 0
    for traj in by_track.values():
        if not traj or traj[0].cls not in VEHICLE_CLASSES:
            continue
        seen_buckets = set()
        for t, _vx, _vy, s in speeds(traj):
            bucket = int(t // BUCKET_SEC)
            max_bucket = max(max_bucket, bucket)
            if s < speed_thresh and bucket not in seen_buckets:
                slow_count_by_bucket[bucket] += 1
                seen_buckets.add(bucket)

    events: list[list] = []
    run_start_bucket = None
    for b in range(max_bucket + 2):
        congested = slow_count_by_bucket.get(b, 0) >= min_vehicles
        if congested:
            if run_start_bucket is None:
                run_start_bucket = b
        else:
            if run_start_bucket is not None:
                start = run_start_bucket * BUCKET_SEC
                end = b * BUCKET_SEC
                if end - start >= min_dur:
                    events.append([start, end, LABEL])
                run_start_bucket = None
    return events

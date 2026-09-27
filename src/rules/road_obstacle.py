"""road_obstacle — a static non-vehicle, non-pedestrian object appearing and
persisting on the carriageway.

PARTIAL / TODO: COCO (what yolov8n is pretrained on) has no generic
"debris"/"obstacle" class — it only names specific objects (backpack,
suitcase, bench, dog, ...). VideoTracker's default `classes=` filter
(COCO_RELEVANT_CLASS_IDS in src/detect.py) doesn't even request these, so as
written this rule sees nothing. To get partial coverage: pass a wider
`classes=` list to VideoTracker (e.g. add COCO ids for backpack=24,
suitcase=28, bench=13, dog=16, ...) and re-run detect_events with that
tracker. True arbitrary-debris detection needs a dedicated
anomaly/obstacle detector (e.g. background-subtraction "unexpected static
blob" or a small fine-tuned model) — not implemented here; note this in the
README status checklist.

The persistence-on-roadway logic below IS real and will fire correctly for
whatever object classes the tracker is given.
"""
from __future__ import annotations

from src.detect import VEHICLE_CLASSES, group_by_track

LABEL = "road_obstacle"
NON_ROAD_USER_HINT = {"backpack", "suitcase", "bench", "dog", "cat", "handbag", "chair"}


def detect(trajectories, scene) -> list[list]:
    if not scene.lanes:
        return []
    by_track = trajectories if isinstance(trajectories, dict) else group_by_track(trajectories)
    speed_thresh = scene.thresholds["stopped_speed_px_s"]
    min_dur = scene.thresholds["obstacle_min_sec"]

    events: list[list] = []
    for traj in by_track.values():
        if not traj:
            continue
        cls = traj[0].cls
        if cls in VEHICLE_CLASSES or cls == "person" or cls not in NON_ROAD_USER_HINT:
            continue
        on_road_pts = [p for p in traj if scene.point_in_any_lane((p.cx, p.cy)) is not None]
        if len(on_road_pts) < 2:
            continue
        # roughly static: bbox center doesn't move far across the whole run
        cxs = [p.cx for p in on_road_pts]
        cys = [p.cy for p in on_road_pts]
        spread = ((max(cxs) - min(cxs)) ** 2 + (max(cys) - min(cys)) ** 2) ** 0.5
        duration = on_road_pts[-1].t_sec - on_road_pts[0].t_sec
        if spread < speed_thresh * max(duration, 1e-6) and duration >= min_dur:
            events.append([on_road_pts[0].t_sec, on_road_pts[-1].t_sec, LABEL])
    return events

"""fire_smoke — visible fire or smoke from a vehicle or on the road.

STUB: this needs either a small image/video classifier (fire/smoke has no
COCO-detectable "object" signature — it's a texture/color/motion pattern, not
a bounding-box-able thing an object detector was trained for) or a public
fire/smoke dataset fine-tune (e.g. a lightweight CNN on crops, or a frame-level
classifier run every N frames). Deliberately NOT faked with a placeholder
heuristic (e.g. "orange pixels") — that would silently produce garbage
predictions that look real but aren't grounded in anything. Leave this
returning [] until a real classifier is plugged in.

TODO: candidate approach once time allows — download a small open fire/smoke
image dataset, fine-tune a yolov8n-cls or a tiny CNN, run it on sampled crops
around detected vehicles + a coarse full-frame pass, threshold + temporal
smoothing into segments the same way the other rules do.
"""
from __future__ import annotations

LABEL = "fire_smoke"


def detect(trajectories, scene) -> list[list]:
    return []

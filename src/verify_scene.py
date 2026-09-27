"""src/verify_scene.py — draw SCENE's geometry on a real frame so you can
eyeball whether the first-pass coordinates in src/config.py actually line up
with the road. One-off inspection script, not part of the submission
pipeline (no ultralytics/torch needed, just cv2 + src.config).

Usage:
    python -m src.verify_scene
    python -m src.verify_scene --frame samples/frames/C3896/clean_frame.jpg --out samples/frames/C3896/overlay_check.jpg

Draws (all in pixel space, converted from SCENE's normalized coords against
the ACTUAL loaded frame's width/height, not SCENE.width/height — so this
still works if you point it at a differently-sized frame):
    lane polygons      -> green
    crossing polygons  -> blue
    stop lines         -> red (thicker)
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.config import SCENE

GREEN = (0, 200, 0)
BLUE = (255, 120, 0)
RED = (0, 0, 255)


def _norm_polygon(polygon_px: list[tuple[float, float]], scene_w: int, scene_h: int) -> list[tuple[float, float]]:
    """SCENE stores pixel coords scaled to SCENE.width/height; convert back to
    normalized [0,1] so we can re-scale against whatever frame we're drawing on."""
    return [(x / scene_w, y / scene_h) for x, y in polygon_px]


def draw_overlay(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    scene_w = SCENE.width or w
    scene_h = SCENE.height or h

    for lane in SCENE.lanes:
        pts_norm = _norm_polygon(lane.polygon, scene_w, scene_h)
        pts = np.array([[int(x * w), int(y * h)] for x, y in pts_norm], dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=GREEN, thickness=3)
        if len(pts) > 0:
            cv2.putText(out, lane.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

    for c in SCENE.crossings:
        pts_norm = _norm_polygon(c.polygon, scene_w, scene_h)
        pts = np.array([[int(x * w), int(y * h)] for x, y in pts_norm], dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=BLUE, thickness=3)
        if len(pts) > 0:
            cv2.putText(out, c.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2)

    for sl in SCENE.stop_lines:
        x1n, y1n = sl.p1[0] / scene_w, sl.p1[1] / scene_h
        x2n, y2n = sl.p2[0] / scene_w, sl.p2[1] / scene_h
        p1 = (int(x1n * w), int(y1n * h))
        p2 = (int(x2n * w), int(y2n * h))
        cv2.line(out, p1, p2, RED, 5)
        cv2.putText(out, sl.name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default="samples/frames/C3896/clean_frame.jpg")
    ap.add_argument("--out", default="samples/frames/C3896/overlay_check.jpg")
    args = ap.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        print(f"could not read {args.frame}")
        return 1

    overlay = draw_overlay(frame)
    cv2.imwrite(args.out, overlay)
    print(f"wrote {args.out} ({overlay.shape[1]}x{overlay.shape[0]})")
    print("green = lanes, blue = crossings, red = stop lines")
    print("REMINDER: SCENE in src/config.py is not visually confirmed yet — check this overlay before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""src/rules/traffic_light.py — traffic-light color classification from a
cropped ROI (src.config.TrafficLightROI), pure HSV hue thresholding, no
learned model. Used by src/rules/red_light.py to tell whether the signal was
actually red when a vehicle crossed its stop line -- replacing that rule's
earlier "queue of >=2 stopped vehicles" proxy heuristic (still used as-is by
src/rules/stop_line.py, untouched here) now that real color classification
is available.

classify_light_state() needs an actual video frame, which the scored rules
pipeline (src.rules.run_all_rules -> detect(trajectories, scene)) never sees.
So solution.detect_events() asks src/video_io.py's single decode pass for a
full-resolution crop around every traffic-light ROI (light_crop_box()), feeds
each crop to LightHistory.add(), and stores the result on
scene.light_history. red_light.py reads that instead of raw frames, keeping
its detect(trajectories, scene) signature identical to every other rule module.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque

import cv2
import numpy as np

# HSV hue ranges (OpenCV convention: H in [0, 180]). Red wraps around 0/180,
# so it needs two ranges; yellow and green each fit in one.
_RED_HUE_RANGES = [(0, 10), (170, 180)]
_YELLOW_HUE_RANGE = (15, 35)
_GREEN_HUE_RANGE = (40, 90)

# Thresholds tuned against samples/C3896.MP4's actual traffic light (a small,
# obliquely-viewed 3-lamp housing where each lit lamp is only ~1-3% of the ROI):
_MIN_SAT = 80     # KEEP HIGH -- this is the key guard. The lit lamp is strongly
                   # saturated (measured S ~= 200); daylight glare / washed-out
                   # housing is nearly gray (measured S ~= 5) and must be rejected,
                   # otherwise glare leaks in as false "green". Do NOT lower this.
_MIN_VALUE = 60   # ignore near-black pixels (unlit lamps, shadow). Lowered from 80
                   # to 60 to keep the dimmer edge pixels of the lit lamp (its V
                   # bottoms out around 60 in this footage).
_MIN_FRACTION = 0.005  # a lit lamp covers only ~0.5-12% of THIS small-lamp ROI, but
                        # the NON-lit colors sit at ~0.000 -- so classification is an
                        # argmax between the three colored fractions, gated by this
                        # small floor (0.5%) just to fall back to 'unknown' when the
                        # whole ROI is dark/between phases. Was 0.10, which the lit
                        # lamp could essentially never reach here -> always 'unknown'.

_VOTE_WINDOW = 5  # rolling majority-vote smoothing window, in classified frames, per ROI

# Per-ROI rolling history for majority-vote smoothing, keyed by TrafficLightROI.name.
# Module-level (not passed explicitly) because classify_light_state()'s signature was
# specified as (frame, roi) -> str -- see reset_light_history() to clear this between
# separate videos/runs (LightHistory() below calls it automatically per video).
_history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=_VOTE_WINDOW))


def reset_light_history() -> None:
    """Clear the rolling per-ROI smoothing state. Call before processing a
    new video so leftover state from a previous run/clip can't leak in."""
    _history.clear()


def _hue_fraction(hsv_roi: np.ndarray, hue_ranges: list[tuple[int, int]]) -> float:
    mask = np.zeros(hsv_roi.shape[:2], dtype=bool)
    h, s, v = hsv_roi[..., 0], hsv_roi[..., 1], hsv_roi[..., 2]
    for lo, hi in hue_ranges:
        mask |= (h >= lo) & (h <= hi)
    mask &= (s >= _MIN_SAT) & (v >= _MIN_VALUE)
    total = mask.size
    return float(mask.sum()) / total if total else 0.0


def classify_light_state(frame: np.ndarray, roi, offset: tuple[int, int] = (0, 0)) -> str:
    """Crop `roi.box` (pixel coords -- see SceneConfig.__post_init__) out of
    `frame`, classify its dominant color in HSV, and smooth over the last
    _VOTE_WINDOW calls for this roi.name via rolling majority vote.
    `offset` is the (x, y) of `frame`'s top-left corner in the full video
    frame, for when `frame` is itself a crop (see LightHistory).
    Returns 'red', 'yellow', 'green', or 'unknown'.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = roi.box
    x1, x2 = sorted((int(round(x1)) - offset[0], int(round(x2)) - offset[0]))
    y1, y2 = sorted((int(round(y1)) - offset[1], int(round(y2)) - offset[1]))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)

    if x2 <= x1 or y2 <= y1:
        raw = "unknown"
    else:
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        fractions = {
            "red": _hue_fraction(hsv, _RED_HUE_RANGES),
            "yellow": _hue_fraction(hsv, [_YELLOW_HUE_RANGE]),
            "green": _hue_fraction(hsv, [_GREEN_HUE_RANGE]),
        }
        best_color, best_frac = max(fractions.items(), key=lambda kv: kv[1])
        raw = best_color if best_frac >= _MIN_FRACTION else "unknown"

    hist = _history[roi.name]
    hist.append(raw)
    return Counter(hist).most_common(1)[0][0]


def light_crop_box(scene, margin_px: int = 8) -> tuple[float, float, float, float] | None:
    """Pixel box covering every scene.traffic_lights ROI (plus a margin), or
    None when the scene has no lights -- what src/video_io.py should crop at
    full resolution alongside the downscaled detector frames."""
    if not scene.traffic_lights:
        return None
    boxes = [tl.box for tl in scene.traffic_lights]
    return (min(min(b[0], b[2]) for b in boxes) - margin_px, min(min(b[1], b[3]) for b in boxes) - margin_px,
            max(max(b[0], b[2]) for b in boxes) + margin_px, max(max(b[1], b[3]) for b in boxes) + margin_px)


class LightHistory:
    """Accumulates {roi_name: [(t_sec, state), ...]} -- the shape
    SceneConfig.light_history expects -- one sampled frame at a time, from
    full-resolution crops whose top-left corner is `offset` in the video."""

    def __init__(self, scene, offset: tuple[int, int]):
        reset_light_history()
        self.scene = scene
        self.offset = offset
        self.history: dict[str, list[tuple[float, str]]] = {tl.name: [] for tl in scene.traffic_lights}

    def add(self, crop: np.ndarray, t_sec: float) -> None:
        for tl in self.scene.traffic_lights:
            self.history[tl.name].append((t_sec, classify_light_state(crop, tl, self.offset)))

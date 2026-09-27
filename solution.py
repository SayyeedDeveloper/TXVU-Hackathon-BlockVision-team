"""solution.py — the ONLY file a team has to implement.

The organizers' harness (run_submission.py) imports this module and calls:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B (optional)

Keep the names and signatures exactly as they are. Everything else — models,
tracking, rules, helper modules under src/ — is up to you.

Pipeline: VideoTracker (src/detect.py, ultralytics YOLO + built-in tracker) ->
per-track trajectories -> src/rules/*.py (one rule function per class, driven
by src/config.py's SCENE geometry) -> src/postprocess.py boundary cleanup.
Part B (src/risk_estimator.py) reuses the same detector but runs strictly
causally, frame by frame, independent of Part A.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import scene_for
from src.detect import DECODE_WIDTH, VideoTracker, group_by_track
from src.postprocess import postprocess_events
from src.risk_estimator import RiskEstimatorImpl
from src.rules import run_all_rules
from src.rules.traffic_light import LightHistory, light_crop_box
from src.video_io import even_box, iter_sampled_frames, probe

# Official class ids (14). See the task description for definitions and
# start/end conventions. Remove entries you never predict; never add.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

# Anticipation horizon used by the metric (seconds). step() should return
# P(an `accident` starts within the next RISK_HORIZON_SEC seconds).
RISK_HORIZON_SEC = 5.0

# Detection/tracking config. The weights file ships in the repo (weights/) and is
# loaded by absolute path: a bare "yolov8n.pt" would make ultralytics try to
# download it, which fails on the offline judge machine. DETECT_STRIDE trades
# recall/boundary precision for speed (see src/video_io.py for the timing).
DETECT_MODEL = str(Path(__file__).resolve().parent / "weights" / "yolov8n.pt")
DETECT_STRIDE = 3


def _select_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection.

    Args:
        video_path: path to one .mp4 file.

    Returns:
        A list of events, each ``[start_sec, end_sec, label]``.
    """
    meta = probe(video_path)
    scene = scene_for(video_path)
    tracker = VideoTracker(model_path=DETECT_MODEL, device=_select_device())

    # ONE decode pass (src/video_io.py) feeds both the tracker (downscaled frames)
    # and the traffic-light classifier (full-resolution crop of the signal head).
    light_box = light_crop_box(scene)
    crop_box = even_box(light_box, meta) if light_box else None
    lights = LightHistory(scene, offset=crop_box[:2]) if crop_box else None

    points = []
    for sample in iter_sampled_frames(video_path, DETECT_STRIDE, DECODE_WIDTH, crop_box, meta):
        points.extend(tracker.track_sampled(sample)[1])
        if lights is not None:
            lights.add(sample.crop, sample.t_sec)
    trajectories = group_by_track(points)
    scene.light_history = lights.history if lights is not None else {}

    raw_events = run_all_rules(trajectories, scene, meta.duration)
    return postprocess_events(raw_events, meta.duration)


class RiskEstimator:
    """Part B — causal accident anticipation (optional, bonus).

    Delegates to src.risk_estimator.RiskEstimatorImpl, which keeps its own
    detector/tracker state and never looks at frames from the future or at
    Part A's output (see that module's docstring for why).
    """

    def __init__(self):
        self._impl = RiskEstimatorImpl(model_path=DETECT_MODEL, device=_select_device())

    def reset(self, meta: dict) -> None:
        """meta = {"video_id": str, "fps": float, "width": int, "height": int, "n_frames": int}"""
        self._impl.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """BGR uint8 frame -> P(accident within RISK_HORIZON_SEC s)."""
        return self._impl.step(frame, t_sec)

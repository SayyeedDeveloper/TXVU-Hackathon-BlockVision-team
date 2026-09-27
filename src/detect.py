"""src/detect.py — YOLO detection + tracking, producing per-track trajectories.

Runnable two ways:
  * Local (dev Mac, no GPU): a tiny CPU smoke test on a handful of frames with
    yolov8n, e.g.:
        python -m src.detect --video samples/clip1.mp4 --device cpu --max-frames 30
  * Colab/Kaggle (T4 GPU): full video, GPU device:
        python -m src.detect --video /content/clip1.mp4 --device cuda --model yolov8n.pt

Ultralytics/torch are imported lazily inside VideoTracker so that
`import solution` (and everything else in this repo) never fails just because
those heavy deps aren't installed yet — the requirements.txt lists them, but
we don't want an import-time crash to be the first thing a teammate hits.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

SEED = 1234


def set_determinism(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # tracker/NMS ops aren't all deterministic-safe
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class TrackPoint:
    frame_idx: int
    t_sec: float
    track_id: int
    cls: str
    cx: float
    cy: float
    w: float
    h: float
    conf: float


def group_by_track(points: list[TrackPoint]) -> dict[int, list[TrackPoint]]:
    by_id: dict[int, list[TrackPoint]] = defaultdict(list)
    for p in points:
        by_id[p.track_id].append(p)
    for pts in by_id.values():
        pts.sort(key=lambda p: p.frame_idx)
    return dict(by_id)


def speeds(traj: list[TrackPoint]) -> list[tuple[float, float, float, float]]:
    """Per-consecutive-pair (t_sec, vx, vy, speed) in px/s, aligned to the later point."""
    out = []
    for prev, cur in zip(traj, traj[1:]):
        dt = cur.t_sec - prev.t_sec
        if dt <= 0:
            continue
        vx = (cur.cx - prev.cx) / dt
        vy = (cur.cy - prev.cy) / dt
        out.append((cur.t_sec, vx, vy, (vx * vx + vy * vy) ** 0.5))
    return out


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
PEDESTRIAN_CLASSES = {"person"}
# COCO ids for the above, used to restrict YOLO's `classes=` filter to what the
# rules actually consume (skips e.g. traffic lights/benches at detect time,
# unless road_obstacle explicitly widens this — see src/rules/road_obstacle.py).
COCO_RELEVANT_CLASS_IDS = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

# Width frames are decoded at for the detector (see src/video_io.py). YOLO letterboxes
# to `imgsz` anyway, so this only needs to stay >= imgsz; raise it with imgsz.
DECODE_WIDTH = 1280


class VideoTracker:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        conf: float = 0.25,
        imgsz: int = 640,
        tracker: str = "bytetrack.yaml",
        classes: list[int] | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.classes = classes if classes is not None else COCO_RELEVANT_CLASS_IDS
        self._model = None

    def _load(self):
        if self._model is None:
            set_determinism()
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def track_frame(self, frame: np.ndarray):
        """Run tracking on a single BGR frame, maintaining track state across calls
        (persist=True). Used both by run() and by RiskEstimatorImpl for causal,
        frame-at-a-time inference."""
        model = self._load()
        return model.track(
            frame, persist=True, device=self.device, conf=self.conf,
            imgsz=self.imgsz, classes=self.classes, tracker=self.tracker, verbose=False,
        )[0]

    def result_to_points(self, result, frame_idx: int, t_sec: float, scale: float = 1.0) -> list[TrackPoint]:
        """`scale` maps the tracked frame's pixels back to the original video's
        (e.g. 3.0 for a 1280px frame from a 3840px video), so trajectories stay
        in the same pixel space as src/config.py's scene geometry and thresholds."""
        pts = []
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return pts
        xywh = boxes.xywh.cpu().numpy() * scale
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        names = result.names
        for (cx, cy, w, h), tid, c, conf in zip(xywh, ids, clss, confs):
            pts.append(TrackPoint(frame_idx, t_sec, int(tid), names[int(c)],
                                   float(cx), float(cy), float(w), float(h), float(conf)))
        return pts

    def track_sampled(self, sample) -> tuple[object, list[TrackPoint]]:
        """Track one src.video_io.SampledFrame -> (ultralytics result, points in
        original-video pixels)."""
        result = self.track_frame(sample.image)
        return result, self.result_to_points(result, sample.frame_idx, sample.t_sec, sample.scale)

    def run(
        self,
        video_path: str,
        stride: int = 1,
        max_frames: int | None = None,
        debug_video_path: str | None = None,
        decode_width: int = DECODE_WIDTH,
    ) -> list[TrackPoint]:
        """Process a whole video file (CLI/debug use; solution.detect_events runs
        the same loop itself so it can share the decode with the traffic-light
        classifier). `stride` skips frames for speed — track IDs are only updated
        on sampled frames."""
        import cv2

        from src.video_io import iter_sampled_frames, probe

        meta = probe(video_path)
        writer = None
        all_points: list[TrackPoint] = []
        for sampled, sample in enumerate(iter_sampled_frames(video_path, stride, decode_width, meta=meta), 1):
            result, points = self.track_sampled(sample)
            all_points.extend(points)
            if debug_video_path:
                if writer is None:
                    h, w = sample.image.shape[:2]
                    writer = cv2.VideoWriter(debug_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             meta.fps / max(1, stride), (w, h))
                writer.write(result.plot())
            if max_frames is not None and sampled >= max_frames:
                break
        if writer is not None:
            writer.release()
        return all_points


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default="cpu", help="'cpu' locally, 'cuda' or '0' on Colab")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None, help="cap sampled frames (smoke test)")
    ap.add_argument("--debug-video", default=None, help="write an annotated .mp4 with boxes drawn")
    args = ap.parse_args()

    tracker = VideoTracker(model_path=args.model, device=args.device)
    points = tracker.run(args.video, stride=args.stride, max_frames=args.max_frames,
                          debug_video_path=args.debug_video)
    by_track = group_by_track(points)
    print(f"{len(points)} detections across {len(by_track)} tracks")
    for tid, traj in list(by_track.items())[:5]:
        print(f"  track {tid}: {traj[0].cls}, {len(traj)} pts, "
              f"t=[{traj[0].t_sec:.2f}, {traj[-1].t_sec:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

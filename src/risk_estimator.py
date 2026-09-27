"""src/risk_estimator.py — causal, frame-by-frame accident-risk scoring.

Strictly causal: `step(frame, t_sec)` only ever sees the current frame plus
whatever state this class has accumulated from PAST calls (a per-track history
buffer). It never opens the video file itself and never reads Part A's
`detect_events()` output — both would leak future information the harness
doesn't give it (and reusing Part A output is an explicit rule violation per
the task spec).

Signal: pairwise time-to-collision (TTC) between currently-visible vehicle/
pedestrian tracks, boosted by sudden-deceleration events (braking hard is
itself a precursor). Calibrated so an imminent collision (TTC ~ RISK_HORIZON)
maps near 1.0 and TTC >> horizon or diverging paths map near 0 — a flat
constant score earns 0 under the metric's chance-normalised AP, so the
tracker-down/no-detections fallback must decay toward 0, not hold last value
forever.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

RISK_HORIZON_SEC = 5.0


class RiskEstimatorImpl:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        stride: int = 3,  # the harness's own cv2 decode already costs ~1.2x real time on
                          # this 4K 10-bit footage; detecting on every 3rd frame (~10 Hz) keeps
                          # Part A + Part B inside the 3x budget. Measure before lowering.
        history_len: int = 15,
        ema_alpha: float = 0.5,
    ):
        self.model_path = model_path
        self.device = device
        self.stride = stride
        self.history_len = history_len
        self.ema_alpha = ema_alpha
        self._tracker = None  # lazy VideoTracker, built in reset() so device/model can be swapped
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_len))
        self._frame_idx = 0
        self.last_score = 0.0
        self.meta = {}

    def reset(self, meta: dict) -> None:
        from src.detect import VideoTracker

        self.meta = meta
        self._tracker = VideoTracker(model_path=self.model_path, device=self.device)
        self._history = defaultdict(lambda: deque(maxlen=self.history_len))
        self._frame_idx = 0
        self.last_score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        run_detector = (self._frame_idx % self.stride == 0)
        self._frame_idx += 1
        if not run_detector:
            self.last_score *= 0.9  # decay toward 0 between sampled frames, never hold flat
            return self.last_score

        result = self._tracker.track_frame(frame)
        points = self._tracker.result_to_points(result, self._frame_idx, t_sec)
        for p in points:
            self._history[p.track_id].append((p.t_sec, p.cx, p.cy))

        risk = self._risk_from_history(t_sec)
        self.last_score = self.ema_alpha * risk + (1 - self.ema_alpha) * self.last_score
        return float(min(1.0, max(0.0, self.last_score)))

    # ---- internals ----------------------------------------------------
    def _velocity(self, hist: deque) -> tuple[float, float, float]:
        if len(hist) < 2:
            return 0.0, 0.0, 0.0
        t0, x0, y0 = hist[-2]
        t1, x1, y1 = hist[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0, 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt, dt

    def _decel_spike(self, hist: deque) -> float:
        """0..1 signal for a sudden deceleration in this track's recent history."""
        if len(hist) < 3:
            return 0.0
        speeds = []
        for (t0, x0, y0), (t1, x1, y1) in zip(hist, list(hist)[1:]):
            dt = t1 - t0
            if dt > 0:
                speeds.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt)
        if len(speeds) < 2:
            return 0.0
        drop = max(0.0, speeds[0] - speeds[-1])
        ref = max(speeds[0], 1e-6)
        return min(1.0, drop / ref)

    def _risk_from_history(self, t_sec: float) -> float:
        active = {tid: hist for tid, hist in self._history.items()
                  if hist and t_sec - hist[-1][0] < 1.0}
        if len(active) < 2:
            # single-track hard braking is still a weak signal
            return max((self._decel_spike(h) for h in active.values()), default=0.0) * 0.3

        ttc_risk = 0.0
        decel_risk = 0.0
        ids = list(active.keys())
        for i, tid_a in enumerate(ids):
            hist_a = active[tid_a]
            xa, ya = hist_a[-1][1], hist_a[-1][2]
            vxa, vya, _ = self._velocity(hist_a)
            for tid_b in ids[i + 1:]:
                hist_b = active[tid_b]
                xb, yb = hist_b[-1][1], hist_b[-1][2]
                vxb, vyb, _ = self._velocity(hist_b)

                dist = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
                dvx, dvy = vxa - vxb, vya - vyb
                closing_speed = ((dvx ** 2 + dvy ** 2) ** 0.5)
                # is distance actually shrinking? project relative velocity onto the line between them
                if dist > 1e-6:
                    rx, ry = (xb - xa) / dist, (yb - ya) / dist
                    approach_speed = -(dvx * rx + dvy * ry)
                else:
                    approach_speed = closing_speed
                if approach_speed <= 1e-6 or dist <= 1e-6:
                    continue
                ttc = dist / approach_speed
                if ttc < RISK_HORIZON_SEC:
                    ttc_risk = max(ttc_risk, 1.0 - ttc / RISK_HORIZON_SEC)

            decel_risk = max(decel_risk, self._decel_spike(hist_a))

        return min(1.0, 0.7 * ttc_risk + 0.3 * decel_risk)

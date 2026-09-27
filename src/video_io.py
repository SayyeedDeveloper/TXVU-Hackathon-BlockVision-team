"""src/video_io.py — one fast, sampled decode pass for Part A.

The camera records 4K H.264 High 4:2:2 10-bit. Converting that to BGR is the
bottleneck: cv2.VideoCapture.read() does it single-threaded at ~25 fps, slower
than real time, and NVDEC can't decode 4:2:2 H.264 at all. So Part A decodes
with ffmpeg (bundled by imageio-ffmpeg) instead, which in one process:
  * decodes multi-threaded,
  * drops all but every `stride`-th frame BEFORE colour conversion,
  * downscales the kept frames to `width` px for the detector,
  * optionally cuts a full-resolution crop (e.g. the traffic-light head,
    which is only ~20 px wide at 4K and would blur away when downscaled)
    and stacks it under the frame, so both come out of one pipe.
Measured on C3905 (RTX 5060 laptop): ~95 source fps = 3.2x real time at
width=1280, stride=3, vs ~25 fps for cv2 read().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class VideoMeta:
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


@dataclass
class SampledFrame:
    frame_idx: int           # index in the ORIGINAL video
    t_sec: float
    image: np.ndarray        # BGR, downscaled to (out_h, out_w)
    scale: float             # original px per `image` px (multiply detections by it)
    crop: np.ndarray | None  # BGR full-resolution crop of `crop_box`, if requested


def probe(video_path: str) -> VideoMeta:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    meta = VideoMeta(
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return meta


def even_box(box: tuple[float, float, float, float], meta: VideoMeta) -> tuple[int, int, int, int]:
    """Pixel box (x1, y1, x2, y2) snapped outward to even coordinates and clipped
    to the frame: ffmpeg's crop rounds odd offsets on 4:2:2 input, which would
    silently shift the crop by a pixel."""
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1) // 2 * 2), max(0, int(y1) // 2 * 2)
    x2, y2 = min(meta.width, -(-int(np.ceil(x2)) // 2) * 2), min(meta.height, -(-int(np.ceil(y2)) // 2) * 2)
    return x1, y1, x2, y2


def iter_sampled_frames(
    video_path: str,
    stride: int,
    width: int,
    crop_box: tuple[int, int, int, int] | None = None,
    meta: VideoMeta | None = None,
) -> Iterator[SampledFrame]:
    """Yield every `stride`-th frame, downscaled to `width` px wide (aspect kept),
    plus a full-resolution crop of `crop_box` (from even_box()) when given."""
    import subprocess

    import imageio_ffmpeg

    meta = meta or probe(video_path)
    out_w = min(width, meta.width) // 2 * 2
    out_h = round(meta.height * out_w / meta.width / 2) * 2
    select = f"select='not(mod(n\\,{stride}))'"
    scale_filter = f"scale={out_w}:{out_h}:flags=bilinear"

    if crop_box is None:
        filters = ["-vf", f"{select},{scale_filter}"]
        crop_w = crop_h = 0
    else:
        x1, y1, x2, y2 = crop_box
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w > out_w:
            raise ValueError(f"crop {crop_w}px wider than output frame {out_w}px")
        filters = ["-filter_complex",
                   f"[0:v]{select},split=2[a][b];[a]{scale_filter}[s];"
                   f"[b]crop={crop_w}:{crop_h}:{x1}:{y1},pad={out_w}:{crop_h}:0:0[c];"
                   f"[s][c]vstack=inputs=2[out]",
                   "-map", "[out]"]

    rows = out_h + crop_h
    frame_bytes = out_w * rows * 3
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", video_path, *filters,
           "-an", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4)
    try:
        k = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, np.uint8).reshape(rows, out_w, 3)
            frame_idx = k * stride
            yield SampledFrame(
                frame_idx=frame_idx,
                t_sec=frame_idx / meta.fps,
                image=arr[:out_h],
                scale=meta.width / out_w,
                crop=arr[out_h:, :crop_w] if crop_box is not None else None,
            )
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.wait()
    if k == 0:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}: {err}")

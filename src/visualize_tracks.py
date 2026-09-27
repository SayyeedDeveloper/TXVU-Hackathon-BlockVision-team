"""src/visualize_tracks.py — tracked-detections video + a vehicle density
heatmap, built directly from VideoTracker's per-track trajectory output.

Standalone visualization/demo tool: consumes src.detect.VideoTracker's
TrackPoint output only (frame_idx, t_sec, track_id, class, cx, cy, w, h,
conf). Does not modify detect.py, postprocess.py, or config.py, and is not
wired into solution.py or src/rules/ — this has no bearing on the actual
submission pipeline.

Usage:
    python -m src.visualize_tracks samples/C3896.MP4 --output-dir samples/frames/C3896 --max-frames 60 --device cpu
    # full video on Colab/Kaggle:
    python -m src.visualize_tracks /content/C3896.MP4 --output-dir /content/out --device cuda

Outputs (written into --output-dir):
    tracked_output.mp4   annotated video: a box + "#<track_id> <class>" label
                          for every detection, frame by frame (all detected
                          classes, not just vehicles — a visual sanity check
                          that tracking is working, same spirit as
                          points_check.jpg but over time instead of one frame)
    heatmap.jpg           VEHICLE-only density heatmap (cars/trucks/buses/
                           motorcycles), overlaid on a representative frame —
                           reuses an already-extracted
                           samples/frames/<video>/frame_*.jpg if present,
                           otherwise falls back to the video's own first frame

Dependencies: only cv2 + numpy (already in requirements.txt). No matplotlib —
cv2.applyColorMap has perceptually-uniform, non-rainbow colormaps built in
(VIRIDIS/MAGMA), so there was no need to add a new dependency for this.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.detect import VEHICLE_CLASSES, TrackPoint, VideoTracker

BOX_COLOR = (0, 200, 255)         # BGR; orange, distinct from the rules'/verify_scene's green/blue/red
TEXT_COLOR = (255, 255, 255)
HEATMAP_GRID_DIVISOR = 16          # accumulate density at ~1/16 resolution, then blur + upscale for display
HEATMAP_ALPHA = 0.55                # overlay opacity
HEATMAP_MIN_DISPLAY = 0.02           # don't tint essentially-untouched cells
COLORMAP = cv2.COLORMAP_VIRIDIS       # perceptually uniform; NOT jet/rainbow, per requirements


def draw_tracks(frame: np.ndarray, points: list[TrackPoint]) -> np.ndarray:
    """Box + '#<track_id> <class>' label for every detection in this frame."""
    out = frame.copy()
    for p in points:
        x1, y1 = int(p.cx - p.w / 2), int(p.cy - p.h / 2)
        x2, y2 = int(p.cx + p.w / 2), int(p.cy + p.h / 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"#{p.track_id} {p.cls}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(out, label, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    return out


def accumulate_density(
    grid: np.ndarray, points: list[TrackPoint], grid_w: int, grid_h: int, img_w: int, img_h: int
) -> None:
    """In place: +1 to every grid cell covered by each VEHICLE's bbox
    footprint this frame (not just its center point, for a denser/more
    informative heatmap on short smoke-test clips)."""
    sx, sy = grid_w / img_w, grid_h / img_h
    for p in points:
        if p.cls not in VEHICLE_CLASSES:
            continue
        x1 = max(0, int((p.cx - p.w / 2) * sx))
        x2 = min(grid_w, int((p.cx + p.w / 2) * sx) + 1)
        y1 = max(0, int((p.cy - p.h / 2) * sy))
        y2 = min(grid_h, int((p.cy + p.h / 2) * sy) + 1)
        if x2 > x1 and y2 > y1:
            grid[y1:y2, x1:x2] += 1.0


def render_heatmap(grid: np.ndarray, n_sampled_frames: int, base_frame: np.ndarray) -> np.ndarray:
    """Normalize by sampled-frame count FIRST (fraction of sampled frames each
    cell had a vehicle over it — a density that doesn't grow just because the
    clip is longer / more frames were sampled), then separately max-normalize
    purely for colormap contrast. The two normalizations are different steps
    for different reasons: the first makes the number meaningful, the second
    only makes it visible."""
    h, w = base_frame.shape[:2]
    density = grid / max(1, n_sampled_frames)
    density = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=max(1.0, w / 150))

    max_val = float(density.max())
    display = density / max_val if max_val > 1e-9 else density

    heat_u8 = np.clip(display * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, COLORMAP)
    mask = (display > HEATMAP_MIN_DISPLAY).astype(np.float32)[..., None]
    overlay = (base_frame.astype(np.float32) * (1 - HEATMAP_ALPHA * mask)
               + colored.astype(np.float32) * (HEATMAP_ALPHA * mask))
    return overlay.astype(np.uint8)


def _find_representative_frame(output_dir: Path, fallback: np.ndarray) -> np.ndarray:
    candidates = sorted(output_dir.glob("frame_*.jpg"))
    if not candidates:
        print(f"no frame_*.jpg found in {output_dir}, using the video's own first frame instead")
        return fallback
    pick = candidates[len(candidates) // 2]
    img = cv2.imread(str(pick))
    if img is None:
        print(f"could not read {pick}, using the video's own first frame instead")
        return fallback
    print(f"using representative frame: {pick}")
    return img


def run(
    video_path: str,
    output_dir: str,
    max_frames: int | None,
    stride: int,
    model: str,
    device: str,
) -> int:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open {video_path}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video_path}: {img_w}x{img_h} @ {fps:.2f} fps, device={device}, model={model}, stride={stride}"
          + (f", max_frames={max_frames}" if max_frames else ""))

    tracker = VideoTracker(model_path=model, device=device)

    tracked_video_path = out_dir / "tracked_output.mp4"
    writer = cv2.VideoWriter(str(tracked_video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps / max(1, stride), (img_w, img_h))

    grid_w = max(1, img_w // HEATMAP_GRID_DIVISOR)
    grid_h = max(1, img_h // HEATMAP_GRID_DIVISOR)
    density_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    first_frame: np.ndarray | None = None
    frame_idx = 0
    sampled = 0
    vehicle_track_ids: set[int] = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first_frame is None:
            first_frame = frame.copy()
        if frame_idx % stride == 0:
            t_sec = frame_idx / fps
            result = tracker.track_frame(frame)
            points = tracker.result_to_points(result, frame_idx, t_sec)
            vehicle_track_ids.update(p.track_id for p in points if p.cls in VEHICLE_CLASSES)

            writer.write(draw_tracks(frame, points))
            accumulate_density(density_grid, points, grid_w, grid_h, img_w, img_h)

            sampled += 1
            if max_frames is not None and sampled >= max_frames:
                break
        frame_idx += 1
    cap.release()
    writer.release()

    print(f"processed {sampled} sampled frame(s), {len(vehicle_track_ids)} distinct vehicle track(s)")
    size_kb = tracked_video_path.stat().st_size / 1024
    print(f"wrote {tracked_video_path} ({size_kb:.0f} KB)")

    if first_frame is None:
        print("no frames read from the video at all, skipping heatmap")
        return 1
    base_frame = _find_representative_frame(out_dir, fallback=first_frame)
    if base_frame.shape[:2] != (img_h, img_w):
        base_frame = cv2.resize(base_frame, (img_w, img_h))

    heatmap = render_heatmap(density_grid, sampled, base_frame)
    heatmap_path = out_dir / "heatmap.jpg"
    cv2.imwrite(str(heatmap_path), heatmap)
    size_kb = heatmap_path.stat().st_size / 1024
    print(f"wrote {heatmap_path} ({size_kb:.0f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to a source .mp4")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="cap sampled frames (CPU smoke test)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default="cpu", help="'cpu' locally, 'cuda' on Colab/Kaggle")
    args = ap.parse_args()
    return run(args.video, args.output_dir, args.max_frames, args.stride, args.model, args.device)


if __name__ == "__main__":
    raise SystemExit(main())

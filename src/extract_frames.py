"""src/extract_frames.py — pull a handful of representative frames from each
sample video for manual scene inspection (lane layout, stop lines, crossings)
since samples/camera.md doesn't exist in this year's kit.

Inspection-only: not part of the detect_events()/RiskEstimator pipeline, no
ultralytics/torch dependency. Uses only cv2 (already in requirements.txt).

Usage:
    python -m src.extract_frames
    python -m src.extract_frames --videos samples --out samples/frames --n-frames 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

DEFAULT_FRACTIONS = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85)
MAX_WIDTH = 1280


def extract_one(video_path: Path, out_dir: Path, fractions=DEFAULT_FRACTIONS, max_width=MAX_WIDTH) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, frac in enumerate(fractions):
        idx = min(n_frames - 1, max(0, int(frac * n_frames)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved += 1
    cap.release()

    return {
        "video": video_path.name,
        "duration": duration,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "n_frames_extracted": saved,
        "out_dir": str(out_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="samples", help="folder with .mp4/.MP4 files")
    ap.add_argument("--out", default="samples/frames", help="output root folder")
    ap.add_argument("--n-frames", type=int, default=len(DEFAULT_FRACTIONS),
                     help="how many evenly-spaced fractions to sample (uses the first N of the default set, or an even spread if > default)")
    ap.add_argument("--max-width", type=int, default=MAX_WIDTH)
    args = ap.parse_args()

    src = Path(args.videos)
    out_root = Path(args.out)
    videos = sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
    if not videos:
        print(f"no .mp4/.MP4 files found in {src}")
        return 1

    if args.n_frames == len(DEFAULT_FRACTIONS):
        fractions = DEFAULT_FRACTIONS
    else:
        n = max(1, args.n_frames)
        fractions = tuple((i + 1) / (n + 1) for i in range(n))

    rows = []
    for video_path in videos:
        row = extract_one(video_path, out_root / video_path.stem, fractions=fractions, max_width=args.max_width)
        rows.append(row)

    print(f"\n{'video':<14}{'duration':>10}{'fps':>8}{'resolution':>14}{'frames':>9}  out_dir")
    for r in rows:
        print(f"{r['video']:<14}{r['duration']:>9.1f}s{r['fps']:>8.2f}{r['resolution']:>14}{r['n_frames_extracted']:>9}  {r['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

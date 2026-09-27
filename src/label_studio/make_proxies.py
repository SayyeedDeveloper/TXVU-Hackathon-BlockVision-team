"""src/label_studio/make_proxies.py — browser-friendly copies of the sample videos for Label Studio.

The raw samples are 4K H.264 at ~6 GB each: a browser can't scrub them smoothly
and Label Studio's upload limit rejects them. This re-encodes each one to 960px
wide H.264, same frame rate and frame count as the original (so a Label Studio
frame number maps 1:1 to an original frame), with a keyframe every second for
fast seeking, and writes index.json with each original's name, fps and duration
for src/label_studio/to_ground_truth.py.

Runs in the separate Label Studio venv (needs imageio-ffmpeg + opencv), not the
submission venv:
    python -m src.label_studio.make_proxies --videos samples --out samples/proxies
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV"}


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "n_frames": n_frames, "duration": round(n_frames / fps, 3)}


def make_proxy(src: Path, dst: Path, width: int, fps: float) -> None:
    # CPU on purpose: the samples are H.264 High 4:2:2 10-bit, which NVDEC can't
    # decode on any NVIDIA GPU, and decoding is the bottleneck. Measured on C3905:
    # NVENC encode 28.5 s vs libx264 30.8 s, with a 2.4x larger file.
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-vf", f"scale={width}:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-g", str(round(fps)),  # keyframe every ~1 s -> fast, accurate seeking in the browser
        "-an", "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="samples")
    ap.add_argument("--out", default="samples/proxies")
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    for src in sorted(p for p in Path(args.videos).iterdir() if p.suffix in VIDEO_EXTS):
        meta = probe(src)
        dst = out_dir / f"{src.stem}.mp4"
        if dst.exists() and index.get(src.stem, {}).get("n_frames") == meta["n_frames"]:
            print(f"[{src.name}] proxy exists, skipping")
        else:
            print(f"[{src.name}] {meta['duration']:.1f}s @ {meta['fps']:.2f} fps -> {dst}")
            make_proxy(src, dst, args.width, meta["fps"])
        index[src.stem] = {"source": src.name, "proxy": dst.name, **meta}
        index_path.write_text(json.dumps(index, indent=1))

    print(f"wrote {index_path} ({len(index)} videos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
